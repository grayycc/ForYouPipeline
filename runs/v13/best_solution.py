#!/usr/bin/env python3
"""
Iteration 1: FM ensemble (5 seeds) with within-user rank averaging +
Bayesian-smoothed per-video long_view rate as a bucketed feature.

Changes from baseline (iteration 0):
1. Compute Bayesian-smoothed per-video long_view rate from training rows only,
   bucket into 10 quantile bins, add as an extra FM feature field.
2. Train 5 FM models with seeds [args.seed+0 .. args.seed+4], convert each
   model's scores to within-user fractional ranks, average ranks as final score.
"""

import argparse
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--out_dir', required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--train_split', default='train', choices=['train', 'train+valid'])
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# Make sure kit/ is on the path
kit_dir = os.path.join(os.path.dirname(__file__), 'kit')
if os.path.isdir(kit_dir) and kit_dir not in sys.path:
    sys.path.insert(0, kit_dir)

from data import load, encode, LABEL
from evaluate import evaluate
from submit import write_submission
from baseline import FM

try:
    from unbiased import load_random_valid, encode_like_train, unbiased_primary
    HAS_UNBIASED = True
except ImportError:
    HAS_UNBIASED = False

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
splits = load(args.data_dir)

# ---------------------------------------------------------------------------
# Build Bayesian-smoothed per-video long_view rate from TRAINING rows only
# ---------------------------------------------------------------------------
# Each row in splits['train'] is a tuple:
#   (date, user_id, video_id, author_id, tab, duration_ms, long_view)
# Index:   0       1          2           3          4     5             6

def build_video_ctr_feature(train_rows, target_rows, n_buckets=10):
    """
    Compute Bayesian-smoothed per-video long_view rate from train_rows.
    Bucket into n_buckets quantile bins.
    Returns a numpy int32 array of bucket indices for each row in target_rows,
    plus the offset to add to turn it into a valid embedding index.
    
    Returns (bucket_array, n_buckets) where bucket values are in [0, n_buckets).
    """
    # Accumulate counts from training
    video_pos = {}
    video_cnt = {}
    for row in train_rows:
        vid = row[2]
        label = row[6]
        video_pos[vid] = video_pos.get(vid, 0) + label
        video_cnt[vid] = video_cnt.get(vid, 0) + 1

    # Global prior
    total_pos = sum(video_pos.values())
    total_cnt = sum(video_cnt.values())
    global_rate = total_pos / total_cnt if total_cnt > 0 else 0.3

    # Bayesian smoothing: alpha = prior strength (equivalent sample size)
    # Use alpha = sqrt(median count) as a reasonable prior
    counts = np.array(list(video_cnt.values()), dtype=float)
    alpha = float(np.sqrt(np.median(counts))) if len(counts) > 0 else 10.0
    alpha = max(alpha, 5.0)  # at least 5

    # Smoothed rate for each video
    def smoothed_rate(vid):
        pos = video_pos.get(vid, 0)
        cnt = video_cnt.get(vid, 0)
        return (pos + alpha * global_rate) / (cnt + alpha)

    # Compute smoothed rates for all training videos to build quantile bins
    all_rates = np.array([
        smoothed_rate(vid) for vid in video_pos.keys()
    ], dtype=float)
    
    # Build quantile bin edges from training rates
    # Use n_buckets quantile bins
    percentiles = np.linspace(0, 100, n_buckets + 1)
    bin_edges = np.percentile(all_rates, percentiles)
    # Ensure strictly increasing edges (handle duplicates)
    bin_edges = np.unique(bin_edges)
    # Actual number of bins after deduplication
    actual_buckets = len(bin_edges) - 1
    if actual_buckets < 1:
        actual_buckets = 1
        bin_edges = np.array([0.0, 1.0])

    # Map each target row's video to its bucket
    result = []
    for row in target_rows:
        vid = row[2]
        rate = smoothed_rate(vid)
        # np.searchsorted with bin_edges[1:-1] as thresholds
        bucket = int(np.searchsorted(bin_edges[1:-1], rate, side='right'))
        bucket = min(bucket, actual_buckets - 1)
        result.append(bucket)

    return np.array(result, dtype=np.int32), actual_buckets


# Build CTR feature for all splits
print("Building video CTR feature...", file=sys.stderr)
train_ctr_buckets, n_ctr_buckets = build_video_ctr_feature(
    splits['train'], splits['train']
)
valid_ctr_buckets, _ = build_video_ctr_feature(
    splits['train'], splits['valid']
)
test_ctr_buckets, _ = build_video_ctr_feature(
    splits['train'], splits['test']
)
print(f"CTR feature: {n_ctr_buckets} buckets, global shape: {train_ctr_buckets.shape}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Encode base features using data.encode()
# ---------------------------------------------------------------------------
enc, dim = encode(splits)

train_X, train_y, train_users = enc['train']
valid_X, valid_y, valid_users = enc['valid']
test_X,  test_y,  test_users  = enc['test']

# Append CTR bucket as an extra column, offset by current dim
# so it indexes into the same shared embedding table
train_X_aug = np.concatenate(
    [train_X, (train_ctr_buckets[:, None] + dim)], axis=1
).astype(np.int32)
valid_X_aug = np.concatenate(
    [valid_X, (valid_ctr_buckets[:, None] + dim)], axis=1
).astype(np.int32)
test_X_aug = np.concatenate(
    [test_X, (test_ctr_buckets[:, None] + dim)], axis=1
).astype(np.int32)

total_dim = dim + n_ctr_buckets
print(f"Base dim: {dim}, total dim with CTR: {total_dim}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Within-user rank averaging utility
# ---------------------------------------------------------------------------
def scores_to_user_ranks(scores, users):
    """
    Convert raw scores to within-user fractional ranks in [0, 1].
    rank = (rank among user's items - 1) / (n_items - 1)
    If user has only 1 item, rank = 0.5.
    """
    scores = np.array(scores, dtype=float)
    users_arr = np.array(users)
    ranks = np.zeros(len(scores), dtype=float)

    unique_users = np.unique(users_arr)
    for u in unique_users:
        mask = users_arr == u
        idx = np.where(mask)[0]
        s = scores[idx]
        n = len(s)
        if n == 1:
            ranks[idx] = 0.5
        else:
            # argsort of argsort gives rank (0-based)
            order = np.argsort(s)
            r = np.empty(n, dtype=float)
            r[order] = np.arange(n)
            ranks[idx] = r / (n - 1)
    return ranks


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
BATCH_SIZE = 8192
N_EPOCHS   = 50
PATIENCE   = 4
N_SEEDS    = 5
ENSEMBLE_SEEDS = [args.seed + i for i in range(N_SEEDS)]


def run_epoch(model, X, y, batch_size, rng):
    idx = rng.permutation(len(y))
    X_s, y_s = X[idx], y[idx]
    losses = []
    for start in range(0, len(y_s), batch_size):
        xb = X_s[start:start + batch_size]
        yb = y_s[start:start + batch_size]
        loss = model.step(xb, yb)
        losses.append(loss)
    return float(np.mean(losses))


def train_single_model(seed, train_X, train_y, valid_X, valid_y, valid_users,
                       total_dim):
    """Train one FM model with given seed, return (model, best_epoch)."""
    rng = np.random.default_rng(seed)
    model = FM(total_dim, k=16, lr=0.001, l2=1e-6, seed=seed)

    best_val = -np.inf
    best_epoch = 0
    patience_left = PATIENCE
    best_V = model.V.copy()
    best_W = model.W.copy()
    best_b = float(model.b)

    for epoch in range(1, N_EPOCHS + 1):
        run_epoch(model, train_X, train_y, BATCH_SIZE, rng)
        scores = model.predict(valid_X)
        val_res = evaluate(valid_users, valid_y.tolist(), scores.tolist())
        val_p = val_res['primary']

        if val_p > best_val + 1e-6:
            best_val = val_p
            best_epoch = epoch
            patience_left = PATIENCE
            best_V = model.V.copy()
            best_W = model.W.copy()
            best_b = float(model.b)
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.V[:] = best_V
    model.W[:] = best_W
    model.b = best_b
    print(f"  Seed {seed}: best_epoch={best_epoch}, val_primary={best_val:.6f}",
          file=sys.stderr)
    return model, best_epoch


# ---------------------------------------------------------------------------
# Main: train split
# ---------------------------------------------------------------------------
if args.train_split == 'train':
    # ------------------------------------------------------------------
    # Train N_SEEDS models, ensemble via within-user rank averaging
    # ------------------------------------------------------------------
    train_rank_accum = np.zeros(len(train_y), dtype=float)
    valid_rank_accum = np.zeros(len(valid_y), dtype=float)
    test_rank_accum  = np.zeros(len(test_y),  dtype=float)

    best_epochs = []

    for seed in ENSEMBLE_SEEDS:
        print(f"Training model with seed={seed}...", file=sys.stderr)
        model, best_epoch = train_single_model(
            seed, train_X_aug, train_y, valid_X_aug, valid_y, valid_users,
            total_dim
        )
        best_epochs.append(best_epoch)

        # Accumulate within-user ranks
        train_scores = model.predict(train_X_aug)
        valid_scores = model.predict(valid_X_aug)
        test_scores  = model.predict(test_X_aug)

        train_rank_accum += scores_to_user_ranks(train_scores, train_users)
        valid_rank_accum += scores_to_user_ranks(valid_scores, valid_users)
        test_rank_accum  += scores_to_user_ranks(test_scores,  test_users)

    # Average ranks
    train_ranks_avg = train_rank_accum / N_SEEDS
    valid_ranks_avg = valid_rank_accum / N_SEEDS
    test_ranks_avg  = test_rank_accum  / N_SEEDS

    median_epochs = int(np.median(best_epochs))
    print(f"Median best epoch: {median_epochs}", file=sys.stderr)

    # Evaluate
    train_res = evaluate(train_users, train_y.tolist(), train_ranks_avg.tolist())
    val_res   = evaluate(valid_users, valid_y.tolist(), valid_ranks_avg.tolist())

    # Unbiased primary
    unb = 0.0
    if HAS_UNBIASED:
        try:
            rand_rows = load_random_valid(args.data_dir)
            X_rand, y_rand, u_rand, _ = encode_like_train(splits['train'], rand_rows)
            # Append CTR bucket for random rows
            rand_ctr_buckets, _ = build_video_ctr_feature(splits['train'], rand_rows)
            X_rand_aug = np.concatenate(
                [X_rand, (rand_ctr_buckets[:, None] + dim)], axis=1
            ).astype(np.int32)

            # Need to rebuild models to score random rows — use last model's weights
            # Actually we need per-seed scores; retrain or reuse stored models
            # For efficiency: re-run inference only (models were already trained)
            # We don't store models, so retrain quickly (deterministic)
            rand_rank_accum = np.zeros(len(y_rand), dtype=float)
            for seed in ENSEMBLE_SEEDS:
                rng2 = np.random.default_rng(seed)
                m2 = FM(total_dim, k=16, lr=0.001, l2=1e-6, seed=seed)
                # Use median epoch count for this quick refit
                for ep in range(median_epochs):
                    run_epoch(m2, train_X_aug, train_y, BATCH_SIZE, rng2)
                rand_scores = m2.predict(X_rand_aug)
                rand_rank_accum += scores_to_user_ranks(rand_scores, u_rand)
            rand_ranks_avg = rand_rank_accum / N_SEEDS
            unb_res = evaluate(u_rand, y_rand.tolist(), rand_ranks_avg.tolist())
            unb = unb_res['primary']
        except Exception as e:
            print(f"Unbiased error: {e}", file=sys.stderr)
            unb = 0.0

    print(f"TRAIN_PRIMARY={train_res['primary']:.6f}")
    print(f"VAL_GAUC={val_res['GAUC']:.6f}")
    print(f"VAL_NDCG5={val_res['nDCG@5']:.6f}")
    print(f"VAL_PRIMARY={val_res['primary']:.6f}")
    print(f"UNBIASED_PRIMARY={unb:.6f}")

    write_submission(
        os.path.join(args.out_dir, 'submission_valid.csv'),
        splits['valid'],
        valid_ranks_avg.tolist()
    )
    write_submission(
        os.path.join(args.out_dir, 'submission_test.csv'),
        splits['test'],
        test_ranks_avg.tolist()
    )

else:
    # ------------------------------------------------------------------
    # train+valid mode: refit on both splits, fixed epoch count
    # ------------------------------------------------------------------
    # Build CTR feature using train rows only (no leakage)
    combined_rows = splits['train']  # CTR built from train only
    # valid rows get the same CTR buckets as above (already built)

    combined_X = np.concatenate([train_X_aug, valid_X_aug], axis=0)
    combined_y = np.concatenate([train_y, valid_y], axis=0)

    # Determine epoch count: do a quick single-seed run to find best epoch
    # Use a safe default based on known baseline behaviour
    FIXED_EPOCHS = 20

    test_rank_accum = np.zeros(len(test_y), dtype=float)

    for seed in ENSEMBLE_SEEDS:
        rng = np.random.default_rng(seed)
        model = FM(total_dim, k=16, lr=0.001, l2=1e-6, seed=seed)
        for epoch in range(1, FIXED_EPOCHS + 1):
            run_epoch(model, combined_X, combined_y, BATCH_SIZE, rng)
        test_scores = model.predict(test_X_aug)
        test_rank_accum += scores_to_user_ranks(test_scores, test_users)

    test_ranks_avg = test_rank_accum / N_SEEDS

    write_submission(
        os.path.join(args.out_dir, 'submission_test.csv'),
        splits['test'],
        test_ranks_avg.tolist()
    )