#!/usr/bin/env python3
"""
Iteration 2 (fixed): FM ensemble with Bayesian-smoothed per-video and per-author CTR buckets.

Fix: In train+valid mode, CTR features are computed from splits['train'] only (not combined_rows),
preventing leakage of validation labels into the feature computation.

Changes from iteration 1:
- Compute per-video and per-author Bayesian-smoothed long_view rates from training rows only
- Smooth formula: rate = (sum_lv + alpha) / (count + alpha/global_rate) where alpha=5
- Bucket each into 5 equal-frequency bins using training distribution
- Add these two bucketed columns as new FM features (7 features total instead of 5)
- Everything else identical: 3-seed ensemble, within-user rank averaging, early stopping
"""
import argparse
import os
import sys
import numpy as np
from scipy.stats import rankdata

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--out_dir', required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--train_split', default='train', choices=['train', 'train+valid'])
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from data import load, encode, FIELDS, LABEL
from evaluate import evaluate
from submit import write_submission
from baseline import FM
from unbiased import load_random_valid, encode_like_train, unbiased_primary

N_SEEDS = 3
ENSEMBLE_SEEDS = [args.seed + i for i in range(N_SEEDS)]


def compute_ctr_features(train_rows, target_rows, alpha=5.0):
    """
    Compute Bayesian-smoothed per-video and per-author long_view rates from
    training rows only. Then bucket into 5 equal-frequency bins.

    train_rows: rows used to compute statistics (must be strictly before target_rows)
    target_rows: rows to assign bucket values to

    Returns:
        video_buckets: int array of shape (len(target_rows),) with values 0..4
        author_buckets: int array of shape (len(target_rows),) with values 0..4
    """
    # Compute global rate from training data
    total_lv = sum(row[6] for row in train_rows)
    total_count = len(train_rows)
    global_rate = total_lv / total_count if total_count > 0 else 0.5

    # Per-video stats
    video_counts = {}
    video_lv_sums = {}
    for row in train_rows:
        vid = row[2]
        lv = row[6]
        video_counts[vid] = video_counts.get(vid, 0) + 1
        video_lv_sums[vid] = video_lv_sums.get(vid, 0) + lv

    # Per-author stats
    author_counts = {}
    author_lv_sums = {}
    for row in train_rows:
        auth = row[3]
        lv = row[6]
        author_counts[auth] = author_counts.get(auth, 0) + 1
        author_lv_sums[auth] = author_lv_sums.get(auth, 0) + lv

    # Smoothed rate formula: (sum_lv + alpha) / (count + alpha/global_rate)
    prior_count = alpha / global_rate if global_rate > 0 else alpha

    def smoothed_rate(lv_sum, count):
        return (lv_sum + alpha) / (count + prior_count)

    # Compute quantile bin edges from training distribution
    n_bins = 5
    video_quantiles = np.percentile(
        [smoothed_rate(video_lv_sums[v], video_counts[v]) for v in video_counts],
        np.linspace(0, 100, n_bins + 1)
    )
    author_quantiles = np.percentile(
        [smoothed_rate(author_lv_sums[a], author_counts[a]) for a in author_counts],
        np.linspace(0, 100, n_bins + 1)
    )

    # Make bin edges unique
    video_quantiles = np.unique(video_quantiles)
    author_quantiles = np.unique(author_quantiles)

    # Prior rate for unseen items
    prior_rate = smoothed_rate(0, 0)

    def get_video_rate(vid):
        if vid in video_counts:
            return smoothed_rate(video_lv_sums[vid], video_counts[vid])
        return prior_rate

    def get_author_rate(auth):
        if auth in author_counts:
            return smoothed_rate(author_lv_sums[auth], author_counts[auth])
        return prior_rate

    target_video_rates = np.array([get_video_rate(row[2]) for row in target_rows])
    target_author_rates = np.array([get_author_rate(row[3]) for row in target_rows])

    # Bucket into bins using training quantiles
    def bucket_rates(rates, quantiles, n_bins):
        if len(quantiles) > 2:
            edges = quantiles[1:-1]  # interior edges
        else:
            edges = quantiles
        buckets = np.searchsorted(edges, rates, side='right')
        return np.clip(buckets, 0, n_bins - 1).astype(np.int32)

    video_buckets = bucket_rates(target_video_rates, video_quantiles, n_bins)
    author_buckets = bucket_rates(target_author_rates, author_quantiles, n_bins)

    return video_buckets, author_buckets


def augment_X_with_ctr(X, video_buckets, author_buckets, base_dim, n_bins=5):
    """
    Add video_ctr_bucket and author_ctr_bucket as new columns to X.
    The new indices are offset by base_dim for video_ctr and base_dim+n_bins for author_ctr.

    Returns augmented X with shape (N, 7) and new total dimension.
    """
    # video_ctr indices: base_dim + 0..4
    video_col = (video_buckets + base_dim).reshape(-1, 1).astype(np.int32)
    # author_ctr indices: base_dim + n_bins + 0..4
    author_col = (author_buckets + base_dim + n_bins).reshape(-1, 1).astype(np.int32)

    X_aug = np.concatenate([X, video_col, author_col], axis=1)
    new_dim = base_dim + 2 * n_bins
    return X_aug, new_dim


def within_user_ranks(scores, users):
    """Convert raw scores to within-user fractional ranks (0..1)."""
    users_arr = np.array(users)
    ranks = np.zeros(len(scores), dtype=np.float64)
    unique_users = np.unique(users_arr)
    for u in unique_users:
        mask = users_arr == u
        u_scores = scores[mask]
        u_ranks = rankdata(u_scores, method='average')
        n = len(u_ranks)
        if n > 1:
            u_ranks = (u_ranks - 1.0) / (n - 1.0)
        else:
            u_ranks = np.array([0.5])
        ranks[mask] = u_ranks
    return ranks


def train_single_model(X_train, y_train, X_valid, y_valid, users_valid, dim, seed):
    """Train one FM model with early stopping, return best model and epoch count."""
    rng = np.random.default_rng(seed)
    model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=seed)

    bs = 8192
    n = len(y_train)
    best_val = -np.inf
    best_weights = None
    patience = 4
    wait = 0
    best_epoch = 0

    for epoch in range(1, 51):
        idx = rng.permutation(n)
        for start in range(0, n, bs):
            batch = idx[start:start + bs]
            model.step(X_train[batch], y_train[batch])

        val_scores = model.predict(X_valid)
        val_metrics = evaluate(users_valid, y_valid, val_scores)
        val_primary = val_metrics['primary']

        if val_primary > best_val + 1e-6:
            best_val = val_primary
            best_epoch = epoch
            best_weights = (model.V.copy(), model.W.copy(), float(model.b))
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best weights
    model.V[:] = best_weights[0]
    model.W[:] = best_weights[1]
    model.b = np.float32(best_weights[2])
    return model, best_epoch, best_val


# ── Load data ─────────────────────────────────────────────────────────────────
data_dir = args.data_dir
splits = load(data_dir)

if args.train_split == 'train+valid':
    # Combine train + valid for final test submission.
    # IMPORTANT: CTR features must be computed from splits['train'] only to avoid leakage.
    # The validation rows' labels must NOT influence the CTR statistics used to score them.
    # We use splits['train'] for CTR computation, then apply those statistics to all splits.
    combined_rows = splits['train'] + splits['valid']

    # Build encoder on combined data (for embedding coverage)
    combined_splits = {'train': combined_rows, 'valid': splits['test'], 'test': splits['test']}
    enc, dim = encode(combined_splits)
    X_train_base, y_train, users_train = enc['train']
    X_test_base, y_test, users_test = enc['test']

    # CTR features: use splits['train'] only (NOT combined_rows) to avoid leakage
    print("[info] Computing CTR features for train+valid mode (using train-only stats)...",
          file=sys.stderr)
    ctr_reference_rows = splits['train']  # strictly historical data only
    video_buckets_train, author_buckets_train = compute_ctr_features(ctr_reference_rows, combined_rows)
    video_buckets_test, author_buckets_test = compute_ctr_features(ctr_reference_rows, splits['test'])

    X_train, new_dim = augment_X_with_ctr(X_train_base, video_buckets_train, author_buckets_train, dim)
    X_test, _ = augment_X_with_ctr(X_test_base, video_buckets_test, author_buckets_test, dim)

    # Train N_SEEDS models for a fixed number of epochs (use epoch count from train run)
    # Fixed at 10 epochs as a reasonable default for the combined training set
    test_rank_sum = np.zeros(len(y_test), dtype=np.float64)
    n_epochs = 10
    for seed in ENSEMBLE_SEEDS:
        rng = np.random.default_rng(seed)
        model = FM(new_dim, k=16, lr=0.001, l2=1e-6, seed=seed)
        n = len(y_train)
        for epoch in range(n_epochs):
            idx = rng.permutation(n)
            for start in range(0, n, 8192):
                batch = idx[start:start + 8192]
                model.step(X_train[batch], y_train[batch])
        test_scores = model.predict(X_test)
        test_ranks = within_user_ranks(test_scores, users_test)
        test_rank_sum += test_ranks

    final_test_scores = test_rank_sum / N_SEEDS
    write_submission(os.path.join(args.out_dir, 'submission_test.csv'),
                     splits['test'], final_test_scores)
    sys.exit(0)

# ── Normal training path ──────────────────────────────────────────────────────
enc, dim = encode(splits)
X_train_base, y_train, users_train = enc['train']
X_valid_base, y_valid, users_valid = enc['valid']
X_test_base, y_test, users_test = enc['test']

# Compute CTR features from training rows only (no leakage)
print("[info] Computing Bayesian-smoothed CTR features...", file=sys.stderr)
train_rows = splits['train']
valid_rows = splits['valid']
test_rows = splits['test']

video_buckets_train, author_buckets_train = compute_ctr_features(train_rows, train_rows)
video_buckets_valid, author_buckets_valid = compute_ctr_features(train_rows, valid_rows)
video_buckets_test, author_buckets_test = compute_ctr_features(train_rows, test_rows)

# Augment X matrices with new CTR bucket columns
X_train, new_dim = augment_X_with_ctr(X_train_base, video_buckets_train, author_buckets_train, dim)
X_valid, _ = augment_X_with_ctr(X_valid_base, video_buckets_valid, author_buckets_valid, dim)
X_test, _ = augment_X_with_ctr(X_test_base, video_buckets_test, author_buckets_test, dim)

print(f"[info] Feature dim: {dim} -> {new_dim} (added video_ctr + author_ctr buckets)",
      file=sys.stderr)

# Train ensemble of N_SEEDS models
val_rank_sum = np.zeros(len(y_valid), dtype=np.float64)
test_rank_sum = np.zeros(len(y_test), dtype=np.float64)
train_rank_sum = np.zeros(len(y_train), dtype=np.float64)

trained_models = []
for seed in ENSEMBLE_SEEDS:
    model, best_epoch, best_val = train_single_model(
        X_train, y_train, X_valid, y_valid, users_valid, new_dim, seed
    )
    print(f"[info] seed={seed}, best_epoch={best_epoch}, best_val={best_val:.6f}",
          file=sys.stderr)
    trained_models.append(model)

    # Train scores
    train_sc = model.predict(X_train)
    train_ranks = within_user_ranks(train_sc, users_train)
    train_rank_sum += train_ranks

    # Valid scores
    val_sc = model.predict(X_valid)
    val_ranks = within_user_ranks(val_sc, users_valid)
    val_rank_sum += val_ranks

    # Test scores
    test_sc = model.predict(X_test)
    test_ranks = within_user_ranks(test_sc, users_test)
    test_rank_sum += test_ranks

# Average ranks
final_train_scores = train_rank_sum / N_SEEDS
final_val_scores = val_rank_sum / N_SEEDS
final_test_scores = test_rank_sum / N_SEEDS

# ── Evaluate ──────────────────────────────────────────────────────────────────
train_metrics = evaluate(users_train, y_train, final_train_scores)
val_metrics = evaluate(users_valid, y_valid, final_val_scores)

# ── Unbiased evaluation ───────────────────────────────────────────────────────
rand_rows = load_random_valid(data_dir)
X_rand_base, y_rand, u_rand, rand_dim = encode_like_train(splits['train'], rand_rows)

# Compute CTR features for random exposure rows using training data only
video_buckets_rand, author_buckets_rand = compute_ctr_features(train_rows, rand_rows)
X_rand, _ = augment_X_with_ctr(X_rand_base, video_buckets_rand, author_buckets_rand, dim)

# Use already-trained models (no retraining needed)
rand_rank_sum = np.zeros(len(y_rand), dtype=np.float64)
for model in trained_models:
    rand_sc = model.predict(X_rand)
    rand_ranks = within_user_ranks(rand_sc, u_rand)
    rand_rank_sum += rand_ranks

final_rand_scores = rand_rank_sum / N_SEEDS

# Compute unbiased primary
unbiased_metrics = evaluate(u_rand, y_rand, final_rand_scores)
unbiased = unbiased_metrics['primary']

# ── Write submissions ─────────────────────────────────────────────────────────
write_submission(os.path.join(args.out_dir, 'submission_valid.csv'),
                 splits['valid'], final_val_scores)
write_submission(os.path.join(args.out_dir, 'submission_test.csv'),
                 splits['test'], final_test_scores)

# ── Print metrics ─────────────────────────────────────────────────────────────
print(f"TRAIN_PRIMARY={train_metrics['primary']:.6f}")
print(f"VAL_GAUC={val_metrics['GAUC']:.6f}")
print(f"VAL_NDCG5={val_metrics['nDCG@5']:.6f}")
print(f"VAL_PRIMARY={val_metrics['primary']:.6f}")
print(f"UNBIASED_PRIMARY={unbiased:.6f}")