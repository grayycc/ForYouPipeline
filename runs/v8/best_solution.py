#!/usr/bin/env python3
"""
FM ensemble with:
1. 5-seed ensemble with within-user rank averaging
2. Bayesian-smoothed per-video and per-author long_view rate, bucketed into 10 bins
"""
import argparse
import os
import sys
import time
import numpy as np

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--out_dir',  required=True)
parser.add_argument('--seed',     type=int, default=0)
parser.add_argument('--train_split', default='train',
                    choices=['train', 'train+valid'])
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
np.random.seed(args.seed)

# ── imports ───────────────────────────────────────────────────────────────────
from data     import load, LABEL
from baseline import FM
from evaluate import evaluate
import submit
from unbiased import load_random_valid, encode_like_train, unbiased_primary

# ── data ──────────────────────────────────────────────────────────────────────
t0 = time.time()
splits = load(args.data_dir)

# ── Bayesian-smoothed CTR features ────────────────────────────────────────────
def compute_smoothed_ctr(train_rows, m=50.0):
    """
    Compute Bayesian-smoothed long_view rate for video_id and author_id
    using only training rows. m is the smoothing count (number of prior
    pseudo-observations equal to the global mean).
    
    Returns:
        video_ctr: dict {video_id -> smoothed_rate}
        author_ctr: dict {author_id -> smoothed_rate}
        global_rate: float
    """
    total_views = 0
    total_count = 0
    video_views = {}
    video_count = {}
    author_views = {}
    author_count = {}
    
    for row in train_rows:
        label = row[6]  # long_view
        vid = row[2]
        auth = row[3]
        
        total_views += label
        total_count += 1
        
        video_views[vid] = video_views.get(vid, 0) + label
        video_count[vid] = video_count.get(vid, 0) + 1
        
        author_views[auth] = author_views.get(auth, 0) + label
        author_count[auth] = author_count.get(auth, 0) + 1
    
    global_rate = total_views / max(total_count, 1)
    
    video_ctr = {}
    for vid in video_count:
        n = video_count[vid]
        v = video_views[vid]
        video_ctr[vid] = (v + m * global_rate) / (n + m)
    
    author_ctr = {}
    for auth in author_count:
        n = author_count[auth]
        v = author_views[auth]
        author_ctr[auth] = (v + m * global_rate) / (n + m)
    
    return video_ctr, author_ctr, global_rate


def bucketize_ctr(ctr_dict, global_rate, n_bins=10):
    """
    Bucketize CTR values into n_bins bins using quantiles from training values.
    Returns (bucket_dict, bin_edges).
    """
    values = np.array(list(ctr_dict.values()))
    # Use quantile-based binning
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(values, quantiles)
    # Make edges unique to avoid issues
    bin_edges = np.unique(bin_edges)
    
    bucket_dict = {}
    for key, val in ctr_dict.items():
        bucket = np.searchsorted(bin_edges[1:-1], val)  # 0 to n_bins-1
        bucket_dict[key] = int(bucket)
    
    return bucket_dict, bin_edges


def build_extended_features(rows, video_bucket, author_bucket, global_video_bucket,
                            global_author_bucket, base_enc_rows, enc_dim,
                            video_offset, author_offset):
    """
    Add video_ctr_bucket and author_ctr_bucket as extra feature columns.
    These are appended to the existing encoded feature matrix.
    
    video_offset: offset for video ctr bucket features in the embedding table
    author_offset: offset for author ctr bucket features in the embedding table
    """
    n = len(base_enc_rows)
    video_feats = np.zeros(n, dtype=np.int32)
    author_feats = np.zeros(n, dtype=np.int32)
    
    for i, row in enumerate(rows):
        vid = row[2]
        auth = row[3]
        vb = video_bucket.get(vid, global_video_bucket)
        ab = author_bucket.get(auth, global_author_bucket)
        video_feats[i] = video_offset + vb
        author_feats[i] = author_offset + ab
    
    return video_feats, author_feats


def custom_encode(splits, train_split_name='train'):
    """
    Custom encoding that extends data.encode() with smoothed CTR buckets.
    Returns extended feature matrices and updated embedding dimension.
    """
    # Use data.encode for the base features
    from data import encode
    enc_base, dim_base = encode(splits)
    
    train_rows = splits['train']
    
    # Compute smoothed CTR from training data only
    video_ctr, author_ctr, global_rate = compute_smoothed_ctr(train_rows, m=50.0)
    
    # Bucketize
    video_bucket, video_bin_edges = bucketize_ctr(video_ctr, global_rate, n_bins=10)
    author_bucket, author_bin_edges = bucketize_ctr(author_ctr, global_rate, n_bins=10)
    
    # Global fallback bucket (median bin = 5)
    global_video_bucket = 5
    global_author_bucket = 5
    
    # Offsets in the embedding table for the new features
    # We add 10 bins each for video_ctr and author_ctr
    n_video_bins = 10
    n_author_bins = 10
    video_ctr_offset = dim_base
    author_ctr_offset = dim_base + n_video_bins
    new_dim = dim_base + n_video_bins + n_author_bins
    
    extended_enc = {}
    for split_name in splits:
        X_base, y, users = enc_base[split_name]
        rows = splits[split_name]
        
        video_feats, author_feats = build_extended_features(
            rows, video_bucket, author_bucket,
            global_video_bucket, global_author_bucket,
            X_base, dim_base,
            video_ctr_offset, author_ctr_offset
        )
        
        # Append the new feature columns
        X_ext = np.concatenate([
            X_base,
            video_feats.reshape(-1, 1),
            author_feats.reshape(-1, 1)
        ], axis=1).astype(np.int32)
        
        extended_enc[split_name] = (X_ext, y, users)
    
    return extended_enc, new_dim, video_bucket, author_bucket, global_video_bucket, global_author_bucket, video_ctr_offset, author_ctr_offset


def within_user_fractional_rank(scores, users):
    """
    Convert raw scores to within-user fractional ranks.
    For each user, rank their scores from 0 to 1 (fractional ranks).
    """
    user_arr = np.array(users)
    ranks = np.zeros(len(scores), dtype=np.float64)
    
    unique_users = np.unique(user_arr)
    for u in unique_users:
        mask = user_arr == u
        idx = np.where(mask)[0]
        user_scores = scores[idx]
        # Fractional rank: 0 = lowest, 1 = highest
        n = len(idx)
        if n == 1:
            ranks[idx] = 0.5
        else:
            order = np.argsort(user_scores)
            frac_ranks = np.arange(n) / (n - 1)
            ranks[idx[order]] = frac_ranks
    
    return ranks


def run_epoch(X, y, model, bs, rng):
    idx = rng.permutation(len(y))
    total_loss = 0.0
    n_batches  = 0
    for start in range(0, len(idx), bs):
        batch = idx[start:start+bs]
        loss  = model.step(X[batch], y[batch])
        total_loss += loss
        n_batches  += 1
    return total_loss / max(n_batches, 1)


# ── encode data with extended features ───────────────────────────────────────
print(f"Encoding data...", file=sys.stderr)
enc, dim, video_bucket, author_bucket, gvb, gab, vc_offset, ac_offset = custom_encode(splits)

(X_tr, y_tr, u_tr) = enc['train']
(X_va, y_va, u_va) = enc['valid']
(X_te, y_te, u_te) = enc['test']

print(f"Data loaded in {time.time()-t0:.1f}s  dim={dim}  X_tr={X_tr.shape}", file=sys.stderr)

N_SEEDS = 5
BS = 8192
PATIENCE = 4
MAX_EPOCHS = 50
FIXED_EPOCHS = 15

if args.train_split == 'train+valid':
    # Combine train + valid
    X_tr_full = np.concatenate([X_tr, X_va], axis=0)
    y_tr_full = np.concatenate([y_tr, y_va], axis=0)
    u_tr_full = u_tr + u_va
    
    # Train N_SEEDS models and collect scores
    all_te_ranks = []
    
    for seed_i in range(N_SEEDS):
        seed_val = args.seed * 100 + seed_i
        model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=seed_val)
        rng = np.random.RandomState(seed_val)
        
        for epoch in range(1, FIXED_EPOCHS + 1):
            run_epoch(X_tr_full, y_tr_full, model, BS, rng)
        
        scores_te = model.predict(X_te)
        ranks_te = within_user_fractional_rank(scores_te, u_te)
        all_te_ranks.append(ranks_te)
    
    # Average ranks
    avg_te_ranks = np.mean(all_te_ranks, axis=0)
    
    submit.write_submission(
        os.path.join(args.out_dir, 'submission_test.csv'),
        splits['test'], avg_te_ranks)
    sys.exit(0)

# ── normal train: 5-seed ensemble ─────────────────────────────────────────────
all_va_ranks = []
all_te_ranks = []
all_tr_ranks = []

best_epochs_list = []

for seed_i in range(N_SEEDS):
    seed_val = args.seed * 100 + seed_i
    print(f"Training seed {seed_i} (seed_val={seed_val})...", file=sys.stderr)
    
    model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=seed_val)
    rng = np.random.RandomState(seed_val)
    
    best_primary = -1.0
    best_epoch   = -1
    patience_cnt = 0
    best_V = None
    best_W = None
    best_b = None
    
    for epoch in range(1, MAX_EPOCHS + 1):
        loss = run_epoch(X_tr, y_tr, model, BS, rng)
        
        scores_va = model.predict(X_va)
        metrics_va = evaluate(u_va, y_va, scores_va)
        primary_va = metrics_va['primary']
        
        print(f"  seed={seed_i} epoch {epoch:2d}  loss={loss:.4f}  val_primary={primary_va:.4f}",
              file=sys.stderr)
        
        if primary_va > best_primary + 1e-6:
            best_primary = primary_va
            best_epoch   = epoch
            best_V = model.V.copy()
            best_W = model.W.copy()
            best_b = float(model.b)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at epoch {epoch}, best epoch {best_epoch}",
                      file=sys.stderr)
                break
    
    best_epochs_list.append(best_epoch)
    
    # Restore best weights
    model.V[:] = best_V
    model.W[:] = best_W
    model.b    = best_b
    
    # Get scores and convert to within-user fractional ranks
    scores_tr = model.predict(X_tr)
    scores_va = model.predict(X_va)
    scores_te = model.predict(X_te)
    
    ranks_tr = within_user_fractional_rank(scores_tr, u_tr)
    ranks_va = within_user_fractional_rank(scores_va, u_va)
    ranks_te = within_user_fractional_rank(scores_te, u_te)
    
    all_tr_ranks.append(ranks_tr)
    all_va_ranks.append(ranks_va)
    all_te_ranks.append(ranks_te)
    
    print(f"  seed={seed_i} best_epoch={best_epoch} best_primary={best_primary:.4f}",
          file=sys.stderr)

print(f"Best epochs across seeds: {best_epochs_list}", file=sys.stderr)

# Average ranks across ensemble members
avg_tr_ranks = np.mean(all_tr_ranks, axis=0)
avg_va_ranks = np.mean(all_va_ranks, axis=0)
avg_te_ranks = np.mean(all_te_ranks, axis=0)

# ── evaluation ────────────────────────────────────────────────────────────────
metrics_tr = evaluate(u_tr, y_tr, avg_tr_ranks)
metrics_va = evaluate(u_va, y_va, avg_va_ranks)

print(f"Final ensemble: train_primary={metrics_tr['primary']:.4f}  val_primary={metrics_va['primary']:.4f}",
      file=sys.stderr)

# ── Unbiased primary ──────────────────────────────────────────────────────────
# We need to score the random rows using our ensemble
rand_rows = load_random_valid(args.data_dir)

# Build the extended features for random rows
# encode_like_train uses train vocabulary; we need to add our CTR features on top
def score_random_rows(rows):
    """Score random rows using the ensemble with extended features."""
    # We need to encode these rows the same way as our training data
    # First get base encoding via encode_like_train
    X_base, y_r, u_r, _ = encode_like_train(splits['train'], rows)
    
    # Add CTR bucket features
    n = len(rows)
    video_feats = np.zeros(n, dtype=np.int32)
    author_feats = np.zeros(n, dtype=np.int32)
    
    for i, row in enumerate(rows):
        vid = row[2]
        auth = row[3]
        vb = video_bucket.get(vid, gvb)
        ab = author_bucket.get(auth, gab)
        video_feats[i] = vc_offset + vb
        author_feats[i] = ac_offset + ab
    
    X_ext = np.concatenate([
        X_base,
        video_feats.reshape(-1, 1),
        author_feats.reshape(-1, 1)
    ], axis=1).astype(np.int32)
    
    return X_ext, u_r


# We need to retrain or reuse the last model for unbiased scoring
# Since we saved all ranks from ensemble, we need to score rand_rows with all models
# Re-train all models with the same seeds to get consistent scoring
# Actually, we need to keep models around. Let's retrain with fixed epochs.

# To avoid retraining, let's use a different approach:
# retrain each model with its best epoch count and score the random rows
all_rand_scores = []

for seed_i in range(N_SEEDS):
    seed_val = args.seed * 100 + seed_i
    best_epoch_i = best_epochs_list[seed_i]
    
    model_i = FM(dim, k=16, lr=0.001, l2=1e-6, seed=seed_val)
    rng_i = np.random.RandomState(seed_val)
    
    for epoch in range(1, best_epoch_i + 1):
        run_epoch(X_tr, y_tr, model_i, BS, rng_i)
    
    X_rand_ext, u_rand = score_random_rows(rand_rows)
    scores_rand_i = model_i.predict(X_rand_ext)
    all_rand_scores.append(scores_rand_i)

# Average raw scores for random rows (rank averaging within user would be ideal
# but unbiased_primary handles it internally via evaluate())
avg_rand_scores = np.mean(all_rand_scores, axis=0)

# Use unbiased_primary with our custom scorer
unb = unbiased_primary(
    args.data_dir, splits['train'],
    lambda rows: np.mean([
        FM(dim, k=16, lr=0.001, l2=1e-6, seed=args.seed*100+si).predict(
            score_random_rows(rows)[0]
        )
        for si in range(1)  # just use seed 0 to avoid retraining all
    ], axis=0)
)

# Actually, let's compute unbiased_primary more carefully using our pre-computed scores
# unbiased_primary(data_dir, train_rows, score_fn) where score_fn takes rows and returns scores
# We already have avg_rand_scores for rand_rows, but unbiased_primary will call score_fn
# on a potentially different set of rows. Let's just pass a simple scorer.

# Use the last trained model (seed N_SEEDS-1) as a proxy for the unbiased score
# This is an approximation but avoids retraining all models again
model_last = FM(dim, k=16, lr=0.001, l2=1e-6, seed=args.seed * 100 + (N_SEEDS - 1))
rng_last = np.random.RandomState(args.seed * 100 + (N_SEEDS - 1))
for epoch in range(1, best_epochs_list[-1] + 1):
    run_epoch(X_tr, y_tr, model_last, BS, rng_last)

def score_fn_for_unbiased(rows):
    X_ext, _ = score_random_rows(rows)
    return model_last.predict(X_ext)

unb = unbiased_primary(args.data_dir, splits['train'], score_fn_for_unbiased)

# ── print required lines ──────────────────────────────────────────────────────
print(f"TRAIN_PRIMARY={metrics_tr['primary']:.4f}")
print(f"VAL_GAUC={metrics_va['GAUC']:.4f}")
print(f"VAL_NDCG5={metrics_va['nDCG@5']:.4f}")
print(f"VAL_PRIMARY={metrics_va['primary']:.4f}")
print(f"UNBIASED_PRIMARY={unb:.4f}")

# ── write submissions ─────────────────────────────────────────────────────────
submit.write_submission(
    os.path.join(args.out_dir, 'submission_valid.csv'),
    splits['valid'], avg_va_ranks)
submit.write_submission(
    os.path.join(args.out_dir, 'submission_test.csv'),
    splits['test'], avg_te_ranks)