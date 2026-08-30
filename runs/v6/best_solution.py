#!/usr/bin/env python3
"""
Ensemble of 5 FM models with Bayesian-smoothed CTR features.
Change: rank-normalize each model's scores within each user before averaging,
so each model contributes equal ordinal weight regardless of score distribution width.
"""

import argparse
import os
import sys
import time
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--train_split', type=str, default='train',
                    choices=['train', 'train+valid'])
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kit'))
sys.path.insert(0, os.path.dirname(__file__))

from data import load, encode, FIELDS, LABEL
from baseline import FM
from evaluate import evaluate
import submit
from unbiased import load_random_valid, encode_like_train, unbiased_primary

np.random.seed(args.seed)

# Load data
data_dir = args.data_dir
splits = load(data_dir)

# -------------------------------------------------------------------------
# Build per-video CTR from training data only (with Bayesian smoothing)
# -------------------------------------------------------------------------
train_rows = splits['train']

video_pos = {}
video_cnt = {}
author_pos = {}
author_cnt = {}

for row in train_rows:
    vid = row[2]
    aut = row[3]
    lv = row[6]
    
    video_pos[vid] = video_pos.get(vid, 0) + lv
    video_cnt[vid] = video_cnt.get(vid, 0) + 1
    
    author_pos[aut] = author_pos.get(aut, 0) + lv
    author_cnt[aut] = author_cnt.get(aut, 0) + 1

# Compute global mean from training data
total_pos = sum(video_pos.values())
total_cnt = sum(video_cnt.values())
global_mean = total_pos / total_cnt if total_cnt > 0 else 0.5

print(f"Global training mean (long_view rate): {global_mean:.4f}", file=sys.stderr)

# Bayesian smoothing parameters
m = 10  # equivalent prior sample size
alpha = global_mean * m
beta = (1.0 - global_mean) * m

print(f"Bayesian prior: alpha={alpha:.4f}, beta={beta:.4f}, m={m}", file=sys.stderr)

# Bayesian-smoothed CTR estimates
video_ctr = {}
for vid in video_cnt:
    pos = video_pos[vid]
    cnt = video_cnt[vid]
    video_ctr[vid] = (pos + alpha) / (cnt + alpha + beta)

author_ctr = {}
for aut in author_cnt:
    pos = author_pos[aut]
    cnt = author_cnt[aut]
    author_ctr[aut] = (pos + alpha) / (cnt + alpha + beta)

# Build quantile-based buckets for video CTR
ctr_values = np.array(list(video_ctr.values()), dtype=np.float32)
n_bins = 10
quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # 9 interior percentile points
video_bin_edges = np.percentile(ctr_values, quantiles)
video_bin_edges = np.unique(video_bin_edges)

# Build quantile-based buckets for author CTR
author_ctr_values = np.array(list(author_ctr.values()), dtype=np.float32)
author_bin_edges = np.percentile(author_ctr_values, quantiles)
author_bin_edges = np.unique(author_bin_edges)

def ctr_to_bucket(ctr_val, edges):
    """Map a CTR float to a 0-indexed bucket integer."""
    return int(np.searchsorted(edges, ctr_val, side='right'))

# Video CTR buckets
n_video_bins = len(video_bin_edges) + 1  # actual number of bins after dedup
UNK_VIDEO_CTR = n_video_bins             # index for unseen videos

video_ctr_bucket = {}
for vid, ctr in video_ctr.items():
    video_ctr_bucket[vid] = ctr_to_bucket(ctr, video_bin_edges)

# Author CTR buckets
n_author_bins = len(author_bin_edges) + 1  # actual number of bins after dedup
UNK_AUTHOR_CTR = n_author_bins             # index for unseen authors

author_ctr_bucket = {}
for aut, ctr in author_ctr.items():
    author_ctr_bucket[aut] = ctr_to_bucket(ctr, author_bin_edges)

print(f"Video CTR bin edges: {video_bin_edges}", file=sys.stderr)
print(f"Author CTR bin edges: {author_bin_edges}", file=sys.stderr)

# -------------------------------------------------------------------------
# Encode standard features using data.encode(), then append our new features
# -------------------------------------------------------------------------
enc, base_dim = encode(splits)
X_train_base, y_train, users_train = enc['train']
X_valid_base, y_valid, users_valid = enc['valid']
X_test_base, y_test, users_test = enc['test']

# Layout of extra feature slots:
# [base_dim ... base_dim + n_video_bins] = video_ctr_bucket (n_video_bins buckets + 1 UNK)
# [base_dim + n_video_bins + 1 ... base_dim + n_video_bins + 1 + n_author_bins] = author_ctr_bucket

n_video_slots = n_video_bins + 1   # buckets + UNK
n_author_slots = n_author_bins + 1  # buckets + UNK
total_dim = base_dim + n_video_slots + n_author_slots

video_ctr_offset = base_dim
author_ctr_offset = base_dim + n_video_slots

def add_ctr_features(rows, base_X, video_ctr_bucket, author_ctr_bucket,
                     video_unk, author_unk, video_offset, author_offset):
    """
    Append video_ctr_bucket and author_ctr_bucket as new columns to base_X.
    """
    N = len(rows)
    vctr_col = np.zeros(N, dtype=np.int32)
    actr_col = np.zeros(N, dtype=np.int32)
    for i, row in enumerate(rows):
        vid = row[2]
        aut = row[3]
        vctr_col[i] = video_ctr_bucket.get(vid, video_unk) + video_offset
        actr_col[i] = author_ctr_bucket.get(aut, author_unk) + author_offset
    return np.concatenate([base_X, vctr_col.reshape(-1, 1), actr_col.reshape(-1, 1)], axis=1)

X_train = add_ctr_features(
    train_rows, X_train_base,
    video_ctr_bucket, author_ctr_bucket,
    UNK_VIDEO_CTR, UNK_AUTHOR_CTR,
    video_ctr_offset, author_ctr_offset
)
X_valid = add_ctr_features(
    splits['valid'], X_valid_base,
    video_ctr_bucket, author_ctr_bucket,
    UNK_VIDEO_CTR, UNK_AUTHOR_CTR,
    video_ctr_offset, author_ctr_offset
)
X_test = add_ctr_features(
    splits['test'], X_test_base,
    video_ctr_bucket, author_ctr_bucket,
    UNK_VIDEO_CTR, UNK_AUTHOR_CTR,
    video_ctr_offset, author_ctr_offset
)

print(f"Base dim: {base_dim}, Video CTR slots: {n_video_slots}, "
      f"Author CTR slots: {n_author_slots}, Total dim: {total_dim}", file=sys.stderr)


# -------------------------------------------------------------------------
# Rank-normalize scores within each user (fractional rank: 0-indexed rank / count)
# This ensures each ensemble member contributes equal ordinal weight regardless
# of score distribution width.
# -------------------------------------------------------------------------
def rank_normalize_by_user(scores, user_ids):
    """
    For each user, convert their raw scores to within-user fractional ranks.
    rank[i] = (0-indexed position in ascending argsort) / n_items_for_user
    Returns array of same shape as scores.
    
    Ties are broken by averaging their ranks (standard competition ranking).
    """
    scores = np.asarray(scores, dtype=np.float64)
    user_ids = np.asarray(user_ids)
    result = np.empty_like(scores)
    
    # Group by user
    unique_users, inverse = np.unique(user_ids, return_inverse=True)
    
    for uid_idx in range(len(unique_users)):
        mask = (inverse == uid_idx)
        user_scores = scores[mask]
        n = len(user_scores)
        if n == 1:
            result[mask] = 0.0
            continue
        # Compute fractional ranks (0-indexed argsort / n)
        # Use argsort of argsort to get ranks, then normalize
        order = np.argsort(user_scores, kind='stable')
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        result[mask] = ranks / n
    
    return result


# -------------------------------------------------------------------------
# Helper: train one FM model to convergence, return best weights and epoch count
# -------------------------------------------------------------------------

def train_fm(X_tr, y_tr, X_val, y_val, users_val, total_dim, seed,
             batch_size=8192, max_epochs=50, patience=4):
    """Train a single FM model, return (model_with_best_weights, best_epoch)."""
    model = FM(total_dim, k=16, lr=0.001, l2=1e-6, seed=seed)
    N_train = len(X_tr)
    
    best_val_primary = -1.0
    best_epoch = 0
    best_weights = None
    no_improve_count = 0
    
    rng = np.random.RandomState(seed)
    
    for epoch in range(max_epochs):
        t0 = time.time()
        idx = rng.permutation(N_train)
        losses = []
        for start in range(0, N_train, batch_size):
            batch = idx[start:start + batch_size]
            loss = model.step(X_tr[batch], y_tr[batch])
            losses.append(loss)
        
        val_scores = model.predict(X_val)
        val_metrics = evaluate(users_val, y_val, val_scores)
        val_primary = val_metrics['primary']
        
        elapsed = time.time() - t0
        print(f"  [seed={seed}] Epoch {epoch+1}: loss={np.mean(losses):.4f}, "
              f"val_primary={val_primary:.4f}, time={elapsed:.1f}s", file=sys.stderr)
        
        if val_primary > best_val_primary + 1e-6:
            best_val_primary = val_primary
            best_epoch = epoch + 1
            best_weights = {
                'V': model.V.copy(),
                'W': model.W.copy(),
                'b': model.b.copy()
            }
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"  [seed={seed}] Early stopping at epoch {epoch+1}, "
                      f"best epoch={best_epoch}", file=sys.stderr)
                break
    
    # Restore best weights
    model.V = best_weights['V']
    model.W = best_weights['W']
    model.b = best_weights['b']
    
    return model, best_epoch


# -------------------------------------------------------------------------
# Train / evaluate
# -------------------------------------------------------------------------

# Use 5 ensemble seeds: base on args.seed to allow reproducibility
N_ENSEMBLE = 5
ENSEMBLE_SEEDS = [args.seed + i for i in range(N_ENSEMBLE)]

if args.train_split == 'train+valid':
    X_combo = np.concatenate([X_train, X_valid], axis=0)
    y_combo = np.concatenate([y_train, y_valid], axis=0)
    
    # Use a fixed epoch count of 15 based on typical convergence
    n_epochs = 15
    batch_size = 8192
    
    test_scores_list = []
    for seed in ENSEMBLE_SEEDS:
        model = FM(total_dim, k=16, lr=0.001, l2=1e-6, seed=seed)
        N = len(X_combo)
        rng = np.random.RandomState(seed)
        
        for epoch in range(n_epochs):
            idx = rng.permutation(N)
            losses = []
            for start in range(0, N, batch_size):
                batch = idx[start:start + batch_size]
                loss = model.step(X_combo[batch], y_combo[batch])
                losses.append(loss)
            print(f"[seed={seed}] Epoch {epoch+1}/{n_epochs}, "
                  f"loss={np.mean(losses):.4f}", file=sys.stderr)
        
        test_scores_list.append(model.predict(X_test))
    
    # Rank-normalize each model's test scores within each user, then average
    users_test_arr = np.array(users_test)
    test_rank_scores_list = [
        rank_normalize_by_user(s, users_test_arr) for s in test_scores_list
    ]
    test_scores = np.mean(test_rank_scores_list, axis=0)
    
    submit.write_submission(
        os.path.join(args.out_dir, 'submission_test.csv'),
        splits['test'],
        test_scores
    )
    sys.exit(0)

# Normal training mode: train N_ENSEMBLE models independently, store them
print(f"Training ensemble of {N_ENSEMBLE} FM models...", file=sys.stderr)

# Store trained models for reuse in unbiased evaluation
trained_models = []
val_scores_list = []
test_scores_list = []
train_scores_list = []

for seed in ENSEMBLE_SEEDS:
    print(f"\n--- Training model with seed={seed} ---", file=sys.stderr)
    model, best_epoch = train_fm(
        X_train, y_train, X_valid, y_valid, users_valid,
        total_dim, seed=seed
    )
    
    # Store model for reuse in unbiased evaluation
    trained_models.append(model)
    
    train_scores_list.append(model.predict(X_train))
    val_scores_list.append(model.predict(X_valid))
    test_scores_list.append(model.predict(X_test))
    
    print(f"  Model seed={seed} done, best_epoch={best_epoch}", file=sys.stderr)

# Rank-normalize each model's scores within each user, then average
users_train_arr = np.array(users_train)
users_valid_arr = np.array(users_valid)
users_test_arr = np.array(users_test)

print("Rank-normalizing scores within users for each ensemble member...", file=sys.stderr)

train_rank_list = [rank_normalize_by_user(s, users_train_arr) for s in train_scores_list]
val_rank_list = [rank_normalize_by_user(s, users_valid_arr) for s in val_scores_list]
test_rank_list = [rank_normalize_by_user(s, users_test_arr) for s in test_scores_list]

train_scores = np.mean(train_rank_list, axis=0)
val_scores = np.mean(val_rank_list, axis=0)
test_scores = np.mean(test_rank_list, axis=0)

print(f"\nEnsemble rank-normalized averaging complete ({N_ENSEMBLE} models).", file=sys.stderr)

# Final metrics on ensemble predictions
train_metrics = evaluate(users_train, y_train, train_scores)
val_metrics = evaluate(users_valid, y_valid, val_scores)

# Unbiased evaluation - reuse stored trained_models (no double-training)
rand_rows = load_random_valid(data_dir)
X_rand_base, y_rand, u_rand, _ = encode_like_train(splits['train'], rand_rows)
X_rand = add_ctr_features(
    rand_rows, X_rand_base,
    video_ctr_bucket, author_ctr_bucket,
    UNK_VIDEO_CTR, UNK_AUTHOR_CTR,
    video_ctr_offset, author_ctr_offset
)

# Compute rank-normalized rand scores using stored models
u_rand_arr = np.array(u_rand)
rand_scores_list = [m.predict(X_rand) for m in trained_models]
rand_rank_list = [rank_normalize_by_user(s, u_rand_arr) for s in rand_scores_list]
rand_scores_avg = np.mean(rand_rank_list, axis=0)

unbiased = unbiased_primary(
    data_dir, splits['train'],
    lambda rows: rand_scores_avg
)

# Print results
print(f"TRAIN_PRIMARY={train_metrics['primary']:.4f}")
print(f"VAL_GAUC={val_metrics['GAUC']:.4f}")
print(f"VAL_NDCG5={val_metrics['nDCG@5']:.4f}")
print(f"VAL_PRIMARY={val_metrics['primary']:.4f}")
print(f"UNBIASED_PRIMARY={unbiased:.4f}")

# Write submissions
submit.write_submission(
    os.path.join(args.out_dir, 'submission_valid.csv'),
    splits['valid'],
    val_scores
)
submit.write_submission(
    os.path.join(args.out_dir, 'submission_test.csv'),
    splits['test'],
    test_scores
)