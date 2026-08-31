#!/usr/bin/env python3
"""
FM with video-level AND author-level historical engagement features
(positive rate + log-count from train), both quantile-bucketed into 10 bins + 1 UNK.
This extends the feature set from 7 fields (base 5 + video pos_rate + video log_count)
to 9 fields by adding author-level pos_rate and author-level log_count.
"""
import argparse
import os
import sys
import csv
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--out_dir', required=True)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports from the repo helpers
# ---------------------------------------------------------------------------
from data import load, encode, FIELDS
from evaluate import evaluate
from submit import write_submission
from baseline import FM

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
splits = load(args.data_dir)
train_rows = splits['train']
valid_rows = splits['valid']
test_rows  = splits['test']

print(f"Train size: {len(train_rows)}, Valid: {len(valid_rows)}, Test: {len(test_rows)}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Use encode() to get base features + vocabulary info
# We'll extend the feature matrix with 4 additional columns
# ---------------------------------------------------------------------------
(enc, dim_base) = encode(splits)

X_tr_base, y_tr, users_tr = enc['train']
X_va_base, y_va, users_va = enc['valid']
X_te_base, y_te, users_te = enc['test']

# ---------------------------------------------------------------------------
# Compute per-video stats from training data ONLY (no leakage)
# ---------------------------------------------------------------------------
video_pos_count = {}
video_total_count = {}

for row in train_rows:
    vid = row[2]
    label = row[6]
    video_total_count[vid] = video_total_count.get(vid, 0) + 1
    video_pos_count[vid] = video_pos_count.get(vid, 0) + label

# Compute positive rate per video
video_pos_rate = {}
for vid, total in video_total_count.items():
    video_pos_rate[vid] = video_pos_count.get(vid, 0) / total

# Compute log10 count per video
video_log_count = {vid: np.log10(1 + cnt) for vid, cnt in video_total_count.items()}

print(f"Unique videos in train: {len(video_total_count)}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Compute per-author stats from training data ONLY (no leakage)
# ---------------------------------------------------------------------------
author_pos_count = {}
author_total_count = {}

for row in train_rows:
    author = row[3]  # author_id is at index 3
    label = row[6]
    author_total_count[author] = author_total_count.get(author, 0) + 1
    author_pos_count[author] = author_pos_count.get(author, 0) + label

# Compute positive rate per author
author_pos_rate = {}
for author, total in author_total_count.items():
    author_pos_rate[author] = author_pos_count.get(author, 0) / total

# Compute log10 count per author
author_log_count = {author: np.log10(1 + cnt) for author, cnt in author_total_count.items()}

print(f"Unique authors in train: {len(author_total_count)}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Build quantile buckets for video features using training videos only
# ---------------------------------------------------------------------------
train_video_pos_rates = np.array(list(video_pos_rate.values()), dtype=np.float32)
train_video_log_counts = np.array(list(video_log_count.values()), dtype=np.float32)

video_pos_rate_edges = np.quantile(train_video_pos_rates, np.linspace(0, 1, 11)[1:-1])
video_log_count_edges = np.quantile(train_video_log_counts, np.linspace(0, 1, 11)[1:-1])

print(f"Video pos rate edges: {video_pos_rate_edges}", file=sys.stderr)
print(f"Video log count edges: {video_log_count_edges}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Build quantile buckets for author features using training authors only
# ---------------------------------------------------------------------------
train_author_pos_rates = np.array(list(author_pos_rate.values()), dtype=np.float32)
train_author_log_counts = np.array(list(author_log_count.values()), dtype=np.float32)

author_pos_rate_edges = np.quantile(train_author_pos_rates, np.linspace(0, 1, 11)[1:-1])
author_log_count_edges = np.quantile(train_author_log_counts, np.linspace(0, 1, 11)[1:-1])

print(f"Author pos rate edges: {author_pos_rate_edges}", file=sys.stderr)
print(f"Author log count edges: {author_log_count_edges}", file=sys.stderr)

def get_video_pos_rate_bucket(vid):
    if vid not in video_pos_rate:
        return 10  # UNK bucket
    return int(np.searchsorted(video_pos_rate_edges, video_pos_rate[vid]))

def get_video_log_count_bucket(vid):
    if vid not in video_log_count:
        return 10  # UNK bucket
    return int(np.searchsorted(video_log_count_edges, video_log_count[vid]))

def get_author_pos_rate_bucket(author):
    if author not in author_pos_rate:
        return 10  # UNK bucket
    return int(np.searchsorted(author_pos_rate_edges, author_pos_rate[author]))

def get_author_log_count_bucket(author):
    if author not in author_log_count:
        return 10  # UNK bucket
    return int(np.searchsorted(author_log_count_edges, author_log_count[author]))

# ---------------------------------------------------------------------------
# Create extended feature columns
# Feature layout:
#   [base features (dim_base indices)]
#   + [video_pos_rate_bucket offset]     (11 values: 0-9 + UNK)
#   + [video_log_count_bucket offset]    (11 values: 0-9 + UNK)
#   + [author_pos_rate_bucket offset]    (11 values: 0-9 + UNK)
#   + [author_log_count_bucket offset]   (11 values: 0-9 + UNK)
# ---------------------------------------------------------------------------
n_bucket_vals = 11  # 10 buckets + 1 UNK

offset_vid_pos_rate   = dim_base
offset_vid_log_count  = dim_base + n_bucket_vals
offset_auth_pos_rate  = dim_base + 2 * n_bucket_vals
offset_auth_log_count = dim_base + 3 * n_bucket_vals
dim_extended = dim_base + 4 * n_bucket_vals

print(f"Base dim: {dim_base}, Extended dim: {dim_extended}", file=sys.stderr)

def make_extra_cols(rows):
    """Build the 4 extra feature columns for a list of rows."""
    n = len(rows)
    col_vid_pos_rate   = np.zeros(n, dtype=np.int32)
    col_vid_log_count  = np.zeros(n, dtype=np.int32)
    col_auth_pos_rate  = np.zeros(n, dtype=np.int32)
    col_auth_log_count = np.zeros(n, dtype=np.int32)
    for i, row in enumerate(rows):
        vid    = row[2]
        author = row[3]
        col_vid_pos_rate[i]   = offset_vid_pos_rate   + get_video_pos_rate_bucket(vid)
        col_vid_log_count[i]  = offset_vid_log_count  + get_video_log_count_bucket(vid)
        col_auth_pos_rate[i]  = offset_auth_pos_rate  + get_author_pos_rate_bucket(author)
        col_auth_log_count[i] = offset_auth_log_count + get_author_log_count_bucket(author)
    return (col_vid_pos_rate.reshape(-1, 1),
            col_vid_log_count.reshape(-1, 1),
            col_auth_pos_rate.reshape(-1, 1),
            col_auth_log_count.reshape(-1, 1))

# Build extended feature matrices
vpr_tr, vlc_tr, apr_tr, alc_tr = make_extra_cols(train_rows)
vpr_va, vlc_va, apr_va, alc_va = make_extra_cols(valid_rows)
vpr_te, vlc_te, apr_te, alc_te = make_extra_cols(test_rows)

X_tr = np.concatenate([X_tr_base, vpr_tr, vlc_tr, apr_tr, alc_tr], axis=1)
X_va = np.concatenate([X_va_base, vpr_va, vlc_va, apr_va, alc_va], axis=1)
X_te = np.concatenate([X_te_base, vpr_te, vlc_te, apr_te, alc_te], axis=1)

print(f"Extended feature shape: {X_tr.shape}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Load random-exposure log for unbiased evaluation (valid window only)
# ---------------------------------------------------------------------------
def load_random_valid(data_dir, valid_start=20220422, valid_end=20220428):
    """Load random-exposure log restricted to validation date window."""
    video_author = {}
    vf_path = os.path.join(data_dir, 'video_features_basic_pure.csv')
    with open(vf_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_author[row['video_id']] = row['author_id']
    
    rand_path = os.path.join(data_dir, 'log_random_4_22_to_5_08_pure.csv')
    rows = []
    with open(rand_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = int(row['date'])
            if date < valid_start or date > valid_end:
                continue
            video_id = row['video_id']
            author_id = video_author.get(video_id, 'UNK')
            duration_ms = float(row['duration_ms']) if row['duration_ms'] else 0.0
            long_view = int(row['long_view'])
            rows.append((
                date,
                row['user_id'],
                video_id,
                author_id,
                row['tab'],
                duration_ms,
                long_view
            ))
    return rows

rand_rows = load_random_valid(args.data_dir)
print(f"Random-exposure valid rows: {len(rand_rows)}", file=sys.stderr)

# Encode random rows using encode() with a temp split, then extend
splits_with_rand = dict(splits)
splits_with_rand['random_valid'] = rand_rows
(enc2, dim2) = encode(splits_with_rand)
X_rand_base, y_rand, users_rand = enc2['random_valid']

# Note: dim2 should equal dim_base since no new vocab from rand rows
assert dim2 == dim_base, f"Dim mismatch: {dim2} vs {dim_base}"

vpr_rand, vlc_rand, apr_rand, alc_rand = make_extra_cols(rand_rows)
X_rand = np.concatenate([X_rand_base, vpr_rand, vlc_rand, apr_rand, alc_rand], axis=1)

print(f"Random valid encoded: {X_rand.shape}, positives: {y_rand.sum()}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
rng = np.random.default_rng(args.seed)

model = FM(dim_extended, k=16, lr=0.001, l2=1e-6, seed=args.seed)

batch_size = 8192
n_train = len(y_tr)
best_val_primary = -1.0
best_weights = None
patience = 4
no_improve = 0
max_epochs = 50

for epoch in range(1, max_epochs + 1):
    idx = rng.permutation(n_train)
    X_shuf = X_tr[idx]
    y_shuf = y_tr[idx]
    
    losses = []
    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        loss = model.step(X_shuf[start:end], y_shuf[start:end])
        losses.append(loss)
    
    train_scores = model.predict(X_tr)
    val_scores   = model.predict(X_va)
    
    train_res = evaluate(users_tr, y_tr, train_scores)
    val_res   = evaluate(users_va, y_va, val_scores)
    
    print(f"Epoch {epoch:02d} | loss={np.mean(losses):.4f} | "
          f"train={train_res['primary']:.4f} | val={val_res['primary']:.4f} | "
          f"gauc={val_res['GAUC']:.4f} | ndcg5={val_res['nDCG@5']:.4f}",
          file=sys.stderr)
    
    if val_res['primary'] > best_val_primary + 1e-6:
        best_val_primary = val_res['primary']
        best_weights = (model.V.copy(), model.W.copy(), float(model.b))
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}", file=sys.stderr)
            break

# Restore best weights
model.V[:] = best_weights[0]
model.W[:] = best_weights[1]
model.b    = best_weights[2]

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------
train_scores = model.predict(X_tr)
val_scores   = model.predict(X_va)
te_scores    = model.predict(X_te)
rand_scores  = model.predict(X_rand)

train_res = evaluate(users_tr, y_tr, train_scores)
val_res   = evaluate(users_va, y_va, val_scores)
rand_res  = evaluate(users_rand, y_rand, rand_scores)

print(f"TRAIN_PRIMARY={train_res['primary']:.4f}")
print(f"VAL_GAUC={val_res['GAUC']:.4f}")
print(f"VAL_NDCG5={val_res['nDCG@5']:.4f}")
print(f"VAL_PRIMARY={val_res['primary']:.4f}")
print(f"UNBIASED_PRIMARY={rand_res['primary']:.4f}")

# ---------------------------------------------------------------------------
# Write submissions
# ---------------------------------------------------------------------------
write_submission(
    os.path.join(args.out_dir, 'submission_valid.csv'),
    splits['valid'],
    val_scores
)
write_submission(
    os.path.join(args.out_dir, 'submission_test.csv'),
    splits['test'],
    te_scores
)