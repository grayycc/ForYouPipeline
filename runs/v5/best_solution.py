#!/usr/bin/env python3
"""FM with video_type and tag features from video_features_basic_pure.csv — iteration 1."""

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data")
parser.add_argument("--out_dir", default="out")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--train_split", default="train", choices=["train", "train+valid"])
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
from data import load, LABEL
from evaluate import evaluate
from submit import write_submission
from unbiased import load_random_valid, unbiased_primary

data_dir = args.data_dir
splits = load(data_dir)

# ---------------------------------------------------------------------------
# Load video_features_basic_pure.csv for video_type and tag
# ---------------------------------------------------------------------------
video_feat_path = os.path.join(data_dir, "video_features_basic_pure.csv")
vf = pd.read_csv(video_feat_path, dtype=str)

# Build video_id -> video_type mapping
# video_type is a categorical column
video_type_map = {}
video_tag_map = {}
for _, row in vf.iterrows():
    vid = str(row["video_id"])
    vtype = str(row.get("video_type", "UNK")) if pd.notna(row.get("video_type")) else "UNK"
    # tag is multi-valued like '20,43'; take the first tag as a categorical feature
    tag_raw = str(row.get("tag", "UNK")) if pd.notna(row.get("tag")) else "UNK"
    if tag_raw and tag_raw != "nan" and tag_raw != "UNK":
        first_tag = tag_raw.split(",")[0].strip()
    else:
        first_tag = "UNK"
    video_type_map[vid] = vtype
    video_tag_map[vid] = first_tag

# ---------------------------------------------------------------------------
# Custom encoder that adds video_type and first_tag to FM features
# ---------------------------------------------------------------------------
# Fields: user_id, video_id, author_id, tab, dur_bucket, video_type, first_tag
# We replicate the encoding logic from data.encode() but add two more fields.

def make_dur_bucket(duration_ms, edges):
    """Bucket duration into bin index using precomputed edges."""
    return int(np.searchsorted(edges, duration_ms, side='right'))

def build_vocab_and_encode(train_rows, target_rows_dict, video_type_map, video_tag_map):
    """
    Build vocabulary from train_rows, then encode train + each split in target_rows_dict.
    Fields: user_id, video_id, author_id, tab, dur_bucket, video_type, first_tag
    Returns: enc dict, total dim
    """
    # Compute duration quantile edges from training data
    train_durations = np.array([r[5] for r in train_rows], dtype=np.float64)
    # 10 buckets via quantiles (same as data.py approach)
    quantiles = np.linspace(0, 100, 11)[1:-1]  # 9 edges -> 10 buckets
    dur_edges = np.percentile(train_durations, quantiles)

    # Build vocabularies from training data
    fields = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'video_type', 'first_tag']
    vocab = {f: {} for f in fields}
    
    def get_dur_bucket_str(duration_ms):
        return str(int(np.searchsorted(dur_edges, duration_ms, side='right')))
    
    def row_to_features(row):
        vid = str(row[2])
        return {
            'user_id': str(row[1]),
            'video_id': vid,
            'author_id': str(row[3]),
            'tab': str(row[4]),
            'dur_bucket': get_dur_bucket_str(row[5]),
            'video_type': video_type_map.get(vid, "UNK"),
            'first_tag': video_tag_map.get(vid, "UNK"),
        }
    
    # Build vocab from train
    for row in train_rows:
        feats = row_to_features(row)
        for f in fields:
            val = feats[f]
            if val not in vocab[f]:
                vocab[f][val] = len(vocab[f])
    
    # Add UNK slot for each field (for unseen values)
    unk_ids = {}
    offsets = {}
    offset = 0
    for f in fields:
        offsets[f] = offset
        n = len(vocab[f])
        unk_ids[f] = offset + n  # UNK is one beyond the known values
        offset += n + 1  # +1 for UNK slot
    
    total_dim = offset
    
    def encode_rows(rows):
        N = len(rows)
        X = np.zeros((N, len(fields)), dtype=np.int32)
        y = np.zeros(N, dtype=np.float32)
        users = []
        for i, row in enumerate(rows):
            feats = row_to_features(row)
            for j, f in enumerate(fields):
                val = feats[f]
                if val in vocab[f]:
                    X[i, j] = offsets[f] + vocab[f][val]
                else:
                    X[i, j] = unk_ids[f]
            y[i] = float(row[6])
            users.append(str(row[1]))
        return X, y, users
    
    enc = {}
    for split_name, rows in target_rows_dict.items():
        X, y, users = encode_rows(rows)
        enc[split_name] = (X, y, users)
    
    return enc, total_dim, dur_edges, vocab, offsets, unk_ids, fields

# Build encoding
all_splits = {
    'train': splits['train'],
    'valid': splits['valid'],
    'test': splits['test'],
}

enc, dim, dur_edges, vocab, offsets, unk_ids, fields = build_vocab_and_encode(
    splits['train'], all_splits, video_type_map, video_tag_map
)

X_train, y_train, users_train = enc["train"]
X_valid, y_valid, users_valid = enc["valid"]
X_test,  y_test,  users_test  = enc["test"]

print(f"Feature dim: {dim}, fields: {fields}", flush=True)

if args.train_split == "train+valid":
    X_tv = np.concatenate([X_train, X_valid], axis=0)
    y_tv = np.concatenate([y_train, y_valid], axis=0)

# ---------------------------------------------------------------------------
# FM model (imported from baseline)
# ---------------------------------------------------------------------------
from baseline import FM

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
BATCH_SIZE = 8192
K = 16
LR = 0.001
L2 = 1e-6
PATIENCE = 4
MAX_EPOCHS = 50

def run_epoch(model, X, y, batch_size, rng):
    """One full pass over data, shuffled."""
    N = len(y)
    idx = rng.permutation(N)
    total_loss = 0.0
    n_batches = 0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bx = X[idx[start:end]]
        by = y[idx[start:end]]
        loss = model.step(bx, by)
        total_loss += loss
        n_batches += 1
    return total_loss / n_batches

def score_split(model, X, y, users):
    scores = model.predict(X)
    result = evaluate(users, y.tolist(), scores.tolist())
    return result

# ---------------------------------------------------------------------------
# Train or train+valid
# ---------------------------------------------------------------------------
if args.train_split == "train+valid":
    FIXED_EPOCHS = 10
    model = FM(dim, k=K, lr=LR, l2=L2, seed=args.seed)
    for ep in range(FIXED_EPOCHS):
        loss = run_epoch(model, X_tv, y_tv, BATCH_SIZE, rng)
        print(f"[train+valid] epoch {ep+1}/{FIXED_EPOCHS}  loss={loss:.4f}", flush=True)
    # Write test submission only
    test_scores = model.predict(X_test)
    write_submission(os.path.join(args.out_dir, "submission_test.csv"),
                     splits["test"], test_scores.tolist())
    sys.exit(0)

# ---------------------------------------------------------------------------
# Normal training with early stopping on validation primary
# ---------------------------------------------------------------------------
model = FM(dim, k=K, lr=LR, l2=L2, seed=args.seed)

best_val_primary = -1.0
best_epoch = 0
patience_count = 0
best_V = None
best_W = None
best_b = None

for ep in range(MAX_EPOCHS):
    t0 = time.time()
    loss = run_epoch(model, X_train, y_train, BATCH_SIZE, rng)
    val_res = score_split(model, X_valid, y_valid, users_valid)
    val_primary = val_res["primary"]
    elapsed = time.time() - t0
    print(f"epoch {ep+1:2d}  loss={loss:.4f}  val_primary={val_primary:.4f}  "
          f"({elapsed:.1f}s)", flush=True)

    if val_primary > best_val_primary + 1e-6:
        best_val_primary = val_primary
        best_epoch = ep + 1
        patience_count = 0
        best_V = model.V.copy()
        best_W = model.W.copy()
        best_b = float(model.b)
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"Early stop at epoch {ep+1}, best epoch {best_epoch}", flush=True)
            break

# Restore best weights
model.V[:] = best_V
model.W[:] = best_W
model.b = best_b

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------
train_res = score_split(model, X_train, y_train, users_train)
val_res   = score_split(model, X_valid, y_valid, users_valid)

# Unbiased evaluation — encode random rows using the same vocab
rand_rows = load_random_valid(data_dir)

def encode_rand_rows(rand_rows):
    """Encode random-exposure rows using the same vocabulary built from training."""
    def get_dur_bucket_str(duration_ms):
        return str(int(np.searchsorted(dur_edges, duration_ms, side='right')))
    
    N = len(rand_rows)
    X = np.zeros((N, len(fields)), dtype=np.int32)
    y = np.zeros(N, dtype=np.float32)
    users = []
    for i, row in enumerate(rand_rows):
        vid = str(row[2])
        feats = {
            'user_id': str(row[1]),
            'video_id': vid,
            'author_id': str(row[3]),
            'tab': str(row[4]),
            'dur_bucket': get_dur_bucket_str(row[5]),
            'video_type': video_type_map.get(vid, "UNK"),
            'first_tag': video_tag_map.get(vid, "UNK"),
        }
        for j, f in enumerate(fields):
            val = feats[f]
            if val in vocab[f]:
                X[i, j] = offsets[f] + vocab[f][val]
            else:
                X[i, j] = unk_ids[f]
        y[i] = float(row[6])
        users.append(str(row[1]))
    return X, y, users

X_rand, y_rand, u_rand = encode_rand_rows(rand_rows)

# Use unbiased_primary with a custom scoring function
unbiased = unbiased_primary(
    data_dir, splits["train"],
    lambda rows: model.predict(X_rand)
)

print(f"TRAIN_PRIMARY={train_res['primary']:.4f}")
print(f"VAL_GAUC={val_res['GAUC']:.4f}")
print(f"VAL_NDCG5={val_res['nDCG@5']:.4f}")
print(f"VAL_PRIMARY={val_res['primary']:.4f}")
print(f"UNBIASED_PRIMARY={unbiased:.4f}")

# ---------------------------------------------------------------------------
# Write submissions
# ---------------------------------------------------------------------------
valid_scores = model.predict(X_valid)
write_submission(os.path.join(args.out_dir, "submission_valid.csv"),
                 splits["valid"], valid_scores.tolist())

test_scores = model.predict(X_test)
write_submission(os.path.join(args.out_dir, "submission_test.csv"),
                 splits["test"], test_scores.tolist())