#!/usr/bin/env python3
"""
Iteration: Replace category (first_level_category_id) with tag (first value from tag column
in video_features_basic_pure.csv) as the 6th field in the FM encoder.
Fixed NaN handling: check isinstance(tag_str, str) before calling .split().
Tag has ~46 distinct values (vs 38 for category) and different taxonomy.
All other hyperparameters identical to previous solution (BPR, PAIRS_PER_EPOCH=600_000).
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--out_dir', required=True)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
rng = np.random.default_rng(args.seed)

# ── Imports from repo ─────────────────────────────────────────────────────────
from data import load, LABEL
from evaluate import evaluate
import submit

# ── Load raw data ─────────────────────────────────────────────────────────────
splits = load(args.data_dir)

# ── Read video features for author_id and tag ────────────────────────────────
video_feat_path = os.path.join(args.data_dir, 'video_features_basic_pure.csv')
vf = pd.read_csv(video_feat_path, usecols=['video_id', 'author_id', 'tag'])
vf['video_id'] = vf['video_id'].astype(str)
vf['author_id'] = vf['author_id'].astype(str)
vid2author = dict(zip(vf['video_id'], vf['author_id']))

# Extract first tag value from each video's tag list
# Fix NaN handling: check isinstance before .split()
def extract_first_tag(tag_val):
    if not isinstance(tag_val, str):
        return 'UNK_TAG'
    tag_val = tag_val.strip()
    if tag_val == '' or tag_val.lower() == 'nan':
        return 'UNK_TAG'
    parts = tag_val.split(',')
    first = parts[0].strip()
    if first == '':
        return 'UNK_TAG'
    return first

vf['first_tag'] = vf['tag'].apply(extract_first_tag)
vid2tag = dict(zip(vf['video_id'], vf['first_tag']))

# Count distinct tag values
tag_vals = set(vid2tag.values())
print(f"Loaded tags for {len(vid2tag)} videos, {len(tag_vals)} distinct values", file=sys.stderr)
print(f"Tag value counts (top 10): {pd.Series(list(vid2tag.values())).value_counts().head(10).to_dict()}", file=sys.stderr)

# ── Custom encoder including tag field ───────────────────────────────────────
# Fields: user_id, video_id, author_id, tab, dur_bucket, tag

# Compute duration bucket edges from training data
train_rows = splits['train']
train_durations = np.array([r[5] for r in train_rows], dtype=np.float64)
# 10 buckets => 9 interior edges (quantiles at 10%, 20%, ..., 90%)
dur_edges = np.quantile(train_durations, np.linspace(0.1, 0.9, 9))
print(f"Duration bucket edges: {dur_edges}", file=sys.stderr)

def rows_to_features(rows):
    """Convert rows to feature arrays including tag.
    Returns raw_features list, y (N,) float32, users list.
    Fields: 0=user_id, 1=video_id, 2=author_id, 3=tab, 4=dur_bucket, 5=tag
    """
    N = len(rows)
    raw_features = []
    ys = []
    users = []
    for r in rows:
        date, uid, vid, author, tab, dur_ms, label = r
        dur_bucket = int(np.searchsorted(dur_edges, dur_ms, side='right') - 1)
        dur_bucket = max(0, min(9, dur_bucket))
        tag = vid2tag.get(str(vid), 'UNK_TAG')
        raw_features.append((str(uid), str(vid), str(author), str(tab), str(dur_bucket), tag))
        ys.append(float(label))
        users.append(str(uid))
    return raw_features, np.array(ys, dtype=np.float32), users

# Build vocabularies from training data
print("Building vocabularies...", file=sys.stderr)
train_raw, ytr, utr = rows_to_features(splits['train'])

# 6 fields
field_names = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'tag']
n_fields = 6

# Build vocab for each field from training data
vocabs = []
for fi in range(n_fields):
    vals = set(r[fi] for r in train_raw)
    vals.discard('UNK')
    vocab = {'UNK': 0}
    for i, v in enumerate(sorted(vals), 1):
        vocab[v] = i
    vocabs.append(vocab)

# Compute offsets for shared embedding table
offsets = [0]
for fi in range(n_fields - 1):
    offsets.append(offsets[-1] + len(vocabs[fi]))
total_dim = offsets[-1] + len(vocabs[-1])

print(f"Vocab sizes per field: {[len(v) for v in vocabs]}", file=sys.stderr)
print(f"Offsets: {offsets}", file=sys.stderr)
print(f"Total embedding dim: {total_dim}", file=sys.stderr)

def encode_raw(raw_features, ys, users):
    """Encode raw feature tuples to integer indices with offsets."""
    N = len(raw_features)
    X = np.zeros((N, n_fields), dtype=np.int32)
    for fi in range(n_fields):
        vocab = vocabs[fi]
        offset = offsets[fi]
        for i, r in enumerate(raw_features):
            val = r[fi]
            idx = vocab.get(val, 0)  # 0 = UNK
            X[i, fi] = offset + idx
    return X, ys, users

Xtr, ytr, utr = encode_raw(train_raw, ytr, utr)

valid_raw, yv, uv = rows_to_features(splits['valid'])
Xv, yv, uv = encode_raw(valid_raw, yv, uv)

test_raw, yt, ut = rows_to_features(splits['test'])
Xt, yt, ut = encode_raw(test_raw, yt, ut)

dim = total_dim
print(f"Train: {Xtr.shape[0]} rows, Valid: {Xv.shape[0]} rows, Test: {Xt.shape[0]} rows", file=sys.stderr)
print(f"Embedding table size: {dim}", file=sys.stderr)

# ── Build BPR pairs from training data ────────────────────────────────────────
print("Building BPR pairs...", file=sys.stderr)

user_to_pos = {}
user_to_neg = {}

for i, (uid, label) in enumerate(zip(utr, ytr)):
    if label == 1:
        if uid not in user_to_pos:
            user_to_pos[uid] = []
        user_to_pos[uid].append(i)
    else:
        if uid not in user_to_neg:
            user_to_neg[uid] = []
        user_to_neg[uid].append(i)

valid_users = [u for u in user_to_pos if u in user_to_neg]
print(f"Users with both pos/neg: {len(valid_users)}", file=sys.stderr)

user_pos_arrays = {u: np.array(user_to_pos[u]) for u in valid_users}
user_neg_arrays = {u: np.array(user_to_neg[u]) for u in valid_users}

user_weights = np.array([min(len(user_pos_arrays[u]), len(user_neg_arrays[u])) 
                         for u in valid_users], dtype=np.float64)
user_weights /= user_weights.sum()
valid_users_arr = np.array(valid_users)

print(f"Total BPR-eligible users: {len(valid_users)}", file=sys.stderr)

# ── FM model with BPR training ────────────────────────────────────────────────
k = 16
lr = 0.001
l2 = 1e-6

np.random.seed(args.seed)

V = np.random.normal(0, 0.01, (dim, k)).astype(np.float64)
W = np.zeros(dim, dtype=np.float64)
b = np.float64(0.0)

V_m = np.zeros_like(V)
V_v = np.zeros_like(V)
W_m = np.zeros_like(W)
W_v = np.zeros_like(W)
b_m = np.float64(0.0)
b_v = np.float64(0.0)
adam_t = 0
beta1, beta2, eps_adam = 0.9, 0.999, 1e-8


def fm_score_batch(X):
    """Compute FM scores for a batch. X: (B, F) int32"""
    emb = V[X]  # (B, F, k)
    w = W[X]    # (B, F)
    
    z = b + w.sum(axis=1)
    
    sum_emb = emb.sum(axis=1)        # (B, k)
    sum_sq  = (emb * emb).sum(axis=1)  # (B, k)
    z += 0.5 * ((sum_emb * sum_emb) - sum_sq).sum(axis=1)
    
    return z, emb, w


def bpr_step(X_pos, X_neg):
    """One BPR step on a batch of (pos, neg) pairs."""
    global V, W, b, V_m, V_v, W_m, W_v, b_m, b_v, adam_t
    
    B = X_pos.shape[0]
    
    z_pos, emb_pos, w_pos = fm_score_batch(X_pos)
    z_neg, emb_neg, w_neg = fm_score_batch(X_neg)
    
    diff = z_pos - z_neg
    sig = 1.0 / (1.0 + np.exp(-np.clip(diff, -30, 30)))
    grad_diff = -(1.0 - sig)
    
    loss = -np.log(sig + 1e-10).mean()
    
    g_pos = grad_diff / B
    g_neg = -grad_diff / B
    
    adam_t += 1
    
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    db = np.float64(0.0)
    
    for sign, X, emb, g in [(1.0, X_pos, emb_pos, g_pos), (-1.0, X_neg, emb_neg, g_neg)]:
        sum_emb = emb.sum(axis=1)  # (B, k)
        
        for fi in range(n_fields):
            xi = X[:, fi]
            v_grad = (sum_emb - emb[:, fi, :]) * g[:, np.newaxis]
            w_grad = g
            
            np.add.at(dV, xi, v_grad)
            np.add.at(dW, xi, w_grad)
        
        db += g.sum()
    
    used_pos = X_pos.flatten()
    used_neg = X_neg.flatten()
    used = np.unique(np.concatenate([used_pos, used_neg]))
    dV[used] += l2 * V[used]
    dW[used] += l2 * W[used]
    
    V_m[used] = beta1 * V_m[used] + (1 - beta1) * dV[used]
    V_v[used] = beta2 * V_v[used] + (1 - beta2) * dV[used]**2
    m_hat = V_m[used] / (1 - beta1**adam_t)
    v_hat = V_v[used] / (1 - beta2**adam_t)
    V[used] -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)
    
    W_m[used] = beta1 * W_m[used] + (1 - beta1) * dW[used]
    W_v[used] = beta2 * W_v[used] + (1 - beta2) * dW[used]**2
    m_hat = W_m[used] / (1 - beta1**adam_t)
    v_hat = W_v[used] / (1 - beta2**adam_t)
    W[used] -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)
    
    b_m = beta1 * b_m + (1 - beta1) * db
    b_v = beta2 * b_v + (1 - beta2) * db**2
    m_hat = b_m / (1 - beta1**adam_t)
    v_hat = b_v / (1 - beta2**adam_t)
    b -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)
    
    return loss


def predict(X, bs=200_000):
    """Predict scores for all rows."""
    N = X.shape[0]
    scores = np.zeros(N, dtype=np.float64)
    for start in range(0, N, bs):
        end = min(start + bs, N)
        z, _, _ = fm_score_batch(X[start:end])
        scores[start:end] = z
    return scores


# ── Training loop with BPR ────────────────────────────────────────────────────
BATCH = 4096
PAIRS_PER_EPOCH = 600_000
STEPS_PER_EPOCH = PAIRS_PER_EPOCH // BATCH
PATIENCE = 4
best_val_primary = -1.0
best_epoch = -1
patience_count = 0

best_V = V.copy()
best_W = W.copy()
best_b = float(b)

print(f"Steps per epoch: {STEPS_PER_EPOCH}, batch size: {BATCH}", file=sys.stderr)

for epoch in range(1, 51):
    losses = []
    
    for step in range(STEPS_PER_EPOCH):
        chosen_users = rng.choice(len(valid_users_arr), size=BATCH, p=user_weights)
        
        pos_indices = np.zeros(BATCH, dtype=np.int64)
        neg_indices = np.zeros(BATCH, dtype=np.int64)
        
        for j, ui in enumerate(chosen_users):
            u = valid_users_arr[ui]
            pos_arr = user_pos_arrays[u]
            neg_arr = user_neg_arrays[u]
            pos_indices[j] = rng.choice(pos_arr)
            neg_indices[j] = rng.choice(neg_arr)
        
        X_pos = Xtr[pos_indices]
        X_neg = Xtr[neg_indices]
        
        loss = bpr_step(X_pos, X_neg)
        losses.append(loss)
    
    mean_loss = np.mean(losses)
    
    val_scores = predict(Xv)
    val_metrics = evaluate(uv, yv, val_scores)
    val_primary = val_metrics['primary']
    
    print(f"Epoch {epoch:3d} | loss={mean_loss:.4f} | val_primary={val_primary:.4f}", file=sys.stderr)
    
    if val_primary > best_val_primary + 1e-6:
        best_val_primary = val_primary
        best_epoch = epoch
        patience_count = 0
        best_V = V.copy()
        best_W = W.copy()
        best_b = float(b)
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"Early stop at epoch {epoch}, best was {best_epoch}", file=sys.stderr)
            break

# Restore best weights
V = best_V
W = best_W
b = best_b

# ── Final evaluation ──────────────────────────────────────────────────────────
train_scores = predict(Xtr)
train_metrics = evaluate(utr, ytr, train_scores)

val_scores = predict(Xv)
val_metrics = evaluate(uv, yv, val_scores)

# ── Unbiased evaluation on random-exposure log ────────────────────────────────
random_log_path = os.path.join(args.data_dir, 'log_random_4_22_to_5_08_pure.csv')

rlog = pd.read_csv(random_log_path)
rlog['date'] = rlog['date'].astype(int)
rlog_valid = rlog[(rlog['date'] >= 20220422) & (rlog['date'] <= 20220428)].copy()
rlog_valid['video_id'] = rlog_valid['video_id'].astype(str)
rlog_valid['user_id'] = rlog_valid['user_id'].astype(str)
rlog_valid['author_id'] = rlog_valid['video_id'].map(vid2author).fillna('UNK')

rand_rows_list = []
for _, r in rlog_valid.iterrows():
    rand_rows_list.append((
        int(r['date']),
        str(r['user_id']),
        str(r['video_id']),
        str(r.get('author_id', 'UNK')),
        str(r['tab']),
        float(r['duration_ms']),
        int(r['long_view'])
    ))

rand_raw, yrand, urand = rows_to_features(rand_rows_list)
Xrand, yrand, urand = encode_raw(rand_raw, yrand, urand)

rand_scores = predict(Xrand)
rand_metrics = evaluate(urand, yrand, rand_scores)
unbiased_primary = rand_metrics['primary']

# ── Print results ──────────────────────────────────────────────────────────────
print(f"TRAIN_PRIMARY={train_metrics['primary']:.4f}")
print(f"VAL_GAUC={val_metrics['GAUC']:.4f}")
print(f"VAL_NDCG5={val_metrics['nDCG@5']:.4f}")
print(f"VAL_PRIMARY={val_metrics['primary']:.4f}")
print(f"UNBIASED_PRIMARY={unbiased_primary:.4f}")

# ── Write submissions ─────────────────────────────────────────────────────────
submit.write_submission(
    os.path.join(args.out_dir, 'submission_valid.csv'),
    splits['valid'],
    val_scores
)

test_scores = predict(Xt)
submit.write_submission(
    os.path.join(args.out_dir, 'submission_test.csv'),
    splits['test'],
    test_scores
)

print("Done.", file=sys.stderr)