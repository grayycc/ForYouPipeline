#!/usr/bin/env python3
"""
BPR pairwise loss + 5-seed ensemble + Bayesian-smoothed video/author CTR buckets.

Adds two new FM features:
  1. video_ctr_bucket: Bayesian-smoothed long_view rate for each video_id (prior=50)
  2. author_ctr_bucket: Bayesian-smoothed long_view rate for each author_id (prior=50)
Both computed from training rows only, discretized into 10 buckets.

Fix applied (reviewer's note): bucket edges are now computed from per-unique-entity rates
rather than per-row rates, so frequent entities don't dominate the quantile calculation.
"""
import argparse
import os
import sys
import time
import numpy as np
from collections import defaultdict
from scipy.stats import rankdata

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--out_dir', required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--train_split', default='train', choices=['train', 'train+valid'])
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── imports from repo ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load, encode
from evaluate import evaluate
from submit import write_submission

# ── unbiased loader ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kit'))
try:
    from unbiased import load_random_valid, encode_like_train
    _has_unbiased = True
except ImportError:
    _has_unbiased = False


# ── Bayesian-smoothed CTR feature computation ─────────────────────────────────
def compute_smoothed_ctr_buckets(train_rows, all_rows_list, key_fn, n_buckets=10, prior=50.0):
    """
    Compute Bayesian-smoothed long_view rate for each entity (video or author).
    
    Args:
        train_rows: list of training row tuples (for computing CTR stats)
        all_rows_list: list of (split_name, rows) to encode
        key_fn: function(row) -> entity key (e.g., video_id or author_id)
        n_buckets: number of discretization buckets
        prior: prior strength (pseudo-count toward global mean)
    
    Returns:
        dict mapping split_name -> np.array of bucket indices (0..n_buckets-1, int32)
        Also returns the number of buckets used (for dim extension)
    """
    # Compute global mean from training rows
    # Row structure: (date, user_id, video_id, author_id, tab, duration_ms, long_view)
    total_lv = sum(row[6] for row in train_rows)
    global_mean = total_lv / len(train_rows) if train_rows else 0.5

    # Accumulate counts per entity from training rows
    entity_sum = defaultdict(float)
    entity_count = defaultdict(int)
    for row in train_rows:
        k = key_fn(row)
        entity_sum[k] += row[6]  # long_view label
        entity_count[k] += 1

    # Compute smoothed rate for each entity
    # smoothed_rate = (sum + prior * global_mean) / (count + prior)
    entity_rate = {}
    for k in entity_sum:
        entity_rate[k] = (entity_sum[k] + prior * global_mean) / (entity_count[k] + prior)

    # FIX: Compute quantile-based bucket edges from **unique entity** rates,
    # so each entity contributes once regardless of how many training rows it has.
    # This prevents frequent entities from dominating the quantile calculation.
    unique_rates = np.array([entity_rate[k] for k in entity_rate], dtype=np.float32)

    # Compute quantile-based bucket edges from unique entity rates
    quantiles = np.linspace(0, 100, n_buckets + 1)
    edges = np.percentile(unique_rates, quantiles)
    # Make edges slightly wider to handle boundary cases
    edges[0] -= 1e-6
    edges[-1] += 1e-6

    # Encode all splits
    result = {}
    for split_name, rows in all_rows_list:
        rates = np.array([
            entity_rate.get(key_fn(row), global_mean)
            for row in rows
        ], dtype=np.float32)
        # Digitize into buckets (0..n_buckets-1)
        buckets = np.digitize(rates, edges[1:-1]).astype(np.int32)  # 0..n_buckets-1
        buckets = np.clip(buckets, 0, n_buckets - 1)
        result[split_name] = buckets

    return result, n_buckets


def augment_features(X_tr, X_va, X_te, train_rows, valid_rows, test_rows, base_dim):
    """
    Add video_ctr_bucket and author_ctr_bucket as new FM feature columns.
    Each gets its own offset in the embedding table.

    Returns:
        X_tr_aug, X_va_aug, X_te_aug: augmented feature matrices
        new_dim: new total embedding table size
    """
    # Video CTR bucket (key: video_id at index 2)
    video_ctr_splits, n_video_buckets = compute_smoothed_ctr_buckets(
        train_rows,
        [('train', train_rows), ('valid', valid_rows), ('test', test_rows)],
        key_fn=lambda row: row[2],  # video_id
        n_buckets=10,
        prior=50.0
    )

    # Author CTR bucket (key: author_id at index 3)
    author_ctr_splits, n_author_buckets = compute_smoothed_ctr_buckets(
        train_rows,
        [('train', train_rows), ('valid', valid_rows), ('test', test_rows)],
        key_fn=lambda row: row[3],  # author_id
        n_buckets=10,
        prior=50.0
    )

    # Assign offsets for new features
    video_offset = base_dim
    author_offset = base_dim + n_video_buckets
    new_dim = base_dim + n_video_buckets + n_author_buckets

    # Apply offsets
    video_tr = video_ctr_splits['train'] + video_offset
    video_va = video_ctr_splits['valid'] + video_offset
    video_te = video_ctr_splits['test'] + video_offset

    author_tr = author_ctr_splits['train'] + author_offset
    author_va = author_ctr_splits['valid'] + author_offset
    author_te = author_ctr_splits['test'] + author_offset

    # Append to feature matrices
    X_tr_aug = np.concatenate([X_tr, video_tr[:, None], author_tr[:, None]], axis=1)
    X_va_aug = np.concatenate([X_va, video_va[:, None], author_va[:, None]], axis=1)
    X_te_aug = np.concatenate([X_te, video_te[:, None], author_te[:, None]], axis=1)

    return X_tr_aug, X_va_aug, X_te_aug, new_dim


# ── FM with BPR ──────────────────────────────────────────────────────────────
class FMwithBPR:
    """
    Factorization Machine with BPR pairwise loss and Adam optimizer.
    """
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.k = k
        self.lr = lr
        self.l2 = l2
        self.dim = dim

        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)

        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def _fm_forward(self, X):
        B, F = X.shape
        E = self.V[X]
        first = self.W[X].sum(axis=1)
        emb_sum = E.sum(axis=1)
        emb_sq_sum = (E ** 2).sum(axis=1)
        interaction = 0.5 * ((emb_sum ** 2) - emb_sq_sum).sum(axis=1)
        z = self.b + first + interaction
        return z, emb_sum, emb_sq_sum, E

    def predict(self, X, bs=200_000):
        scores = []
        for start in range(0, len(X), bs):
            xb = X[start:start+bs]
            z, _, _, _ = self._fm_forward(xb)
            scores.append(z)
        return np.concatenate(scores)

    def bpr_step(self, X_pos, X_neg):
        P = len(X_pos)
        if P == 0:
            return 0.0

        self.t += 1

        z_pos, sum_pos, sq_pos, E_pos = self._fm_forward(X_pos)
        z_neg, sum_neg, sq_neg, E_neg = self._fm_forward(X_neg)

        diff = z_pos - z_neg
        sig_diff = 1.0 / (1.0 + np.exp(-diff.clip(-30, 30)))
        loss = -np.log(sig_diff + 1e-10).mean()

        g_pos = (sig_diff - 1.0) / P
        g_neg = -g_pos

        F = X_pos.shape[1]

        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)

        for sign, X_batch, emb_sum_batch, E_batch, g_batch in [
            (1.0, X_pos, sum_pos, E_pos, g_pos),
            (-1.0, X_neg, sum_neg, E_neg, g_neg),
        ]:
            g_col = g_batch[:, np.newaxis, np.newaxis]
            emb_sum_expanded = emb_sum_batch[:, np.newaxis, :]
            dV_contrib = g_col * (emb_sum_expanded - E_batch)

            flat_idx = X_batch.reshape(-1)
            flat_dV = dV_contrib.reshape(-1, self.k)
            np.add.at(grad_V, flat_idx, flat_dV)

            dW_contrib = g_batch[:, np.newaxis] * np.ones((P, F), dtype=np.float32)
            flat_dW = dW_contrib.reshape(-1)
            np.add.at(grad_W, flat_idx, flat_dW)

        grad_V += self.l2 * self.V
        grad_W += self.l2 * self.W

        self.mV = self.beta1 * self.mV + (1 - self.beta1) * grad_V
        self.vV = self.beta2 * self.vV + (1 - self.beta2) * (grad_V ** 2)
        mV_hat = self.mV / (1 - self.beta1 ** self.t)
        vV_hat = self.vV / (1 - self.beta2 ** self.t)
        self.V -= self.lr * mV_hat / (np.sqrt(vV_hat) + self.eps)

        self.mW = self.beta1 * self.mW + (1 - self.beta1) * grad_W
        self.vW = self.beta2 * self.vW + (1 - self.beta2) * (grad_W ** 2)
        mW_hat = self.mW / (1 - self.beta1 ** self.t)
        vW_hat = self.vW / (1 - self.beta2 ** self.t)
        self.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)

        grad_b = g_pos.sum() + g_neg.sum()
        self.b -= self.lr * grad_b

        return float(loss)


def build_bpr_pairs(X, y, users, rng):
    user_pos = defaultdict(list)
    user_neg = defaultdict(list)
    for i, (u, label) in enumerate(zip(users, y)):
        if label > 0.5:
            user_pos[u].append(i)
        else:
            user_neg[u].append(i)

    pos_indices = []
    neg_indices = []
    for u in user_pos:
        if u in user_neg:
            pi = rng.integers(0, len(user_pos[u]))
            ni = rng.integers(0, len(user_neg[u]))
            pos_indices.append(user_pos[u][pi])
            neg_indices.append(user_neg[u][ni])

    if not pos_indices:
        return None, None

    pos_indices = np.array(pos_indices, dtype=np.int64)
    neg_indices = np.array(neg_indices, dtype=np.int64)
    return X[pos_indices], X[neg_indices]


def within_user_ranks(scores, users):
    ranks = np.zeros(len(scores), dtype=np.float64)
    user_idx = defaultdict(list)
    for i, u in enumerate(users):
        user_idx[u].append(i)
    for u, idxs in user_idx.items():
        idxs = np.array(idxs)
        s = scores[idxs]
        r = rankdata(s, method='average')
        n = len(r)
        if n > 1:
            r = (r - 1.0) / (n - 1.0)
        else:
            r = np.array([0.5])
        ranks[idxs] = r
    return ranks


# ── load data ────────────────────────────────────────────────────────────────
t0 = time.time()
splits = load(args.data_dir)
enc, base_dim = encode(splits)
X_tr_base, y_tr, u_tr = enc['train']
X_va_base, y_va, u_va = enc['valid']
X_te_base, y_te, u_te = enc['test']
print(f"Data loaded in {time.time()-t0:.1f}s. base_dim={base_dim}")

# ── compute CTR bucket features ───────────────────────────────────────────────
t1 = time.time()
train_rows = splits['train']
valid_rows = splits['valid']
test_rows = splits['test']

X_tr, X_va, X_te, dim = augment_features(
    X_tr_base, X_va_base, X_te_base,
    train_rows, valid_rows, test_rows,
    base_dim
)
print(f"CTR bucket features added in {time.time()-t1:.1f}s. new dim={dim}, "
      f"features: {X_tr.shape[1]} cols (was {X_tr_base.shape[1]})")

N_ENSEMBLE = 5
ENSEMBLE_SEEDS = [args.seed + i for i in range(N_ENSEMBLE)]

BS = 8192
MAX_EPOCHS = 30
PATIENCE = 4


def train_bpr_epoch(model, X, y, users, seed, epoch):
    rng = np.random.default_rng(seed * 1000 + epoch)
    N = len(y)
    perm = rng.permutation(N)
    X_shuf = X[perm]
    y_shuf = y[perm]
    u_shuf = [users[i] for i in perm]

    losses = []
    for start in range(0, N, BS):
        xb = X_shuf[start:start+BS]
        yb = y_shuf[start:start+BS]
        ub = u_shuf[start:start+BS]

        X_pos, X_neg = build_bpr_pairs(xb, yb, ub, rng)
        if X_pos is None:
            continue
        loss = model.bpr_step(X_pos, X_neg)
        losses.append(loss)

    return np.mean(losses) if losses else 0.0


# ── train+valid mode ──────────────────────────────────────────────────────────
if args.train_split == 'train+valid':
    # For train+valid, use train rows for CTR stats (no leakage from valid labels),
    # but train on combined train+valid rows for the FM.
    X_tv = np.concatenate([X_tr, X_va], axis=0)
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    u_tv = list(u_tr) + list(u_va)
    N_EPOCHS_FIXED = 8

    all_scores_te = []
    for seed in ENSEMBLE_SEEDS:
        model = FMwithBPR(dim, k=16, lr=0.001, l2=1e-6, seed=seed)
        for epoch in range(N_EPOCHS_FIXED):
            train_bpr_epoch(model, X_tv, y_tv, u_tv, seed, epoch)
        scores_te = model.predict(X_te)
        all_scores_te.append(scores_te)

    rank_scores_te = np.zeros(len(y_te), dtype=np.float64)
    for scores in all_scores_te:
        rank_scores_te += within_user_ranks(scores, u_te)
    rank_scores_te /= N_ENSEMBLE

    write_submission(os.path.join(args.out_dir, 'submission_test.csv'),
                     splits['test'], rank_scores_te)
    sys.exit(0)

# ── normal train mode ─────────────────────────────────────────────────────────
all_scores_va = []
all_scores_te = []
all_scores_tr = []
best_epochs = []

for seed in ENSEMBLE_SEEDS:
    print(f"\n--- Training model seed={seed} (BPR + CTR buckets) ---", flush=True)
    model = FMwithBPR(dim, k=16, lr=0.001, l2=1e-6, seed=seed)

    best_primary = -1.0
    best_weights = None
    no_improve = 0
    best_epoch = -1

    for epoch in range(MAX_EPOCHS):
        loss = train_bpr_epoch(model, X_tr, y_tr, u_tr, seed, epoch)

        scores_va = model.predict(X_va)
        res_va = evaluate(u_va, y_va, scores_va)
        primary = res_va['primary']

        print(f"  Epoch {epoch+1:2d}  bpr_loss={loss:.4f}  "
              f"GAUC={res_va['GAUC']:.4f}  nDCG@5={res_va['nDCG@5']:.4f}  "
              f"primary={primary:.4f}", flush=True)

        if primary > best_primary + 1e-6:
            best_primary = primary
            best_epoch = epoch
            best_weights = {
                'V': model.V.copy(),
                'W': model.W.copy(),
                'b': float(model.b),
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1}, best was epoch {best_epoch+1}")
                break

    best_epochs.append(best_epoch + 1)

    model.V = best_weights['V']
    model.W = best_weights['W']
    model.b = np.float32(best_weights['b'])

    all_scores_tr.append(model.predict(X_tr))
    all_scores_va.append(model.predict(X_va))
    all_scores_te.append(model.predict(X_te))
    last_model = model

print(f"\nBest epochs per seed: {best_epochs}")

# ── ensemble: within-user rank averaging ─────────────────────────────────────
print("Computing rank-averaged ensemble scores...", flush=True)

rank_scores_va = np.zeros(len(y_va), dtype=np.float64)
for scores in all_scores_va:
    rank_scores_va += within_user_ranks(scores, u_va)
rank_scores_va /= N_ENSEMBLE

rank_scores_te = np.zeros(len(y_te), dtype=np.float64)
for scores in all_scores_te:
    rank_scores_te += within_user_ranks(scores, u_te)
rank_scores_te /= N_ENSEMBLE

rank_scores_tr = np.zeros(len(y_tr), dtype=np.float64)
for scores in all_scores_tr:
    rank_scores_tr += within_user_ranks(scores, u_tr)
rank_scores_tr /= N_ENSEMBLE

# ── final evaluation ──────────────────────────────────────────────────────────
res_tr = evaluate(u_tr, y_tr, rank_scores_tr)
res_va = evaluate(u_va, y_va, rank_scores_va)

# unbiased evaluation
unbiased_val = 0.0
if _has_unbiased:
    try:
        rand_rows = load_random_valid(args.data_dir)
        # For unbiased eval, augment the random rows with the same CTR bucket features
        # computed from training rows only
        X_rand_base, y_rand, u_rand, _ = encode_like_train(splits['train'], rand_rows)

        # Compute CTR buckets for random rows using same training stats
        video_ctr_rand, _ = compute_smoothed_ctr_buckets(
            train_rows,
            [('rand', rand_rows)],
            key_fn=lambda row: row[2],
            n_buckets=10,
            prior=50.0
        )
        author_ctr_rand, _ = compute_smoothed_ctr_buckets(
            train_rows,
            [('rand', rand_rows)],
            key_fn=lambda row: row[3],
            n_buckets=10,
            prior=50.0
        )

        video_offset = base_dim
        author_offset = base_dim + 10

        video_rand = video_ctr_rand['rand'] + video_offset
        author_rand = author_ctr_rand['rand'] + author_offset

        X_rand = np.concatenate([X_rand_base, video_rand[:, None], author_rand[:, None]], axis=1)

        scores_rand = last_model.predict(X_rand)
        res_unb = evaluate(u_rand, y_rand, scores_rand)
        unbiased_val = res_unb['primary']
    except Exception as e:
        print(f"Unbiased eval failed: {e}", file=sys.stderr)
        unbiased_val = 0.0

# ── write submissions ─────────────────────────────────────────────────────────
write_submission(os.path.join(args.out_dir, 'submission_valid.csv'),
                 splits['valid'], rank_scores_va)
write_submission(os.path.join(args.out_dir, 'submission_test.csv'),
                 splits['test'], rank_scores_te)

# ── print required lines ──────────────────────────────────────────────────────
print(f"TRAIN_PRIMARY={res_tr['primary']:.4f}")
print(f"VAL_GAUC={res_va['GAUC']:.4f}")
print(f"VAL_NDCG5={res_va['nDCG@5']:.4f}")
print(f"VAL_PRIMARY={res_va['primary']:.4f}")
print(f"UNBIASED_PRIMARY={unbiased_val:.4f}")