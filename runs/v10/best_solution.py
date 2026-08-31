#!/usr/bin/env python3
"""
Iteration 4: Replace pointwise BCE with BPR pairwise loss.

Changes from iteration 2/3:
- BPR pairwise training: sample (pos, neg) pairs within each minibatch per user
- Optimize sigmoid(score_pos - score_neg) directly, which aligns with GAUC objective
- Keep: 7 features, 5-seed ensemble, rank averaging, Bayesian-smoothed CTR buckets
"""

import argparse
import os
import sys
import numpy as np

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--out_dir', required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--train_split', default='train',
                    choices=['train', 'train+valid'])
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from data import load, encode, FIELDS
from evaluate import evaluate
import submit
from unbiased import load_random_valid, encode_like_train, unbiased_primary

# ── Data ──────────────────────────────────────────────────────────────────────
splits = load(args.data_dir)

enc, dim = encode(splits)
X_tr_base, y_tr, u_tr = enc['train']
X_va_base, y_va, u_va = enc['valid']
X_te_base, y_te, u_te = enc['test']

# ── Compute Bayesian-smoothed CTR features from training data only ────────────
GLOBAL_MEAN_PRIOR_STRENGTH = 50

def compute_bayesian_ctr(train_rows, key_idx, label_idx=6):
    counts = {}
    for row in train_rows:
        key = row[key_idx]
        label = row[label_idx]
        if key not in counts:
            counts[key] = [0, 0]
        counts[key][0] += 1
        counts[key][1] += label

    total_all = sum(v[0] for v in counts.values())
    pos_all = sum(v[1] for v in counts.values())
    global_rate = pos_all / total_all if total_all > 0 else 0.5

    alpha = GLOBAL_MEAN_PRIOR_STRENGTH
    smoothed = {}
    for key, (total, pos) in counts.items():
        smoothed[key] = (pos + alpha * global_rate) / (total + alpha)

    return smoothed, global_rate


def make_bucket_feature(rows, ctr_map, global_rate, key_idx):
    rates = np.array([
        ctr_map.get(row[key_idx], global_rate)
        for row in rows
    ], dtype=np.float64)
    return rates


def compute_buckets_from_rates(train_rates, query_rates, n_bins=10):
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(train_rates, quantiles)
    edges = np.unique(edges)
    buckets = np.searchsorted(edges[1:-1], query_rates, side='right')
    buckets = np.clip(buckets, 0, n_bins - 1)
    return buckets.astype(np.int32)


train_rows = splits['train']
valid_rows = splits['valid']
test_rows = splits['test']

print("Computing video CTR map...", flush=True)
video_ctr_map, video_global_rate = compute_bayesian_ctr(train_rows, key_idx=2)
print(f"  {len(video_ctr_map)} videos, global rate={video_global_rate:.4f}", flush=True)

print("Computing author CTR map...", flush=True)
author_ctr_map, author_global_rate = compute_bayesian_ctr(train_rows, key_idx=3)
print(f"  {len(author_ctr_map)} authors, global rate={author_global_rate:.4f}", flush=True)

N_BINS = 10

train_video_rates = make_bucket_feature(train_rows, video_ctr_map, video_global_rate, key_idx=2)
train_author_rates = make_bucket_feature(train_rows, author_ctr_map, author_global_rate, key_idx=3)

valid_video_rates = make_bucket_feature(valid_rows, video_ctr_map, video_global_rate, key_idx=2)
valid_author_rates = make_bucket_feature(valid_rows, author_ctr_map, author_global_rate, key_idx=3)

test_video_rates = make_bucket_feature(test_rows, video_ctr_map, video_global_rate, key_idx=2)
test_author_rates = make_bucket_feature(test_rows, author_ctr_map, author_global_rate, key_idx=3)

train_video_buckets = compute_buckets_from_rates(train_video_rates, train_video_rates, N_BINS)
train_author_buckets = compute_buckets_from_rates(train_author_rates, train_author_rates, N_BINS)

valid_video_buckets = compute_buckets_from_rates(train_video_rates, valid_video_rates, N_BINS)
valid_author_buckets = compute_buckets_from_rates(train_author_rates, valid_author_rates, N_BINS)

test_video_buckets = compute_buckets_from_rates(train_video_rates, test_video_rates, N_BINS)
test_author_buckets = compute_buckets_from_rates(train_author_rates, test_author_rates, N_BINS)

VIDEO_CTR_OFFSET = dim
AUTHOR_CTR_OFFSET = dim + N_BINS
NEW_DIM = dim + N_BINS + N_BINS

print(f"Original embedding dim: {dim}, New dim: {NEW_DIM}", flush=True)


def extend_features(X_base, video_buckets, author_buckets):
    vid_col = (video_buckets + VIDEO_CTR_OFFSET).reshape(-1, 1).astype(np.int32)
    auth_col = (author_buckets + AUTHOR_CTR_OFFSET).reshape(-1, 1).astype(np.int32)
    return np.concatenate([X_base, vid_col, auth_col], axis=1)


X_tr = extend_features(X_tr_base, train_video_buckets, train_author_buckets)
X_va = extend_features(X_va_base, valid_video_buckets, valid_author_buckets)
X_te = extend_features(X_te_base, test_video_buckets, test_author_buckets)

print(f"Feature matrix shapes: train={X_tr.shape}, valid={X_va.shape}, test={X_te.shape}", flush=True)

# ── FM model with BPR training ────────────────────────────────────────────────
# We implement our own FM with Adam optimizer to support BPR loss
# FM score: b + sum_f(W[f]*x[f]) + sum_{f<g}(V[f]·V[g])
# Using the kernel trick: interaction = 0.5*(||sum_f V[f]||^2 - sum_f ||V[f]||^2)

class FMModel:
    """
    Factorization Machine with Adam optimizer.
    Supports both pointwise BCE (for warm-up) and BPR pairwise training.
    """
    def __init__(self, n_features, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.k = k
        self.lr = lr
        self.l2 = l2
        self.n_features = n_features

        # Parameters
        self.V = rng.normal(0, 0.01, (n_features, k)).astype(np.float32)
        self.W = np.zeros(n_features, dtype=np.float32)
        self.b = np.float32(0.0)

        # Adam state for V
        self.m_V = np.zeros_like(self.V)
        self.v_V = np.zeros_like(self.V)
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = 0.0
        self.v_b = 0.0

        self.t = 0  # Adam step count
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def _fm_score(self, X):
        """
        Compute FM scores for rows in X.
        X: int32 array (B, F) of feature indices
        Returns: (scores, sum_emb, sq_sum_emb) for gradient computation
        """
        B, F = X.shape
        # First order
        scores = np.full(B, self.b, dtype=np.float64)
        for f in range(F):
            idx = X[:, f]
            scores += self.W[idx]

        # Second order interaction via kernel trick
        # sum_emb[b] = sum_f V[X[b,f]] shape (B, k)
        sum_emb = np.zeros((B, self.k), dtype=np.float64)
        sq_sum_emb = np.zeros((B, self.k), dtype=np.float64)
        for f in range(F):
            idx = X[:, f]
            vf = self.V[idx].astype(np.float64)
            sum_emb += vf
            sq_sum_emb += vf ** 2

        # interaction = 0.5 * (||sum_emb||^2 - sum(sq_sum_emb))
        interaction = 0.5 * (np.sum(sum_emb ** 2, axis=1) - np.sum(sq_sum_emb, axis=1))
        scores += interaction
        return scores.astype(np.float32), sum_emb.astype(np.float32), sq_sum_emb.astype(np.float32)

    def _adam_update(self, param, grad, m, v):
        """One Adam step. Returns updated param, m, v."""
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        param = param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param, m, v

    def predict(self, X, bs=200_000):
        """Predict scores for all rows in X."""
        scores = []
        for start in range(0, len(X), bs):
            Xb = X[start:start + bs]
            s, _, _ = self._fm_score(Xb)
            scores.append(s)
        return np.concatenate(scores)

    def bpr_step(self, X_pos, X_neg):
        """
        One BPR update step on a batch of (positive, negative) pairs.
        X_pos, X_neg: int32 arrays (P, F) where P = number of pairs
        Returns: mean BPR loss
        """
        if len(X_pos) == 0:
            return 0.0

        self.t += 1
        P, F = X_pos.shape

        # Compute scores for positive and negative items
        s_pos, sum_pos, sq_pos = self._fm_score(X_pos)
        s_neg, sum_neg, sq_neg = self._fm_score(X_neg)

        # BPR: loss = -mean(log(sigmoid(s_pos - s_neg)))
        diff = s_pos.astype(np.float64) - s_neg.astype(np.float64)
        # sigmoid(diff)
        sig = 1.0 / (1.0 + np.exp(-np.clip(diff, -30, 30)))

        loss = -np.mean(np.log(sig + 1e-10))

        # Gradient: d_loss/d_diff = -(1 - sigmoid(diff)) per pair
        # d_loss/d_s_pos = -(1-sig)/P, d_loss/d_s_neg = +(1-sig)/P
        grad_diff = -(1.0 - sig) / P  # shape (P,)

        # Now backprop through FM for pos and neg
        # For each row i, d_loss/d_b += grad_diff[i]
        # d_loss/d_W[idx] += grad_diff[i] * 1  (first-order)
        # d_loss/d_V[idx,j] += grad_diff[i] * (sum_emb[i,j] - V[idx,j])  (interaction)

        # --- Bias gradient ---
        db = np.sum(grad_diff) - np.sum(grad_diff)  # cancel: pos and neg each contribute
        # Actually: grad for s_pos goes to pos rows, grad for s_neg goes to neg rows
        db_pos = np.sum(grad_diff)
        db_neg = -np.sum(grad_diff)
        db = db_pos + db_neg  # = 0, but let's keep it general

        # --- W gradient ---
        dW = np.zeros(self.n_features, dtype=np.float64)
        for f in range(F):
            idx_pos = X_pos[:, f]
            idx_neg = X_neg[:, f]
            np.add.at(dW, idx_pos, grad_diff)
            np.add.at(dW, idx_neg, -grad_diff)

        # L2 regularization on W
        dW += self.l2 * self.W

        # --- V gradient ---
        dV = np.zeros((self.n_features, self.k), dtype=np.float64)

        sum_pos_d = sum_pos.astype(np.float64)
        sum_neg_d = sum_neg.astype(np.float64)
        V_d = self.V.astype(np.float64)

        for f in range(F):
            idx_pos = X_pos[:, f]
            idx_neg = X_neg[:, f]

            # For positive rows: grad = grad_diff[:, None] * (sum_pos - V[idx_pos])
            v_pos_f = V_d[idx_pos]  # (P, k)
            g_pos = grad_diff[:, None] * (sum_pos_d - v_pos_f)  # (P, k)
            np.add.at(dV, idx_pos, g_pos)

            # For negative rows: grad = -grad_diff[:, None] * (sum_neg - V[idx_neg])
            v_neg_f = V_d[idx_neg]  # (P, k)
            g_neg = -grad_diff[:, None] * (sum_neg_d - v_neg_f)  # (P, k)
            np.add.at(dV, idx_neg, g_neg)

        # L2 regularization on V (only for indices that appear)
        touched = np.unique(np.concatenate([X_pos.ravel(), X_neg.ravel()]))
        dV[touched] += self.l2 * V_d[touched]

        # --- Adam updates ---
        self.b, self.m_b, self.v_b = self._adam_update(
            float(self.b), db, self.m_b, self.v_b
        )
        self.b = np.float32(self.b)

        self.W, self.m_W, self.v_W = self._adam_update(
            self.W.astype(np.float64), dW, self.m_W, self.v_W
        )
        self.W = self.W.astype(np.float32)

        self.V, self.m_V, self.v_V = self._adam_update(
            self.V.astype(np.float64), dV, self.m_V, self.v_V
        )
        self.V = self.V.astype(np.float32)

        return float(loss)


def build_bpr_pairs(X, y, users, rng):
    """
    For each user in batch that has at least one positive and one negative,
    sample one (positive, negative) pair.
    
    Returns: X_pos (P, F), X_neg (P, F) arrays
    """
    y_arr = np.asarray(y)
    users_arr = np.asarray(users)

    # Group by user
    user_to_pos = {}
    user_to_neg = {}

    for i in range(len(y_arr)):
        u = users_arr[i]
        if y_arr[i] == 1:
            if u not in user_to_pos:
                user_to_pos[u] = []
            user_to_pos[u].append(i)
        else:
            if u not in user_to_neg:
                user_to_neg[u] = []
            user_to_neg[u].append(i)

    pos_indices = []
    neg_indices = []

    for u in user_to_pos:
        if u in user_to_neg:
            # Sample one positive and one negative
            p_idx = rng.choice(user_to_pos[u])
            n_idx = rng.choice(user_to_neg[u])
            pos_indices.append(p_idx)
            neg_indices.append(n_idx)

    if len(pos_indices) == 0:
        return np.zeros((0, X.shape[1]), dtype=np.int32), np.zeros((0, X.shape[1]), dtype=np.int32)

    pos_indices = np.array(pos_indices)
    neg_indices = np.array(neg_indices)

    return X[pos_indices], X[neg_indices]


# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH = 8192
PATIENCE = 4
MAX_EPOCHS = 50
N_SEEDS = 5

# ── Helper: within-user fractional ranks ──────────────────────────────────────
def within_user_ranks(users, scores):
    user_arr = np.array(users)
    ranks = np.zeros(len(scores), dtype=np.float64)
    unique_users = np.unique(user_arr)
    for u in unique_users:
        mask = user_arr == u
        s = scores[mask]
        n = mask.sum()
        if n == 1:
            ranks[mask] = 0.5
            continue
        order = np.argsort(s)
        r = np.empty(n, dtype=np.float64)
        r[order] = np.arange(n, dtype=np.float64)
        r = r / (n - 1)
        ranks[mask] = r
    return ranks


def run_bpr_epoch(model, X, y, users, rng, batch_size=BATCH):
    """One full pass over data with BPR pairwise loss."""
    idx = rng.permutation(len(y))
    X_shuf = X[idx]
    y_shuf = y[idx]
    users_shuf = [users[i] for i in idx]

    losses = []
    n_pairs_total = 0

    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))
        Xb = X_shuf[start:end]
        yb = y_shuf[start:end]
        ub = users_shuf[start:end]

        X_pos, X_neg = build_bpr_pairs(Xb, yb, ub, rng)
        n_pairs = len(X_pos)
        n_pairs_total += n_pairs

        if n_pairs > 0:
            loss = model.bpr_step(X_pos, X_neg)
            losses.append(loss)

    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, n_pairs_total


def train_single_model_bpr(seed, X_tr, y_tr, u_tr, X_va, y_va, u_va, new_dim):
    """Train one FM model with BPR loss, return best model and epoch count."""
    rng = np.random.default_rng(seed)
    model = FMModel(new_dim, k=16, lr=0.001, l2=1e-6, seed=seed)

    best_val = -1.0
    best_weights = None
    patience_left = PATIENCE
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, n_pairs = run_bpr_epoch(model, X_tr, y_tr, u_tr, rng)

        val_scores = model.predict(X_va)
        val_res = evaluate(u_va, y_va, val_scores)
        val_primary = val_res['primary']

        print(f"  Seed {seed} Epoch {epoch:02d}  bpr_loss={tr_loss:.4f}  "
              f"n_pairs={n_pairs}  val_primary={val_primary:.4f}", flush=True)

        if val_primary > best_val + 1e-6:
            best_val = val_primary
            best_weights = (
                model.V.copy(),
                model.W.copy(),
                float(model.b),
                # Save Adam state too
                model.m_V.copy(), model.v_V.copy(),
                model.m_W.copy(), model.v_W.copy(),
                model.m_b, model.v_b,
                model.t
            )
            patience_left = PATIENCE
            best_epoch = epoch
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  Seed {seed} early stop at epoch {epoch}, best was {best_epoch}",
                      flush=True)
                break

    # Restore best weights
    if best_weights is not None:
        model.V[:] = best_weights[0]
        model.W[:] = best_weights[1]
        model.b = np.float32(best_weights[2])

    return model, best_epoch


if args.train_split == 'train':
    # ── Load random data for unbiased evaluation ──────────────────────────────
    rand_rows = load_random_valid(args.data_dir)
    X_rand_base, y_rand, u_rand, _ = encode_like_train(splits['train'], rand_rows)

    rand_video_rates = make_bucket_feature(rand_rows, video_ctr_map, video_global_rate, key_idx=2)
    rand_author_rates = make_bucket_feature(rand_rows, author_ctr_map, author_global_rate, key_idx=3)
    rand_video_buckets = compute_buckets_from_rates(train_video_rates, rand_video_rates, N_BINS)
    rand_author_buckets = compute_buckets_from_rates(train_author_rates, rand_author_rates, N_BINS)
    X_rand = extend_features(X_rand_base, rand_video_buckets, rand_author_buckets)

    # ── Train N_SEEDS models and collect scores ────────────────────────────────
    va_rank_sum = np.zeros(len(y_va), dtype=np.float64)
    te_rank_sum = np.zeros(len(y_te), dtype=np.float64)
    tr_rank_sum = np.zeros(len(y_tr), dtype=np.float64)
    rand_rank_sum = np.zeros(len(y_rand), dtype=np.float64)

    best_epochs = []

    for s in range(N_SEEDS):
        print(f"\n=== Training seed {s} (BPR) ===", flush=True)
        model, best_epoch = train_single_model_bpr(
            s, X_tr, y_tr, u_tr, X_va, y_va, u_va, NEW_DIM
        )
        best_epochs.append(best_epoch)

        # Compute scores for each split
        va_scores = model.predict(X_va)
        te_scores = model.predict(X_te)
        tr_scores = model.predict(X_tr)
        rand_scores = model.predict(X_rand)

        # Convert to within-user fractional ranks
        va_ranks = within_user_ranks(u_va, va_scores)
        te_ranks = within_user_ranks(u_te, te_scores)
        tr_ranks = within_user_ranks(u_tr, tr_scores)
        rand_ranks = within_user_ranks(u_rand, rand_scores)

        va_rank_sum += va_ranks
        te_rank_sum += te_ranks
        tr_rank_sum += tr_ranks
        rand_rank_sum += rand_ranks

    # Average ranks
    va_final = va_rank_sum / N_SEEDS
    te_final = te_rank_sum / N_SEEDS
    tr_final = tr_rank_sum / N_SEEDS
    rand_final = rand_rank_sum / N_SEEDS

    print(f"\nBest epochs across seeds: {best_epochs}", flush=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    tr_res = evaluate(u_tr, y_tr, tr_final)
    va_res = evaluate(u_va, y_va, va_final)

    # Unbiased evaluation using averaged ranks
    unbiased = unbiased_primary(
        args.data_dir, splits['train'],
        lambda rows: rand_final
    )

    print(f"TRAIN_PRIMARY={tr_res['primary']:.4f}")
    print(f"VAL_GAUC={va_res['GAUC']:.4f}")
    print(f"VAL_NDCG5={va_res['nDCG@5']:.4f}")
    print(f"VAL_PRIMARY={va_res['primary']:.4f}")
    print(f"UNBIASED_PRIMARY={unbiased:.4f}")

    # ── Write submissions ──────────────────────────────────────────────────────
    submit.write_submission(
        os.path.join(args.out_dir, 'submission_valid.csv'),
        splits['valid'], va_final
    )
    submit.write_submission(
        os.path.join(args.out_dir, 'submission_test.csv'),
        splits['test'], te_final
    )

else:
    # train+valid mode
    X_all = np.concatenate([X_tr, X_va], axis=0)
    y_all = np.concatenate([y_tr, y_va], axis=0)
    u_all = list(u_tr) + list(u_va)

    FIXED_EPOCHS = 10

    te_rank_sum = np.zeros(len(y_te), dtype=np.float64)

    for s in range(N_SEEDS):
        print(f"\n=== Training seed {s} (train+valid, BPR) ===", flush=True)
        rng = np.random.default_rng(s)
        model = FMModel(NEW_DIM, k=16, lr=0.001, l2=1e-6, seed=s)

        for epoch in range(1, FIXED_EPOCHS + 1):
            tr_loss, n_pairs = run_bpr_epoch(model, X_all, y_all, u_all, rng)
            print(f"  Seed {s} Epoch {epoch:02d}  bpr_loss={tr_loss:.4f}  n_pairs={n_pairs}", flush=True)

        te_scores = model.predict(X_te)
        te_ranks = within_user_ranks(u_te, te_scores)
        te_rank_sum += te_ranks

    te_final = te_rank_sum / N_SEEDS

    submit.write_submission(
        os.path.join(args.out_dir, 'submission_test.csv'),
        splits['test'], te_final
    )