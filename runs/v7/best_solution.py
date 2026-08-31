#!/usr/bin/env python3
"""
Iteration 0: Reproduce the FM baseline.
"""
import argparse
import os
import sys
import random
import numpy as np

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data")
parser.add_argument("--out_dir",  default="out")
parser.add_argument("--seed",     type=int, default=0)
parser.add_argument("--train_split", default="train",
                    choices=["train", "train+valid"])
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)

# ── Imports ───────────────────────────────────────────────────────────────────
from data import load, encode
from evaluate import evaluate
from submit import write_submission
from baseline import FM
from unbiased import load_random_valid, encode_like_train, unbiased_primary

# ── Load data ─────────────────────────────────────────────────────────────────
splits = load(args.data_dir)

# ── Encode ────────────────────────────────────────────────────────────────────
enc, dim = encode(splits)
(X_tr, y_tr, u_tr) = enc["train"]
(X_va, y_va, u_va) = enc["valid"]
(X_te, y_te, u_te) = enc["test"]

# ── If train+valid mode: retrain on combined split, write test submission only ──
if args.train_split == "train+valid":
    # Use the epoch count from a typical run (we'll fix to 20 epochs max)
    X_tv = np.concatenate([X_tr, X_va], axis=0)
    y_tv = np.concatenate([y_tr, y_va], axis=0)

    model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=args.seed)
    batch_size = 8192
    N = len(X_tv)
    best_epochs = 15  # conservative fixed epoch count for refit

    for epoch in range(best_epochs):
        idx = np.random.permutation(N)
        X_shuf = X_tv[idx]
        y_shuf = y_tv[idx]
        losses = []
        for start in range(0, N, batch_size):
            xb = X_shuf[start:start+batch_size]
            yb = y_shuf[start:start+batch_size]
            loss = model.step(xb, yb)
            losses.append(loss)

    scores_te = model.predict(X_te)
    write_submission(
        os.path.join(args.out_dir, "submission_test.csv"),
        splits["test"], scores_te
    )
    sys.exit(0)

# ── Standard training on train split ─────────────────────────────────────────
model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=args.seed)

batch_size = 8192
N_tr = len(X_tr)
patience = 4
best_val_primary = -1.0
best_weights = None
no_improve = 0
max_epochs = 50

for epoch in range(max_epochs):
    # Shuffle training data
    idx = np.random.permutation(N_tr)
    X_shuf = X_tr[idx]
    y_shuf = y_tr[idx]

    losses = []
    for start in range(0, N_tr, batch_size):
        xb = X_shuf[start:start+batch_size]
        yb = y_shuf[start:start+batch_size]
        loss = model.step(xb, yb)
        losses.append(loss)

    mean_loss = float(np.mean(losses))

    # Evaluate on validation
    scores_va = model.predict(X_va)
    val_metrics = evaluate(u_va, y_va, scores_va)
    val_primary = val_metrics["primary"]

    print(f"Epoch {epoch+1:3d}  loss={mean_loss:.4f}  "
          f"val_primary={val_primary:.4f}  "
          f"GAUC={val_metrics['GAUC']:.4f}  "
          f"nDCG@5={val_metrics['nDCG@5']:.4f}",
          flush=True)

    if val_primary > best_val_primary + 1e-6:
        best_val_primary = val_primary
        # Save weights
        best_weights = (model.V.copy(), model.W.copy(), float(model.b))
        best_epoch = epoch + 1
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stop at epoch {epoch+1}, best epoch={best_epoch}", flush=True)
            break

# Restore best weights
model.V[:] = best_weights[0]
model.W[:] = best_weights[1]
model.b    = best_weights[2]

# ── Final evaluation ─────────────────────────────────────────────────────────
scores_tr = model.predict(X_tr)
train_metrics = evaluate(u_tr, y_tr, scores_tr)

scores_va = model.predict(X_va)
val_metrics = evaluate(u_va, y_va, scores_va)

scores_te = model.predict(X_te)

# ── Unbiased evaluation ───────────────────────────────────────────────────────
rand_rows = load_random_valid(args.data_dir)
X_rand, y_rand, u_rand, _ = encode_like_train(splits["train"], rand_rows)
unbiased = unbiased_primary(
    args.data_dir, splits["train"],
    lambda rows: model.predict(X_rand)
)

# ── Print required metrics ────────────────────────────────────────────────────
print(f"TRAIN_PRIMARY={train_metrics['primary']:.6f}")
print(f"VAL_GAUC={val_metrics['GAUC']:.6f}")
print(f"VAL_NDCG5={val_metrics['nDCG@5']:.6f}")
print(f"VAL_PRIMARY={val_metrics['primary']:.6f}")
print(f"UNBIASED_PRIMARY={unbiased:.6f}")

# ── Write submissions ─────────────────────────────────────────────────────────
write_submission(
    os.path.join(args.out_dir, "submission_valid.csv"),
    splits["valid"], scores_va
)
write_submission(
    os.path.join(args.out_dir, "submission_test.csv"),
    splits["test"], scores_te
)