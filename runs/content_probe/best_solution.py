#!/usr/bin/env python3
"""
Reproduce the FM baseline. Trains a Factorization Machine over 5 categorical
fields with embedding dim k=16, Adam lr=0.001, batch size 8192, early stopping
on validation primary with patience 4.
"""

import argparse
import os
import sys
import time
import numpy as np

# ── project imports ────────────────────────────────────────────────────────────
from data import load, encode, FIELDS, LABEL
from baseline import FM
from evaluate import evaluate
import submit


def run(data_dir: str, out_dir: str, seed: int):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    # ── load & encode ──────────────────────────────────────────────────────────
    splits = load(data_dir)
    (enc, dim) = encode(splits)

    X_tr, y_tr, users_tr = enc['train']
    X_va, y_va, users_va = enc['valid']
    X_te, y_te, users_te = enc['test']

    # ── load random-exposure log for unbiased evaluation ──────────────────────
    import csv
    import importlib
    # We need to encode the random log the same way data.encode() does.
    # Read raw random log, join author_id, bucket duration with same edges.
    rand_log_path = os.path.join(data_dir, 'log_random_4_22_to_5_08_pure.csv')
    video_feat_path = os.path.join(data_dir, 'video_features_basic_pure.csv')

    # Build video -> author_id map
    video2author = {}
    with open(video_feat_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video2author[row['video_id']] = row.get('author_id', 'UNK') or 'UNK'

    # Compute duration quantile edges from training split
    train_durations = np.array([row[5] for row in splits['train']], dtype=np.float32)
    dur_edges = np.quantile(train_durations, np.linspace(0, 1, 11)[1:-1])  # 9 edges -> 10 buckets

    def dur_bucket(d):
        return int(np.searchsorted(dur_edges, d))

    # We need the same field encoders as data.encode() used.
    # Reconstruct by reading the encoders from encode()'s output.
    # Since we can't directly access the encoder dictionaries, we rebuild them
    # from the training data using the same logic as data.py.

    # Actually, let's use data.py's encode() output to get the offset table,
    # then we can map raw values ourselves.
    # The encode() function returns X as already-offset indices into a shared
    # embedding table. We need to know how to map raw categorical values.
    # The safest approach: re-read data.py's source to understand offset structure,
    # then replicate for the random log.

    # From data.py: FIELDS = ['user_id','video_id','author_id','tab','dur_bucket']
    # It builds a vocab per field, adds UNK, concatenates with cumulative offsets.
    # Let's rebuild the same vocab from training data + valid + test (for UNK handling)
    # Actually: UNK is per-field, new items in valid/test map to UNK.
    # We rebuild vocabs from training data only (same as data.encode does).

    FIELDS_LOCAL = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

    def build_vocab(rows):
        """Build vocab from training rows."""
        vocabs = [{} for _ in FIELDS_LOCAL]
        for row in rows:
            date, user_id, video_id, author_id, tab, duration_ms, label = row
            db = str(dur_bucket(duration_ms))
            vals = [user_id, video_id, author_id, tab, db]
            for i, v in enumerate(vals):
                if v not in vocabs[i]:
                    vocabs[i][v] = len(vocabs[i]) + 1  # 0 = UNK
        return vocabs

    train_rows = splits['train']
    vocabs = build_vocab(train_rows)

    field_sizes = [len(v) + 1 for v in vocabs]  # +1 for UNK at index 0
    offsets = np.cumsum([0] + field_sizes[:-1])

    def encode_rows(rows):
        """Encode rows into offset feature indices."""
        N = len(rows)
        X = np.zeros((N, len(FIELDS_LOCAL)), dtype=np.int32)
        y = np.zeros(N, dtype=np.float32)
        users = []
        for i, row in enumerate(rows):
            date, user_id, video_id, author_id, tab, duration_ms, label = row
            db = str(dur_bucket(duration_ms))
            vals = [user_id, video_id, author_id, tab, db]
            for j, v in enumerate(vals):
                idx = vocabs[j].get(v, 0)  # 0 = UNK
                X[i, j] = offsets[j] + idx
            y[i] = float(label)
            users.append(user_id)
        return X, y, users

    # Verify our encoding matches encode()'s output by checking dim
    total_dim = sum(field_sizes)
    # Note: encode() may differ slightly; we use our own encoding for consistency

    # Parse random log (validation window only: dates 20220422-20220428)
    rand_rows = []
    with open(rand_log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = int(row['date'])
            if date < 20220422 or date > 20220428:
                continue
            user_id = row['user_id']
            video_id = row['video_id']
            author_id = video2author.get(video_id, 'UNK') or 'UNK'
            tab = row.get('tab', 'UNK') or 'UNK'
            duration_ms = float(row.get('duration_ms', 0) or 0)
            label = int(row.get('long_view', 0) or 0)
            rand_rows.append((date, user_id, video_id, author_id, tab, duration_ms, label))

    # Now use our own encoder consistently for all splits
    X_tr2, y_tr2, users_tr2 = encode_rows(train_rows)
    X_va2, y_va2, users_va2 = encode_rows(splits['valid'])
    X_te2, y_te2, users_te2 = encode_rows(splits['test'])
    X_rn, y_rn, users_rn = encode_rows(rand_rows)

    # ── train FM ──────────────────────────────────────────────────────────────
    model = FM(total_dim, k=16, lr=0.001, l2=1e-6, seed=seed)

    batch_size = 8192
    patience = 4
    best_val = -1.0
    best_weights = None
    no_improve = 0
    N_tr = len(y_tr2)

    t0 = time.time()
    for epoch in range(1, 51):
        # Shuffle
        idx = rng.permutation(N_tr)
        X_sh = X_tr2[idx]
        y_sh = y_tr2[idx]

        losses = []
        for start in range(0, N_tr, batch_size):
            Xb = X_sh[start:start + batch_size]
            yb = y_sh[start:start + batch_size]
            loss = model.step(Xb, yb)
            losses.append(loss)

        # Evaluate on validation
        va_scores = model.predict(X_va2)
        va_res = evaluate(users_va2, y_va2, va_scores)
        val_primary = va_res['primary']

        print(f"Epoch {epoch:2d} | loss={np.mean(losses):.4f} | "
              f"val_primary={val_primary:.4f} | elapsed={time.time()-t0:.1f}s",
              flush=True)

        if val_primary > best_val + 1e-6:
            best_val = val_primary
            best_weights = (model.V.copy(), model.W.copy(), float(model.b))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch}", flush=True)
                break

    # Restore best weights
    model.V[:] = best_weights[0]
    model.W[:] = best_weights[1]
    model.b = best_weights[2]

    # ── final scores ──────────────────────────────────────────────────────────
    tr_scores = model.predict(X_tr2)
    tr_res = evaluate(users_tr2, y_tr2, tr_scores)

    va_scores = model.predict(X_va2)
    va_res = evaluate(users_va2, y_va2, va_scores)

    te_scores = model.predict(X_te2)

    rn_scores = model.predict(X_rn)
    rn_res = evaluate(users_rn, y_rn, rn_scores)

    print(f"TRAIN_PRIMARY={tr_res['primary']:.6f}")
    print(f"VAL_GAUC={va_res['GAUC']:.6f}")
    print(f"VAL_NDCG5={va_res['nDCG@5']:.6f}")
    print(f"VAL_PRIMARY={va_res['primary']:.6f}")
    print(f"UNBIASED_PRIMARY={rn_res['primary']:.6f}")

    # ── write submissions ─────────────────────────────────────────────────────
    submit.write_submission(
        os.path.join(out_dir, 'submission_valid.csv'),
        splits['valid'],
        va_scores
    )
    submit.write_submission(
        os.path.join(out_dir, 'submission_test.csv'),
        splits['test'],
        te_scores
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args.data_dir, args.out_dir, args.seed)