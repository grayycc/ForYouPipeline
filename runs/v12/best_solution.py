#!/usr/bin/env python3
"""
Iteration 2: 10-seed ensemble with rank averaging + Bayesian-smoothed per-video CTR feature
+ Bayesian-smoothed per-author long_view rate feature (bucketed into 10 quantile bins).

Changes from iteration 1:
- Add a sixth feature field: Bayesian-smoothed per-author long_view rate computed from
  train rows only, smoothed with prior_weight=20 toward the global rate, bucketed into 10
  quantile bins. The bucket index is added as a new offset field in the FM embedding table.
"""

import argparse
import os
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_split', type=str, default='train',
                        choices=['train', 'train+valid'])
    return parser.parse_args()


def build_ctr_buckets(train_rows, n_buckets=10, prior_weight=20):
    """
    Compute Bayesian-smoothed per-video long_view rate from train_rows only.
    Returns (video_smoothed dict, global_rate, bin_edges array).
    """
    video_pos = {}
    video_cnt = {}
    total_pos = 0
    total_cnt = 0
    for row in train_rows:
        vid = row[2]
        label = row[6]
        video_pos[vid] = video_pos.get(vid, 0) + label
        video_cnt[vid] = video_cnt.get(vid, 0) + 1
        total_pos += label
        total_cnt += 1

    global_rate = total_pos / total_cnt if total_cnt > 0 else 0.5

    video_smoothed = {}
    for vid in video_cnt:
        n = video_cnt[vid]
        pos = video_pos[vid]
        smoothed = (pos + prior_weight * global_rate) / (n + prior_weight)
        video_smoothed[vid] = smoothed

    rates = list(video_smoothed.values())
    quantiles = np.linspace(0, 100, n_buckets + 1)
    bin_edges = np.percentile(rates, quantiles)
    bin_edges = np.unique(bin_edges)

    return video_smoothed, global_rate, bin_edges


def build_author_ctr_buckets(train_rows, n_buckets=10, prior_weight=20):
    """
    Compute Bayesian-smoothed per-author long_view rate from train_rows only.
    Returns (author_smoothed dict, global_rate, bin_edges array).
    """
    author_pos = {}
    author_cnt = {}
    total_pos = 0
    total_cnt = 0
    for row in train_rows:
        author_id = row[3]
        label = row[6]
        author_pos[author_id] = author_pos.get(author_id, 0) + label
        author_cnt[author_id] = author_cnt.get(author_id, 0) + 1
        total_pos += label
        total_cnt += 1

    global_rate = total_pos / total_cnt if total_cnt > 0 else 0.5

    author_smoothed = {}
    for aid in author_cnt:
        n = author_cnt[aid]
        pos = author_pos[aid]
        smoothed = (pos + prior_weight * global_rate) / (n + prior_weight)
        author_smoothed[aid] = smoothed

    rates = list(author_smoothed.values())
    quantiles = np.linspace(0, 100, n_buckets + 1)
    bin_edges = np.percentile(rates, quantiles)
    bin_edges = np.unique(bin_edges)

    return author_smoothed, global_rate, bin_edges


def get_ctr_bucket(video_id, video_smoothed, global_rate, bin_edges):
    rate = video_smoothed.get(video_id, global_rate)
    bucket = int(np.searchsorted(bin_edges, rate, side='right')) - 1
    bucket = max(0, min(bucket, len(bin_edges) - 2))
    return bucket


def get_author_ctr_bucket(author_id, author_smoothed, global_rate, bin_edges):
    rate = author_smoothed.get(author_id, global_rate)
    bucket = int(np.searchsorted(bin_edges, rate, side='right')) - 1
    bucket = max(0, min(bucket, len(bin_edges) - 2))
    return bucket


def encode_with_ctr(train_rows, rows_dict,
                    video_smoothed, video_global_rate, video_bin_edges,
                    author_smoothed, author_global_rate, author_bin_edges):
    """
    Custom encoder: user_id, video_id, author_id, tab, dur_bucket,
                    video_ctr_bucket, author_ctr_bucket.
    Vocab and bin edges derived from train_rows only.
    Returns enc dict {split: (X, y, users)} and dim.
    """
    field_vocabs = {fn: {} for fn in ['user_id', 'video_id', 'author_id', 'tab']}

    dur_values = np.array([row[5] for row in train_rows], dtype=np.float64)
    dur_edges = np.unique(np.percentile(dur_values, np.linspace(0, 100, 11)))

    for row in train_rows:
        for fn, val in [('user_id', row[1]), ('video_id', row[2]),
                        ('author_id', row[3]), ('tab', row[4])]:
            if val not in field_vocabs[fn]:
                field_vocabs[fn][val] = len(field_vocabs[fn])

    unk = {fn: len(field_vocabs[fn]) for fn in field_vocabs}

    field_sizes = {
        'user_id':           len(field_vocabs['user_id']) + 1,
        'video_id':          len(field_vocabs['video_id']) + 1,
        'author_id':         len(field_vocabs['author_id']) + 1,
        'tab':               len(field_vocabs['tab']) + 1,
        'dur_bucket':        len(dur_edges),
        'video_ctr_bucket':  len(video_bin_edges),
        'author_ctr_bucket': len(author_bin_edges),
    }

    field_order = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket',
                   'video_ctr_bucket', 'author_ctr_bucket']
    offsets = {}
    current = 0
    for fn in field_order:
        offsets[fn] = current
        current += field_sizes[fn]
    dim = current

    def row_to_features(row):
        user_id = row[1]
        video_id = row[2]
        author_id = row[3]
        tab = row[4]
        dur = row[5]

        u_idx = field_vocabs['user_id'].get(user_id, unk['user_id'])
        v_idx = field_vocabs['video_id'].get(video_id, unk['video_id'])
        a_idx = field_vocabs['author_id'].get(author_id, unk['author_id'])
        t_idx = field_vocabs['tab'].get(tab, unk['tab'])

        d_bucket = int(np.searchsorted(dur_edges, dur, side='right')) - 1
        d_bucket = max(0, min(d_bucket, len(dur_edges) - 2))

        c_bucket = get_ctr_bucket(video_id, video_smoothed, video_global_rate, video_bin_edges)
        ac_bucket = get_author_ctr_bucket(author_id, author_smoothed, author_global_rate, author_bin_edges)

        return [
            u_idx + offsets['user_id'],
            v_idx + offsets['video_id'],
            a_idx + offsets['author_id'],
            t_idx + offsets['tab'],
            d_bucket + offsets['dur_bucket'],
            c_bucket + offsets['video_ctr_bucket'],
            ac_bucket + offsets['author_ctr_bucket'],
        ]

    enc = {}
    for split_name, rows in rows_dict.items():
        X = np.array([row_to_features(r) for r in rows], dtype=np.int32)
        y = np.array([r[6] for r in rows], dtype=np.float32)
        users = [r[1] for r in rows]
        enc[split_name] = (X, y, users)

    return enc, dim


def within_user_rank(scores, users):
    """Convert scores to within-user fractional ranks in [0, 1]."""
    users_arr = np.array(users)
    ranks = np.zeros(len(scores), dtype=np.float64)
    unique_users = np.unique(users_arr)
    for u in unique_users:
        mask = users_arr == u
        s = scores[mask]
        n = len(s)
        if n == 1:
            ranks[mask] = 0.5
        else:
            order = np.argsort(np.argsort(s))
            ranks[mask] = order / (n - 1)
    return ranks


def train_single_model(X_train, y_train, X_valid, y_valid, users_valid,
                       seed, dim, evaluate_fn,
                       n_epochs=50, batch_size=8192, patience=4):
    """Train one FM model with early stopping; return (model, best_epoch, best_val)."""
    from baseline import FM

    model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=seed)
    rng = np.random.RandomState(seed)
    N = X_train.shape[0]

    best_val = -np.inf
    best_epoch = 0
    patience_count = 0
    best_weights = None

    for epoch in range(n_epochs):
        idx = rng.permutation(N)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]
        for start in range(0, N, batch_size):
            xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]
            model.step(xb, yb)

        val_scores = model.predict(X_valid)
        val_metrics = evaluate_fn(users_valid, y_valid, val_scores)
        vp = val_metrics['primary']

        if vp > best_val + 1e-6:
            best_val = vp
            best_epoch = epoch + 1
            patience_count = 0
            best_weights = (model.V.copy(), model.W.copy(), float(model.b))
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_weights is not None:
        model.V[:] = best_weights[0]
        model.W[:] = best_weights[1]
        model.b = best_weights[2]

    return model, best_epoch, best_val


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    from data import load
    from evaluate import evaluate
    from submit import write_submission
    from unbiased import load_random_valid, unbiased_primary

    data_dir = args.data_dir
    splits = load(data_dir)

    N_ENSEMBLE = 10
    ENSEMBLE_SEEDS = [args.seed + i for i in range(N_ENSEMBLE)]

    if args.train_split == 'train+valid':
        train_rows = splits['train'] + splits['valid']
        test_rows = splits['test']

        video_smoothed, video_global_rate, video_bin_edges = build_ctr_buckets(
            train_rows, n_buckets=10, prior_weight=20
        )
        author_smoothed, author_global_rate, author_bin_edges = build_author_ctr_buckets(
            train_rows, n_buckets=10, prior_weight=20
        )
        enc, dim = encode_with_ctr(
            train_rows,
            {'train': train_rows, 'test': test_rows},
            video_smoothed, video_global_rate, video_bin_edges,
            author_smoothed, author_global_rate, author_bin_edges
        )
        X_train, y_train, users_train = enc['train']
        X_test, y_test, users_test = enc['test']

        n_fixed_epochs = 8
        batch_size = 8192
        N = X_train.shape[0]

        all_test_ranks = []
        for seed in ENSEMBLE_SEEDS:
            from baseline import FM
            model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=seed)
            rng = np.random.RandomState(seed)
            for epoch in range(n_fixed_epochs):
                idx = rng.permutation(N)
                X_shuf = X_train[idx]
                y_shuf = y_train[idx]
                for start in range(0, N, batch_size):
                    xb = X_shuf[start:start + batch_size]
                    yb = y_shuf[start:start + batch_size]
                    model.step(xb, yb)
            test_scores = model.predict(X_test)
            test_ranks = within_user_rank(test_scores, users_test)
            all_test_ranks.append(test_ranks)

        final_test_scores = np.mean(all_test_ranks, axis=0)
        write_submission(
            os.path.join(args.out_dir, 'submission_test.csv'),
            test_rows, final_test_scores
        )
        return

    # --- Standard training path ---
    train_rows = splits['train']
    valid_rows = splits['valid']
    test_rows = splits['test']

    # Compute CTR features from train only
    video_smoothed, video_global_rate, video_bin_edges = build_ctr_buckets(
        train_rows, n_buckets=10, prior_weight=20
    )
    author_smoothed, author_global_rate, author_bin_edges = build_author_ctr_buckets(
        train_rows, n_buckets=10, prior_weight=20
    )
    print(f"Video CTR feature: {len(video_smoothed)} videos, global_rate={video_global_rate:.4f}, "
          f"n_bin_edges={len(video_bin_edges)}", flush=True)
    print(f"Author CTR feature: {len(author_smoothed)} authors, global_rate={author_global_rate:.4f}, "
          f"n_bin_edges={len(author_bin_edges)}", flush=True)

    # Load random rows for unbiased evaluation
    rand_rows = load_random_valid(data_dir)

    enc, dim = encode_with_ctr(
        train_rows,
        {'train': train_rows, 'valid': valid_rows, 'test': test_rows, 'rand': rand_rows},
        video_smoothed, video_global_rate, video_bin_edges,
        author_smoothed, author_global_rate, author_bin_edges
    )
    X_train, y_train, users_train = enc['train']
    X_valid, y_valid, users_valid = enc['valid']
    X_test, y_test, users_test = enc['test']
    X_rand, y_rand, users_rand = enc['rand']

    print(f"Embedding dim: {dim}", flush=True)
    print(f"Train: {X_train.shape[0]}, Valid: {X_valid.shape[0]}, "
          f"Test: {X_test.shape[0]}, Rand: {X_rand.shape[0]}", flush=True)

    # Train ensemble, collecting all scores in one pass
    all_train_ranks = []
    all_val_ranks = []
    all_test_ranks = []
    all_rand_scores = []
    best_epochs = []

    for i, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"Training model {i + 1}/{N_ENSEMBLE} (seed={seed})...", flush=True)
        model, best_epoch, best_val = train_single_model(
            X_train, y_train, X_valid, y_valid, users_valid,
            seed=seed, dim=dim, evaluate_fn=evaluate,
            n_epochs=50, batch_size=8192, patience=4
        )
        best_epochs.append(best_epoch)
        print(f"  Best epoch: {best_epoch}, val_primary: {best_val:.4f}", flush=True)

        train_scores = model.predict(X_train)
        val_scores = model.predict(X_valid)
        test_scores = model.predict(X_test)
        rand_scores = model.predict(X_rand)

        all_train_ranks.append(within_user_rank(train_scores, users_train))
        all_val_ranks.append(within_user_rank(val_scores, users_valid))
        all_test_ranks.append(within_user_rank(test_scores, users_test))
        all_rand_scores.append(rand_scores)

    # Average ranks for train/valid/test; average raw scores for rand
    final_train_scores = np.mean(all_train_ranks, axis=0)
    final_val_scores = np.mean(all_val_ranks, axis=0)
    final_test_scores = np.mean(all_test_ranks, axis=0)
    final_rand_scores = np.mean(all_rand_scores, axis=0)

    print(f"Best epochs per model: {best_epochs}", flush=True)

    # Evaluate
    train_metrics = evaluate(users_train, y_train, final_train_scores)
    val_metrics = evaluate(users_valid, y_valid, final_val_scores)

    # Unbiased evaluation using pre-computed scores
    unbiased = unbiased_primary(
        data_dir, splits['train'],
        lambda rows_ignored: final_rand_scores
    )

    print(f"TRAIN_PRIMARY={train_metrics['primary']:.6f}")
    print(f"VAL_GAUC={val_metrics['GAUC']:.6f}")
    print(f"VAL_NDCG5={val_metrics['nDCG@5']:.6f}")
    print(f"VAL_PRIMARY={val_metrics['primary']:.6f}")
    print(f"UNBIASED_PRIMARY={unbiased:.6f}")

    write_submission(
        os.path.join(args.out_dir, 'submission_valid.csv'),
        valid_rows, final_val_scores
    )
    write_submission(
        os.path.join(args.out_dir, 'submission_test.csv'),
        test_rows, final_test_scores
    )


if __name__ == '__main__':
    main()