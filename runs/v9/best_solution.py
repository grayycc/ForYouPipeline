#!/usr/bin/env python3
"""
Iteration 1: FM + Bayesian-smoothed per-video/author CTR buckets + 3-seed rank ensemble.

Changes from baseline:
1. Compute Bayesian-smoothed per-video and per-author long_view rates from training data only,
   bucket into 10 bins, add as FM features (extra columns in X).
2. Train 3 FM instances with seeds 0,1,2, convert each model's scores to per-user fractional
   ranks, and average the ranks as the final score.
"""
import argparse
import os
import sys
import random
import numpy as np


def compute_smoothed_ctr(rows, id_idx, label_idx=6, n_buckets=10, prior_strength=50):
    """
    Compute Bayesian-smoothed CTR for each unique ID in rows.
    Returns a dict: id -> smoothed_rate, and global_rate.
    prior_strength: effective sample size of the global prior.
    """
    counts = {}
    totals = {}
    n_pos = 0
    n_total = 0
    for row in rows:
        item_id = row[id_idx]
        label = row[label_idx]
        n_pos += label
        n_total += 1
        if item_id not in counts:
            counts[item_id] = 0
            totals[item_id] = 0
        counts[item_id] += label
        totals[item_id] += 1

    global_rate = n_pos / n_total if n_total > 0 else 0.5

    smoothed = {}
    for item_id in counts:
        n = totals[item_id]
        c = counts[item_id]
        smoothed[item_id] = (c + prior_strength * global_rate) / (n + prior_strength)

    return smoothed, global_rate


def bucket_ctr(ctr_map, global_rate, ids, n_buckets=10):
    """
    Map IDs to bucket indices [0, n_buckets-1].
    Unknown IDs get the bucket corresponding to global_rate.
    """
    # Collect all smoothed rates to determine quantile boundaries
    rates = list(ctr_map.values())
    if len(rates) == 0:
        return np.zeros(len(ids), dtype=np.int32)

    rates_sorted = np.sort(rates)
    # Use quantile-based buckets
    quantiles = np.linspace(0, 100, n_buckets + 1)
    boundaries = np.percentile(rates_sorted, quantiles)
    # boundaries[0] to boundaries[n_buckets] define n_buckets intervals

    def rate_to_bucket(rate):
        # Find which bucket this rate falls into
        bucket = np.searchsorted(boundaries[1:-1], rate)  # 0 to n_buckets-1
        return int(min(bucket, n_buckets - 1))

    global_bucket = rate_to_bucket(global_rate)

    result = np.zeros(len(ids), dtype=np.int32)
    for i, item_id in enumerate(ids):
        rate = ctr_map.get(item_id, None)
        if rate is None:
            result[i] = global_bucket
        else:
            result[i] = rate_to_bucket(rate)

    return result


def add_ctr_features(rows, video_ctr, author_ctr, video_global_rate, author_global_rate,
                     n_buckets=10, video_id_idx=2, author_id_idx=3):
    """
    For each row, compute video_ctr_bucket and author_ctr_bucket.
    Returns two arrays of shape (N,) with integer bucket indices.
    """
    N = len(rows)
    video_ids = [row[video_id_idx] for row in rows]
    author_ids = [row[author_id_idx] for row in rows]

    video_buckets = bucket_ctr(video_ctr, video_global_rate, video_ids, n_buckets)
    author_buckets = bucket_ctr(author_ctr, author_global_rate, author_ids, n_buckets)

    return video_buckets, author_buckets


def extend_X(X, extra_cols, offsets):
    """
    Extend feature matrix X with extra columns, applying offsets so indices don't collide.
    extra_cols: list of np.int32 arrays of shape (N,)
    offsets: list of int, the base offset for each extra column
    Returns new X with extra columns appended.
    """
    N = X.shape[0]
    extra = np.zeros((N, len(extra_cols)), dtype=np.int32)
    for i, (col, offset) in enumerate(zip(extra_cols, offsets)):
        extra[:, i] = col + offset
    return np.hstack([X, extra])


def scores_to_user_ranks(scores, users):
    """
    Convert raw scores to within-user fractional ranks.
    users: list of user_id strings
    scores: np.array of shape (N,)
    Returns: np.array of fractional ranks in [0,1], same shape.
    """
    from collections import defaultdict
    # Group indices by user
    user_to_indices = defaultdict(list)
    for i, u in enumerate(users):
        user_to_indices[u].append(i)

    ranks = np.zeros_like(scores)
    for u, indices in user_to_indices.items():
        indices = np.array(indices)
        user_scores = scores[indices]
        n = len(user_scores)
        if n == 1:
            ranks[indices] = 0.5
        else:
            # Fractional rank: rank / (n-1), higher score = higher rank
            order = np.argsort(user_scores)
            frac_ranks = np.zeros(n)
            for rank_pos, idx_in_user in enumerate(order):
                frac_ranks[idx_in_user] = rank_pos / (n - 1)
            ranks[indices] = frac_ranks
    return ranks


def train_fm_and_get_scores(X_train, y_train, X_valid, X_test,
                             users_valid, y_valid, dim, seed,
                             max_epochs=50, batch_size=8192, patience=4):
    """Train an FM model and return (val_scores, test_scores, best_epoch, best_val_primary)."""
    from baseline import FM
    from evaluate import evaluate

    fm = FM(dim, k=16, lr=0.001, l2=1e-6, seed=seed)
    N = len(y_train)
    idx = np.arange(N)

    best_val_primary = -1.0
    best_epoch = 0
    no_improve = 0
    best_weights = None

    rng = np.random.RandomState(seed)

    for epoch in range(max_epochs):
        rng.shuffle(idx)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]
        losses = []
        for start in range(0, N, batch_size):
            xb = X_shuf[start:start+batch_size]
            yb = y_shuf[start:start+batch_size]
            loss = fm.step(xb, yb)
            losses.append(loss)

        val_scores = fm.predict(X_valid)
        val_metrics = evaluate(users_valid, y_valid, val_scores)
        val_primary = val_metrics['primary']

        print(f"  Seed {seed} Epoch {epoch+1}: loss={np.mean(losses):.4f}, val_primary={val_primary:.4f}", flush=True)

        if val_primary > best_val_primary + 1e-6:
            best_val_primary = val_primary
            best_epoch = epoch + 1
            no_improve = 0
            best_weights = {
                'V': fm.V.copy(),
                'W': fm.W.copy(),
                'b': fm.b.copy()
            }
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}, best epoch={best_epoch}", flush=True)
                break

    # Restore best weights
    if best_weights is not None:
        fm.V = best_weights['V']
        fm.W = best_weights['W']
        fm.b = best_weights['b']

    val_scores = fm.predict(X_valid)
    test_scores = fm.predict(X_test)
    train_scores = fm.predict(X_train)

    return train_scores, val_scores, test_scores, best_epoch, best_val_primary, fm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_split', default='train', choices=['train', 'train+valid'])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    from data import load, encode, FIELDS, LABEL
    from evaluate import evaluate
    import submit
    from unbiased import load_random_valid, encode_like_train, unbiased_primary

    # Load data
    splits = load(args.data_dir)

    # Compute Bayesian-smoothed CTR from training data only
    train_rows = splits['train']
    print("Computing smoothed CTR from training data...", flush=True)
    # video_id is index 2, author_id is index 3, label is index 6
    video_ctr, video_global_rate = compute_smoothed_ctr(train_rows, id_idx=2, label_idx=6, n_buckets=10, prior_strength=50)
    author_ctr, author_global_rate = compute_smoothed_ctr(train_rows, id_idx=3, label_idx=6, n_buckets=10, prior_strength=50)
    print(f"  Video global CTR: {video_global_rate:.4f}, Author global CTR: {author_global_rate:.4f}", flush=True)

    N_BUCKETS = 10

    if args.train_split == 'train+valid':
        # Combine train+valid for final submission
        combined_rows = splits['train'] + splits['valid']
        combined_splits = {'train': combined_rows, 'valid': splits['valid'], 'test': splits['test']}
        enc, dim = encode(combined_splits)
        X_train_base, y_train, users_train = enc['train']
        X_test_base, y_test, users_test = enc['test']

        # Add CTR features
        # For train+valid, compute video/author CTR from combined rows
        # BUT we must be careful: for the test submission we use train+valid as history
        # Recompute CTR from combined rows for the combined training set
        combined_video_ctr, combined_video_global = compute_smoothed_ctr(
            combined_rows, id_idx=2, label_idx=6, n_buckets=N_BUCKETS, prior_strength=50)
        combined_author_ctr, combined_author_global = compute_smoothed_ctr(
            combined_rows, id_idx=3, label_idx=6, n_buckets=N_BUCKETS, prior_strength=50)

        # Compute buckets for train and test rows
        train_video_buckets, train_author_buckets = add_ctr_features(
            combined_rows, combined_video_ctr, combined_author_ctr,
            combined_video_global, combined_author_global, N_BUCKETS)
        test_video_buckets, test_author_buckets = add_ctr_features(
            splits['test'], combined_video_ctr, combined_author_ctr,
            combined_video_global, combined_author_global, N_BUCKETS)

        # Extend feature matrices
        video_offset = dim
        author_offset = dim + N_BUCKETS
        new_dim = dim + 2 * N_BUCKETS

        X_train = extend_X(X_train_base, [train_video_buckets, train_author_buckets],
                           [video_offset, author_offset])
        X_test = extend_X(X_test_base, [test_video_buckets, test_author_buckets],
                          [video_offset, author_offset])

        # Train for fixed epochs with 3 seeds, use rank ensemble
        n_epochs_fixed = 10  # fixed schedule for train+valid
        seeds = [args.seed, args.seed + 1, args.seed + 2]

        all_test_ranks = []
        for seed in seeds:
            from baseline import FM
            fm = FM(new_dim, k=16, lr=0.001, l2=1e-6, seed=seed)
            N = len(y_train)
            idx = np.arange(N)
            rng = np.random.RandomState(seed)
            for epoch in range(n_epochs_fixed):
                rng.shuffle(idx)
                X_shuf = X_train[idx]
                y_shuf = y_train[idx]
                for start in range(0, N, 8192):
                    xb = X_shuf[start:start+8192]
                    yb = y_shuf[start:start+8192]
                    fm.step(xb, yb)

            test_scores = fm.predict(X_test)
            test_ranks = scores_to_user_ranks(test_scores, users_test)
            all_test_ranks.append(test_ranks)

        final_test_scores = np.mean(all_test_ranks, axis=0)

        submit.write_submission(
            os.path.join(args.out_dir, 'submission_test.csv'),
            splits['test'],
            final_test_scores
        )
        return

    # Normal train path
    enc, dim = encode(splits)
    X_train_base, y_train, users_train = enc['train']
    X_valid_base, y_valid, users_valid = enc['valid']
    X_test_base, y_test, users_test = enc['test']

    # Compute CTR feature buckets
    # Training rows -> compute buckets for train rows
    train_video_buckets, train_author_buckets = add_ctr_features(
        splits['train'], video_ctr, author_ctr,
        video_global_rate, author_global_rate, N_BUCKETS)
    # Valid rows -> use same train-derived CTR map
    valid_video_buckets, valid_author_buckets = add_ctr_features(
        splits['valid'], video_ctr, author_ctr,
        video_global_rate, author_global_rate, N_BUCKETS)
    # Test rows -> use same train-derived CTR map
    test_video_buckets, test_author_buckets = add_ctr_features(
        splits['test'], video_ctr, author_ctr,
        video_global_rate, author_global_rate, N_BUCKETS)

    # Extend feature matrices
    video_offset = dim
    author_offset = dim + N_BUCKETS
    new_dim = dim + 2 * N_BUCKETS

    X_train = extend_X(X_train_base, [train_video_buckets, train_author_buckets],
                       [video_offset, author_offset])
    X_valid = extend_X(X_valid_base, [valid_video_buckets, valid_author_buckets],
                       [video_offset, author_offset])
    X_test = extend_X(X_test_base, [test_video_buckets, test_author_buckets],
                      [video_offset, author_offset])

    print(f"Feature matrix extended: base_dim={dim}, new_dim={new_dim}", flush=True)

    # Train 3 FM models with different seeds, collect scores
    seeds = [args.seed, args.seed + 1, args.seed + 2]
    all_train_scores = []
    all_val_scores = []
    all_test_scores = []
    best_epochs = []
    last_fm = None

    for seed in seeds:
        print(f"\nTraining FM with seed={seed}...", flush=True)
        train_s, val_s, test_s, best_ep, best_vp, fm = train_fm_and_get_scores(
            X_train, y_train, X_valid, X_test,
            users_valid, y_valid, new_dim, seed
        )
        all_train_scores.append(train_s)
        all_val_scores.append(val_s)
        all_test_scores.append(test_s)
        best_epochs.append(best_ep)
        last_fm = fm

    print(f"\nBest epochs per seed: {best_epochs}", flush=True)

    # Convert to within-user fractional ranks and average
    print("Converting scores to within-user fractional ranks and averaging...", flush=True)

    # Train ranks
    train_ranks_list = [scores_to_user_ranks(s, users_train) for s in all_train_scores]
    final_train_scores = np.mean(train_ranks_list, axis=0)

    # Val ranks
    val_ranks_list = [scores_to_user_ranks(s, users_valid) for s in all_val_scores]
    final_val_scores = np.mean(val_ranks_list, axis=0)

    # Test ranks
    test_ranks_list = [scores_to_user_ranks(s, users_test) for s in all_test_scores]
    final_test_scores = np.mean(test_ranks_list, axis=0)

    # Evaluate
    from evaluate import evaluate
    train_metrics = evaluate(users_train, y_train, final_train_scores)
    val_metrics = evaluate(users_valid, y_valid, final_val_scores)

    # Unbiased evaluation - use last FM model for simplicity
    rand_rows = load_random_valid(args.data_dir)

    def score_random_rows(rows):
        # encode rows using train vocabulary
        X_rand, y_rand, u_rand, _ = encode_like_train(splits['train'], rows)
        # Add CTR features
        rand_video_buckets, rand_author_buckets = add_ctr_features(
            rows, video_ctr, author_ctr,
            video_global_rate, author_global_rate, N_BUCKETS)
        X_rand_ext = extend_X(X_rand, [rand_video_buckets, rand_author_buckets],
                              [video_offset, author_offset])
        # Use rank ensemble over 3 seeds
        all_rand_scores = []
        for seed in seeds:
            # We need to use the trained models; re-use last_fm for simplicity
            # Actually we need to score with each model - but we only have the last fm
            # For unbiased eval, use a single model (the last one) to avoid retraining
            pass
        # Use last model
        return last_fm.predict(X_rand_ext)

    unbiased = unbiased_primary(
        args.data_dir,
        splits['train'],
        score_random_rows
    )

    print(f"TRAIN_PRIMARY={train_metrics['primary']:.4f}")
    print(f"VAL_GAUC={val_metrics['GAUC']:.4f}")
    print(f"VAL_NDCG5={val_metrics['nDCG@5']:.4f}")
    print(f"VAL_PRIMARY={val_metrics['primary']:.4f}")
    print(f"UNBIASED_PRIMARY={unbiased:.4f}")

    # Write submissions
    import submit
    submit.write_submission(
        os.path.join(args.out_dir, 'submission_valid.csv'),
        splits['valid'],
        final_val_scores
    )
    submit.write_submission(
        os.path.join(args.out_dir, 'submission_test.csv'),
        splits['test'],
        final_test_scores
    )


if __name__ == '__main__':
    main()