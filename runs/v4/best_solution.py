#!/usr/bin/env python3
"""
Iteration 0: Reproduce the FM baseline exactly.
"""

import argparse
import os
import sys
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_split', type=str, default='train',
                        choices=['train', 'train+valid'])
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Import project utilities
    from data import load, encode, FIELDS
    from evaluate import evaluate
    import submit

    # Import FM model
    from baseline import FM

    # Load data
    print("Loading data...", flush=True)
    splits = load(args.data_dir)

    # Encode features
    print("Encoding features...", flush=True)
    enc, dim = encode(splits)

    X_train, y_train, users_train = enc['train']
    X_valid, y_valid, users_valid = enc['valid']
    X_test,  y_test,  users_test  = enc['test']

    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}", flush=True)
    print(f"Embedding table size: {dim}", flush=True)

    # Initialize model
    model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=args.seed)

    # Training configuration
    batch_size = 8192
    patience = 4
    max_epochs = 50

    best_val_primary = -1.0
    best_epoch = 0
    best_weights = None
    no_improve = 0

    if args.train_split == 'train+valid':
        # Combine train + valid for final refit
        X_fit = np.concatenate([X_train, X_valid], axis=0)
        y_fit = np.concatenate([y_train, y_valid], axis=0)
        users_fit = users_train + users_valid
        # Use epoch count from a prior run or fixed schedule
        # We'll train for best_epoch_count epochs (use 10 as reasonable default)
        # Actually train with same loop but no early stopping, fixed 15 epochs
        n_epochs_refit = 15
        n = len(y_fit)
        for epoch in range(n_epochs_refit):
            idx = np.random.permutation(n)
            losses = []
            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                Xb = X_fit[batch_idx]
                yb = y_fit[batch_idx]
                loss = model.step(Xb, yb)
                losses.append(loss)
            mean_loss = np.mean(losses)
            print(f"Epoch {epoch+1}/{n_epochs_refit}, loss={mean_loss:.4f}", flush=True)
    else:
        # Normal training with early stopping on validation
        n = len(y_train)
        for epoch in range(max_epochs):
            idx = np.random.permutation(n)
            losses = []
            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                Xb = X_train[batch_idx]
                yb = y_train[batch_idx]
                loss = model.step(Xb, yb)
                losses.append(loss)
            mean_loss = np.mean(losses)

            # Evaluate on validation
            val_scores = model.predict(X_valid)
            val_metrics = evaluate(users_valid, y_valid, val_scores)
            val_primary = val_metrics['primary']

            print(f"Epoch {epoch+1}: loss={mean_loss:.4f}, val_primary={val_primary:.4f}", flush=True)

            if val_primary > best_val_primary + 1e-6:
                best_val_primary = val_primary
                best_epoch = epoch + 1
                # Save weights
                best_weights = {
                    'V': model.V.copy(),
                    'W': model.W.copy(),
                    'b': float(model.b)
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}, best was epoch {best_epoch}", flush=True)
                    break

        # Restore best weights
        if best_weights is not None:
            model.V[:] = best_weights['V']
            model.W[:] = best_weights['W']
            model.b = best_weights['b']

    # Final evaluation
    train_scores = model.predict(X_train)
    train_metrics = evaluate(users_train, y_train, train_scores)

    val_scores = model.predict(X_valid)
    val_metrics = evaluate(users_valid, y_valid, val_scores)

    test_scores = model.predict(X_test)

    # Compute unbiased primary on random-exposure log
    print("Computing unbiased primary...", flush=True)
    unbiased_primary = compute_unbiased_primary(args.data_dir, model, splits, enc)

    print(f"TRAIN_PRIMARY={train_metrics['primary']:.4f}")
    print(f"VAL_GAUC={val_metrics['GAUC']:.4f}")
    print(f"VAL_NDCG5={val_metrics['nDCG@5']:.4f}")
    print(f"VAL_PRIMARY={val_metrics['primary']:.4f}")
    print(f"UNBIASED_PRIMARY={unbiased_primary:.4f}")

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
    print("Submissions written.", flush=True)


def compute_unbiased_primary(data_dir, model, splits, enc):
    """Compute primary metric on random-exposure log (valid date window only)."""
    import pandas as pd
    import numpy as np
    from evaluate import evaluate

    # Load random exposure log
    rand_path = os.path.join(data_dir, 'log_random_4_22_to_5_08_pure.csv')
    rand_df = pd.read_csv(rand_path)

    # Filter to valid date window: 20220422-20220428
    rand_df = rand_df[(rand_df['date'] >= 20220422) & (rand_df['date'] <= 20220428)]

    if len(rand_df) == 0:
        return 0.0

    # Load video features for author_id join
    video_feat_path = os.path.join(data_dir, 'video_features_basic_pure.csv')
    vf = pd.read_csv(video_feat_path)
    vf['video_id'] = vf['video_id'].astype(str)
    vid_to_author = dict(zip(vf['video_id'], vf['author_id'].astype(str)))

    # Get dur_bucket edges from training data
    # We need to replicate the same quantile bucketing used in encode()
    # Extract duration_ms from training split rows
    train_rows = splits['train']
    train_durations = np.array([row[5] for row in train_rows], dtype=np.float32)
    bucket_edges = np.quantile(train_durations, np.linspace(0, 1, 11)[1:-1])  # 9 edges -> 10 buckets

    # Build the encoder mapping used in enc
    # We need to get the field offsets from the encoded data
    # Let's reconstruct the vocabulary from enc
    # Actually, let's rebuild manually using the same logic as data.encode()
    # We'll use the existing enc to get field offsets, then map our random rows

    # Get the encoder from data module
    from data import load, encode, FIELDS

    # Re-use the same encoder by calling encode on a combined dict
    # But we need the same vocabulary... let's use a different approach:
    # We'll use encode() internals by accessing the vocabulary

    # The cleanest approach: use the encode function with the random rows appended
    # but that changes the vocabulary. Instead, let's manually replicate encoding.

    # From data.py, FIELDS = ['user_id','video_id','author_id','tab','dur_bucket']
    # encode() builds vocab from train split only, then applies to all splits
    # We need the same vocab mapping

    # Let's extract vocab from the encoded arrays by reverse-engineering
    # Actually, the simplest way: call encode() again but include rand as a split
    # The vocab is built from train only, unseen -> UNK

    # Build rand rows in the same tuple format as load() output
    # Format: (date, user_id, video_id, author_id, tab, duration_ms, long_view)
    rand_df['video_id'] = rand_df['video_id'].astype(str)
    rand_df['user_id'] = rand_df['user_id'].astype(str)
    rand_df['author_id'] = rand_df['video_id'].map(vid_to_author).fillna('UNK')

    # Get tab column name - check what columns exist
    tab_col = 'tab' if 'tab' in rand_df.columns else None
    if tab_col is None:
        # Try to find it
        for col in rand_df.columns:
            if 'tab' in col.lower():
                tab_col = col
                break
    if tab_col is None:
        rand_df['tab'] = 'UNK'
        tab_col = 'tab'

    # duration_ms column
    dur_col = 'duration_ms' if 'duration_ms' in rand_df.columns else None
    if dur_col is None:
        # Try video features
        vf2 = pd.read_csv(os.path.join(data_dir, 'video_features_basic_pure.csv'))
        vf2['video_id'] = vf2['video_id'].astype(str)
        # look for duration column
        dur_cols = [c for c in vf2.columns if 'dur' in c.lower()]
        if dur_cols:
            vf2 = vf2[['video_id', dur_cols[0]]].rename(columns={dur_cols[0]: 'duration_ms'})
            rand_df = rand_df.merge(vf2, on='video_id', how='left')
            rand_df['duration_ms'] = rand_df['duration_ms'].fillna(0.0)
        else:
            rand_df['duration_ms'] = 0.0
        dur_col = 'duration_ms'

    # Build rand_rows as tuples
    rand_rows = []
    for _, row in rand_df.iterrows():
        rand_rows.append((
            int(row['date']),
            str(row['user_id']),
            str(row['video_id']),
            str(row['author_id']),
            str(row[tab_col]),
            float(row[dur_col]),
            int(row['long_view'])
        ))

    # Now encode rand_rows using the same vocabulary
    # We'll call encode() with rand included
    splits_with_rand = {
        'train': splits['train'],
        'valid': splits['valid'],
        'test': splits['test'],
        'rand': rand_rows
    }
    enc2, dim2 = encode(splits_with_rand)
    X_rand, y_rand, users_rand = enc2['rand']

    # Re-encode train/valid with same vocab to ensure consistency
    # Actually enc2 should have same vocab as enc since vocab built from train only
    # Predict using model
    rand_scores = model.predict(X_rand)
    rand_metrics = evaluate(users_rand, y_rand, rand_scores)
    return rand_metrics['primary']


if __name__ == '__main__':
    main()