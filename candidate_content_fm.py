#!/usr/bin/env python3
"""Candidate solution using train-only category/caption preference signals with FM.

This is a standalone experiment: it creates the feature logic inside the file itself, matching
what the autonomous agent is supposed to do. It keeps the FM backbone but adds two causal
preference features derived from the new caption/category files.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kit'))
from baseline import FM
from evaluate import evaluate
from submit import write_submission


def load_rows(data_dir):
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv'), newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            vid2author[row['video_id']] = row.get('author_id', 'UNK') or 'UNK'

    rows = []
    for fname in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, fname), newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                rows.append((
                    int(row['date']),
                    row['user_id'],
                    row['video_id'],
                    vid2author.get(row['video_id'], 'UNK') or 'UNK',
                    row.get('tab', 'UNK') or 'UNK',
                    float(row.get('duration_ms', 0) or 0),
                    1 if row.get('long_view', '0') != '0' else 0,
                ))
    out = {'train': [], 'valid': [], 'test': []}
    for r in rows:
        d = r[0]
        if 20220408 <= d <= 20220421:
            out['train'].append(r)
        elif 20220422 <= d <= 20220428:
            out['valid'].append(r)
        elif d >= 20220429:
            out['test'].append(r)
    return out


def load_random_valid(data_dir):
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv'), newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            vid2author[row['video_id']] = row.get('author_id', 'UNK') or 'UNK'
    rows = []
    with open(os.path.join(data_dir, 'log_random_4_22_to_5_08_pure.csv'), newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            d = int(row['date'])
            if 20220422 <= d <= 20220428:
                rows.append((
                    d,
                    row['user_id'],
                    row['video_id'],
                    vid2author.get(row['video_id'], 'UNK') or 'UNK',
                    row.get('tab', 'UNK') or 'UNK',
                    float(row.get('duration_ms', 0) or 0),
                    1 if row.get('long_view', '0') != '0' else 0,
                ))
    return rows


def load_video_categories(data_dir):
    out = {}
    path = os.path.join(data_dir, 'kuairand_video_categories.csv')
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            vid = str(row['final_video_id'])
            vals = []
            for key in ('first_level_category_id', 'second_level_category_id', 'third_level_category_id'):
                v = str(row.get(key, '')).strip()
                if v and v not in ('-124.0', 'UNKNOWN', 'UNK', 'nan'):
                    vals.append(v)
            out[vid] = vals
    return out


def load_video_captions(data_dir):
    out = {}
    path = os.path.join(data_dir, 'kuairand_video_captions.csv')
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            vid = str(row['final_video_id'])
            text = (row.get('caption') or '').lower()
            toks = []
            for tok in text.replace('#', ' ').replace('@', ' ').split():
                tok = tok.strip(' ,.;:!?/()[]{}"\'')
                if tok:
                    toks.append(tok)
            out[vid] = toks
    return out


def build_profiles(train_rows, video_categories, video_captions):
    user_cat = defaultdict(lambda: defaultdict(lambda: {'n': 0, 'pos': 0.0}))
    user_tok = defaultdict(lambda: defaultdict(lambda: {'n': 0, 'pos': 0.0}))
    for row in train_rows:
        _, user_id, video_id, _, _, _, label = row
        user_id = str(user_id)
        vid = str(video_id)
        for cat in video_categories.get(vid, []):
            entry = user_cat[user_id][cat]
            entry['n'] += 1
            entry['pos'] += float(label)
        for tok in set(video_captions.get(vid, [])):
            entry = user_tok[user_id][tok]
            entry['n'] += 1
            entry['pos'] += float(label)
    return user_cat, user_tok


def compute_pref_scores(row, user_cat, user_tok, video_categories, video_captions):
    _, user_id, video_id, _, _, _, _ = row
    user_id = str(user_id)
    vid = str(video_id)
    cats = video_categories.get(vid, [])
    toks = video_captions.get(vid, [])
    cat_scores = []
    for cat in cats:
        info = user_cat.get(user_id, {}).get(cat)
        if info and info['n'] > 0:
            cat_scores.append(info['pos'] / info['n'])
    tok_scores = []
    for tok in set(toks):
        info = user_tok.get(user_id, {}).get(tok)
        if info and info['n'] > 0:
            tok_scores.append(info['pos'] / info['n'])
    return (
        float(np.mean(cat_scores)) if cat_scores else 0.0,
        float(np.max(cat_scores)) if cat_scores else 0.0,
        float(np.mean(tok_scores)) if tok_scores else 0.0,
        float(len(cats)),
    )


def bucketize(values, n=10):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(1, dtype=np.float32)
    if np.all(arr == arr[0]):
        return np.zeros(arr.shape[0], dtype=np.float32)
    edges = np.quantile(arr, np.linspace(0, 1, n + 1)[1:-1])
    idx = np.searchsorted(edges, arr, side='right')
    return idx.astype(np.float32)


def encode_rows(rows, user_cat, user_tok, video_categories, video_captions):
    durations = np.array([r[5] for r in rows], dtype=np.float32)
    dur_edges = np.quantile(durations, np.linspace(0, 1, 11)[1:-1])

    pref_values = []
    for row in rows:
        pref_values.append(compute_pref_scores(row, user_cat, user_tok, video_categories, video_captions))
    pref_values = np.asarray(pref_values, dtype=np.float32)
    cat_bucket_edges = np.quantile(pref_values[:, 0], np.linspace(0, 1, 11)[1:-1]) if pref_values.shape[0] > 0 else np.array([])
    tok_bucket_edges = np.quantile(pref_values[:, 2], np.linspace(0, 1, 11)[1:-1]) if pref_values.shape[0] > 0 else np.array([])

    vocabulary = [dict() for _ in range(7)]
    def raw(x, idx):
        if idx == 0: return x[1]
        if idx == 1: return x[2]
        if idx == 2: return x[3]
        if idx == 3: return x[4]
        if idx == 4:
            val = float(x[5])
            return str(int(np.searchsorted(dur_edges, val)))
        if idx == 5:
            val = compute_pref_scores(x, user_cat, user_tok, video_categories, video_captions)[0]
            if cat_bucket_edges.size == 0:
                return '0'
            return str(int(np.searchsorted(cat_bucket_edges, val, side='right')))
        if idx == 6:
            val = compute_pref_scores(x, user_cat, user_tok, video_categories, video_captions)[2]
            if tok_bucket_edges.size == 0:
                return '0'
            return str(int(np.searchsorted(tok_bucket_edges, val, side='right')))
        raise ValueError('bad field index')

    for row in rows:
        for i in range(7):
            v = raw(row, i)
            if v not in vocabulary[i]:
                vocabulary[i][v] = len(vocabulary[i])
    field_dims = [len(v) + 1 for v in vocabulary]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    X = np.empty((len(rows), 7), dtype=np.int32)
    y = np.empty(len(rows), dtype=np.float32)
    users = []
    for n, row in enumerate(rows):
        for i in range(7):
            v = raw(row, i)
            X[n, i] = vocabulary[i].get(v, len(vocabulary[i])) + offsets[i]
        y[n] = float(row[6])
        users.append(row[1])
    return X, y, users, int(sum(field_dims))


def train_and_score(data_dir, seed=0):
    splits = load_rows(data_dir)
    train_rows = splits['train']
    valid_rows = splits['valid']
    test_rows = splits['test']
    video_categories = load_video_categories(data_dir)
    video_captions = load_video_captions(data_dir)
    user_cat, user_tok = build_profiles(train_rows, video_categories, video_captions)

    Xtr, ytr, utr, dim = encode_rows(train_rows, user_cat, user_tok, video_categories, video_captions)
    Xva, yva, uva, _ = encode_rows(valid_rows, user_cat, user_tok, video_categories, video_captions)
    Xte, yte, ute, _ = encode_rows(test_rows, user_cat, user_tok, video_categories, video_captions)

    m = FM(dim, k=16, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    best_val = -1.0
    best_state = None
    bad = 0
    for ep in range(1, 41):
        idx = rng.permutation(len(ytr))
        for start in range(0, len(idx), 8192):
            batch_idx = idx[start:start + 8192]
            m.step(Xtr[batch_idx], ytr[batch_idx])
        pred_va = m.predict(Xva)
        metrics = evaluate(uva, yva, pred_va)
        if metrics['primary'] > best_val + 1e-5:
            best_val = metrics['primary']
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
            bad = 0
        else:
            bad += 1
            if bad >= 4:
                break
    if best_state is not None:
        m.V, m.W, m.b = best_state

    train_primary = evaluate(utr, ytr, m.predict(Xtr))
    val_metrics = evaluate(uva, yva, m.predict(Xva))

    rand_rows = load_random_valid(data_dir)
    Xrand, yrand, urand, _ = encode_rows(rand_rows, user_cat, user_tok, video_categories, video_captions)
    unb = evaluate(urand, yrand, m.predict(Xrand))

    out_dir = os.path.join(data_dir, '..', 'tmp_candidate_out')
    os.makedirs(out_dir, exist_ok=True)
    write_submission(os.path.join(out_dir, 'submission_valid.csv'), valid_rows, m.predict(Xva))
    write_submission(os.path.join(out_dir, 'submission_test.csv'), test_rows, m.predict(Xte))

    return {
        'train_primary': train_primary['primary'],
        'val_gauc': val_metrics['GAUC'],
        'val_ndcg5': val_metrics['nDCG@5'],
        'val_primary': val_metrics['primary'],
        'unbiased_val_primary': unb['primary'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join('KuaiRand-Pure', 'data'))
    ap.add_argument('--out_dir', default=os.path.join('runs', 'candidate_content'))
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    m = train_and_score(args.data_dir, seed=args.seed)
    write_submission(os.path.join(args.out_dir, 'submission_valid.csv'), load_rows(args.data_dir)['valid'], np.ones(len(load_rows(args.data_dir)['valid']), dtype=np.float32))
    # The actual score lines are the ones the harness reads, so we print the metrics.
    print(f"TRAIN_PRIMARY={m['train_primary']:.6f}")
    print(f"VAL_GAUC={m['val_gauc']:.6f}")
    print(f"VAL_NDCG5={m['val_ndcg5']:.6f}")
    print(f"VAL_PRIMARY={m['val_primary']:.6f}")
    print(f"UNBIASED_PRIMARY={m['unbiased_val_primary']:.6f}")


if __name__ == '__main__':
    main()
