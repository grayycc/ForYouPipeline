"""Leakage-safe preference features from video captions and categories.

This module is intentionally lightweight and pure-stdlib + numpy. It does not attempt a full
end-to-end model; instead it creates user-content preference summaries and video content vocabularies
that can be fed into an FM or ranking model using only information from the training split.

The main design goal is to model user preference rather than exposure bias:
  - user × category affinity from prior training impressions
  - user token/profile affinity from prior watched video captions
  - video category and caption metadata for candidate-side matching

All aggregates are computed from the training rows only and are therefore valid for future rows in
train/valid/test under a causal use convention.
"""
import csv
import math
import os
import re
from collections import defaultdict

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fa5]+")

# The caption and category files are sized for the whole KuaiRand family (32M rows, 3+ GB
# each), but KuaiRand-Pure uses only video ids 0..7582 -- and those are the first 7,583
# records of both files, in ascending order. Verified by matching the caption file's
# `duration` against video_features_basic_pure.csv's `video_duration`: 7,583 of 7,583 agree.
# Reading to EOF costs ~55 s and ~4 GB resident per file to collect ids from the 1k/27k
# variants that are never scored here; stopping at 7582 costs ~0.02 s and ~2 MB.
MAX_PURE_VIDEO_ID = 7582


def _pure_video_id(row):
    """Normalised key, or None past the end of Pure's id range.

    The id column reads as '0', '1', ... while category *values* carry a '.0' suffix
    ('39.0'), so both are parsed through float and re-rendered as a plain integer string.
    That is what makes the join to data.load()'s string `video_id` land -- a silent key
    mismatch here yields an all-UNK feature that looks like a failed idea rather than a bug.
    """
    raw = row.get('final_video_id')
    if raw is None:
        return None
    try:
        vid = int(float(raw))
    except (TypeError, ValueError):
        return None
    return str(vid) if vid <= MAX_PURE_VIDEO_ID else None


def load_video_categories(data_dir):
    """Return video -> [first, second, third] category ids from the additional category file."""
    path = os.path.join(data_dir, 'kuairand_video_categories.csv')
    out = {}
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            vid = _pure_video_id(row)
            if vid is None:
                if row.get('final_video_id'):
                    break                       # ascending file: past Pure's id range
                continue
            cats = []
            for key in ('first_level_category_id', 'second_level_category_id', 'third_level_category_id'):
                val = row.get(key, '').strip()
                if val and val not in ('-124.0', 'UNKNOWN', 'UNK', 'nan'):
                    cats.append(val)
            out[vid] = cats
    return out


def load_video_captions(data_dir):
    """Return video -> normalized token list from the additional caption file."""
    path = os.path.join(data_dir, 'kuairand_video_captions.csv')
    out = {}
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            vid = _pure_video_id(row)
            if vid is None:
                if row.get('final_video_id'):
                    break                       # ascending file: past Pure's id range
                continue
            text = (row.get('caption') or '').lower()
            toks = TOKEN_RE.findall(text)
            out[vid] = toks
    return out


def _safe_rate(pos, total):
    if total <= 0:
        return 0.0
    return pos / total


def compute_user_content_profiles(train_rows, video_categories, video_captions):
    """Compute causal user-category and user-caption preference profiles from train rows only."""
    user_cat = defaultdict(lambda: defaultdict(lambda: {'views': 0, 'long_views': 0}))
    user_tokens = defaultdict(lambda: defaultdict(lambda: {'views': 0, 'long_views': 0}))

    for row in train_rows:
        date, user_id, video_id, author_id, tab, duration_ms, label = row
        cats = video_categories.get(str(video_id), [])
        toks = video_captions.get(str(video_id), [])

        if not cats and not toks:
            continue

        for cat in cats:
            entry = user_cat[user_id][cat]
            entry['views'] += 1
            entry['long_views'] += int(label)

        for tok in set(toks):
            entry = user_tokens[user_id][tok]
            entry['views'] += 1
            entry['long_views'] += int(label)

    profiles = {}
    for user_id, hist in user_cat.items():
        profile = {}
        for cat, stats in hist.items():
            profile[cat] = {
                'rate': _safe_rate(stats['long_views'], stats['views']),
                'views': stats['views'],
                'long_views': stats['long_views'],
            }
        profiles.setdefault('user_category', {})[user_id] = profile

    for user_id, hist in user_tokens.items():
        profile = {}
        for tok, stats in hist.items():
            profile[tok] = {
                'rate': _safe_rate(stats['long_views'], stats['views']),
                'views': stats['views'],
                'long_views': stats['long_views'],
            }
        profiles.setdefault('user_tokens', {})[user_id] = profile

    return profiles


def row_content_features(row, user_profiles, video_categories, video_captions):
    """Return a compact scalar feature dict for one row.

    This is intentionally deterministic and small. A later model can use these values as extra
    numerical features or embed them into categorical buckets. The values are causal because they
    are computed from historical train statistics only.
    """
    _, user_id, video_id, _, _, _, _ = row
    user_name = str(user_id)
    vid_name = str(video_id)

    categories = video_categories.get(vid_name, [])
    tokens = video_captions.get(vid_name, [])

    feat = {
        'user_cat_count': 0,
        'user_cat_rate_mean': 0.0,
        'user_cat_rate_max': 0.0,
        'user_token_score': 0.0,
        'video_cat_count': len(categories),
    }

    cat_stats = user_profiles.get('user_category', {}).get(user_name, {})
    if cat_stats:
        vals = [v['rate'] for v in cat_stats.values()]
        feat['user_cat_count'] = len(cat_stats)
        feat['user_cat_rate_mean'] = float(np.mean(vals)) if vals else 0.0
        feat['user_cat_rate_max'] = float(np.max(vals)) if vals else 0.0

    if categories:
        rates = []
        for cat in categories:
            info = cat_stats.get(cat)
            if info:
                rates.append(info['rate'])
        if rates:
            feat['user_cat_rate_max'] = max(feat['user_cat_rate_max'], float(np.max(rates)))
            feat['user_cat_rate_mean'] = float(np.mean(rates))

    token_stats = user_profiles.get('user_tokens', {}).get(user_name, {})
    if token_stats:
        scores = []
        for tok in set(tokens):
            info = token_stats.get(tok)
            if info:
                scores.append(info['rate'])
        if scores:
            feat['user_token_score'] = float(np.mean(scores))

    return feat


def summarize_content_features(data_dir):
    """Quick profile for the data, useful when deciding whether content features are promising."""
    cats = load_video_categories(data_dir)
    caps = load_video_captions(data_dir)
    print(f'video_categories={len(cats)} video_captions={len(caps)}')
    counts = {}
    for vid, items in cats.items():
        for item in items:
            counts[item] = counts.get(item, 0) + 1
    if counts:
        print('top_categories=', sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10])
    token_counts = defaultdict(int)
    for toks in caps.values():
        for tok in toks:
            token_counts[tok] += 1
    print('top_tokens=', sorted(token_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20])
    return len(cats), len(caps), counts, token_counts


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='KuaiRand-Pure/data')
    args = ap.parse_args()
    summarize_content_features(args.data_dir)
