"""Feature-generation template for autonomous solutions.

This file is a helper for the agent to follow when it creates a new standalone solution: it
shows the pattern the generated code should use when it computes new preference features in the
same file. The agent is expected to generate this logic inside each solution, not rely on a
static repo-wide feature file.
"""

import csv
import os
import re
from collections import defaultdict

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fa5]+")

# See kit/content_features.py for the measurements behind this: the content files cover the
# whole KuaiRand family, but Pure is ids 0..7582 and those are the first records of both
# files. Stopping there turns a ~55 s, ~4 GB read into a ~0.02 s, ~2 MB one.
MAX_PURE_VIDEO_ID = 7582


def _pure_video_id(row):
    """Normalised key ('39.0' and '39' both become '39'), or None past Pure's id range."""
    raw = row.get('final_video_id')
    if raw is None:
        return None
    try:
        vid = int(float(raw))
    except (TypeError, ValueError):
        return None
    return str(vid) if vid <= MAX_PURE_VIDEO_ID else None


def load_video_categories(data_dir):
    path = os.path.join(data_dir, 'kuairand_video_categories.csv')
    out = {}
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            vid = _pure_video_id(row)
            if vid is None:
                if row.get('final_video_id'):
                    break
                continue
            cats = []
            for key in ('first_level_category_id', 'second_level_category_id', 'third_level_category_id'):
                v = row.get(key, '').strip()
                if v and v not in ('-124.0', 'UNKNOWN', 'UNK', 'nan'):
                    cats.append(v)
            out[vid] = cats
    return out


def load_video_captions(data_dir):
    path = os.path.join(data_dir, 'kuairand_video_captions.csv')
    out = {}
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            vid = _pure_video_id(row)
            if vid is None:
                if row.get('final_video_id'):
                    break
                continue
            text = (row.get('caption') or '').lower()
            tok = TOKEN_RE.findall(text)
            out[vid] = tok
    return out


def compute_user_profiles(train_rows, video_categories, video_captions):
    """Build train-only user-category and user-topic preference profiles."""
    user_cat = defaultdict(lambda: defaultdict(lambda: {'views': 0, 'long_views': 0}))
    user_tok = defaultdict(lambda: defaultdict(lambda: {'views': 0, 'long_views': 0}))

    for row in train_rows:
        _, user_id, video_id, _, _, _, label = row
        label = int(label)
        cats = video_categories.get(str(video_id), [])
        toks = video_captions.get(str(video_id), [])
        for cat in cats:
            user_cat[str(user_id)][cat]['views'] += 1
            user_cat[str(user_id)][cat]['long_views'] += label
        for tok in set(toks):
            user_tok[str(user_id)][tok]['views'] += 1
            user_tok[str(user_id)][tok]['long_views'] += label
    return user_cat, user_tok


def append_content_features(rows, user_cat, user_tok, video_categories, video_captions):
    """Return a small, causal, preference-driven feature table for each row."""
    feats = []
    for row in rows:
        _, user_id, video_id, _, _, _, _ = row
        user_id = str(user_id)
        vid = str(video_id)
        cat_match = []
        for cat in video_categories.get(vid, []):
            info = user_cat.get(user_id, {}).get(cat)
            if info and info['views'] > 0:
                cat_match.append(info['long_views'] / info['views'])

        tok_match = []
        for tok in set(video_captions.get(vid, [])):
            info = user_tok.get(user_id, {}).get(tok)
            if info and info['views'] > 0:
                tok_match.append(info['long_views'] / info['views'])

        feats.append([
            float(np.mean(cat_match)) if cat_match else 0.0,
            float(np.max(cat_match)) if cat_match else 0.0,
            float(np.mean(tok_match)) if tok_match else 0.0,
            float(len(video_categories.get(vid, []))),
        ])
    return np.asarray(feats, dtype=np.float32)


if __name__ == '__main__':
    print('This is a template for feature-generation logic inside a standalone solution.')
