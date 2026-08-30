"""Unbiased (random-exposure) scoring, done once so every solution stops reimplementing it.

Why this file exists. `UNBIASED_PRIMARY` is required of every solution, and every solution was
writing its own ~40 lines to load `log_random_4_22_to_5_08_pure.csv`, filter it to the
validation window, join `author_id`, and re-encode it against the training vocabulary. That
boilerplate was the single largest source of both real bugs (one run stubbed the metric out as
`val_metrics['primary']`; another silently re-derived different `dur_bucket` edges) and of false
leakage reports, because the leakage reviewer re-litigated the same re-encoding dance on every
iteration and blocked half a run's nodes over it.

None of it is a research question. It has exactly one correct implementation, so it lives here,
is tested in tests/test_unbiased.py, and a solution reduces to one import.

The rows are the same tuple shape `data.load()` produces, so anything that consumes a split
consumes these:
    (date, user_id, video_id, author_id, tab, duration_ms, long_view)
"""
import csv
import os

import numpy as np

from data import FIELDS, LABEL, SPLITS, _bucket_edges

# The random-exposure log spans the valid+test window. Restricting it to the validation dates
# keeps the metric comparable with VAL_PRIMARY and keeps the test window untouched.
VALID_LO, VALID_HI = SPLITS['valid']


def load_random_valid(data_dir, lo=VALID_LO, hi=VALID_HI):
    """Random-exposure impressions inside the validation window, in data.load() row format.

    `author_id` is joined from video_features_basic_pure.csv exactly as data.load() does it --
    the log files carry no author column, and defaulting every row to 'UNK' instead would put
    these rows in a different feature space from the ones the model trained on.
    """
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    with open(os.path.join(data_dir, 'log_random_4_22_to_5_08_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            date = int(r['date'])
            if not (lo <= date <= hi):
                continue
            rows.append((date, r['user_id'], r['video_id'],
                         vid2author.get(r['video_id'], 'UNK'), r['tab'],
                         float(r['duration_ms']), 1 if r[LABEL] != '0' else 0))
    return rows


def encode_like_train(train_rows, target_rows):
    """Encode `target_rows` with the vocabulary and bucket edges derived from `train_rows`.

    This mirrors data.encode() exactly -- same quantile edges, same per-field vocabularies,
    same UNK slots, same offsets -- but takes the target rows as a plain argument instead of an
    extra dict key. tests/test_unbiased.py asserts the output is identical to
    `encode({'train': train_rows, 'x': target_rows})['x']`.

    Returns (X, y, users, dim), matching what enc[split] plus dim give you from data.encode().
    """
    edges = _bucket_edges([x[5] for x in train_rows])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    vocabs = [dict() for _ in FIELDS]
    for x in train_rows:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    X = np.empty((len(target_rows), len(FIELDS)), dtype=np.int32)
    y = np.empty(len(target_rows), dtype=np.float32)
    users = []
    for n, x in enumerate(target_rows):
        for i, v in enumerate(raw(x)):
            X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
        y[n] = x[6]
        users.append(x[1])
    return X, y, users, int(sum(field_dims))


def unbiased_primary(data_dir, train_rows, score_fn):
    """The number to print as UNBIASED_PRIMARY.

    `score_fn` maps rows -> one score per row. Two shapes are supported, so this works whether
    a solution scores from encoded features or straight from the raw tuples:

        # a model over data.encode()-style features
        X, y, users, dim = encode_like_train(train_rows, rows)
        unbiased_primary(data_dir, train_rows, lambda rows: model.predict(X))

        # or anything that consumes rows directly (a GBDT over its own feature frame, say)
        unbiased_primary(data_dir, train_rows, lambda rows: my_features_then_predict(rows))

    Returns a float. Never raises on an empty result -- it returns 0.0 -- because a missing
    metric line marks the whole node buggy and costs an iteration.
    """
    from evaluate import evaluate
    rows = load_random_valid(data_dir)
    if not rows:
        return 0.0
    scores = score_fn(rows)
    return float(evaluate([r[1] for r in rows], [r[6] for r in rows], scores)['primary'])
