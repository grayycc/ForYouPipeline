"""Auxiliary training targets, aligned to `data.load()`'s rows so nobody has to hand-roll it.

Why this file exists. `long_view` is not the only supervision in the logs. Eleven columns record
what happened during an impression, and several are strongly related to the label -- measured
over 400,000 training rows, `P(long_view | is_click=1) = 0.7226` against
`P(long_view | is_click=0) = 0.0030`. None of them may be a *feature*: at serving time the video
has not been watched, so they do not exist. All of them may be a *target*, because fitting
parameters on training labels is training, and the model is scored through its `long_view` head
alone.

Getting the alignment right is the whole difficulty, and it is not a research question. The
labels have to line up row-for-row with `load(data_dir)['train']` or the auxiliary head learns
noise -- silently, since a misaligned target still trains and still produces a plausible score.
Hand-rolled versions of this drew a leakage flag for exactly that reason ("if the row ordering
in that CSV does not match the row ordering produced by data.load()"), which was a fair
objection to unverifiable code. So the alignment lives here and is asserted in
tests/test_aux_labels.py.

    from aux_labels import train_aux_labels

    clicks = train_aux_labels(data_dir, 'is_click')      # float32, len == len(splits['train'])
    watch  = train_aux_labels(data_dir, 'play_time_ms')  # continuous, for a regression head

Only the train-window file is ever opened. Validation and test rows live in a different file and
are never read, so there is no path by which an evaluation row's post-interaction value can
reach the model.
"""
import csv
import os

import numpy as np

from data import SPLITS

# The train split is exactly this file, in this order. `load()` concatenates the two log files
# and then filters by date with a list comprehension, which preserves order; every row of the
# train-window file falls inside the train date range (measured: 1,141,112 rows, 0 outside). So
# reading it top to bottom reproduces `load(data_dir)['train']` index for index.
_TRAIN_FILE = 'log_standard_4_08_to_4_21_pure.csv'

# Columns that describe what happened during the impression. Usable as targets, never as
# features. `long_view` is excluded: it is the main label, not an auxiliary one.
BINARY = ('is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate',
          'is_profile_enter')
CONTINUOUS = ('play_time_ms', 'profile_stay_time', 'comment_stay_time')


def train_aux_labels(data_dir, column, log1p=True):
    """One auxiliary target per training row, aligned to `load(data_dir)['train']`.

    Binary columns come back as 0.0/1.0. Continuous ones come back `log1p`-scaled by default,
    because `play_time_ms` spans several orders of magnitude and a raw-scale regression head
    would dominate the shared embeddings; pass `log1p=False` for the raw value.
    """
    if column not in BINARY and column not in CONTINUOUS:
        raise ValueError(
            f'{column!r} is not an auxiliary target. Binary: {BINARY}. Continuous: {CONTINUOUS}. '
            f'`long_view` is the main label; features must come from data.load().')

    lo, hi = SPLITS['train']
    out = []
    with open(os.path.join(data_dir, _TRAIN_FILE)) as fh:
        for r in csv.DictReader(fh):
            date = int(r['date'])
            if not (lo <= date <= hi):
                # Never observed, but a silently-dropped row would misalign every later index.
                raise ValueError(f'{_TRAIN_FILE} contains date {date} outside the train window')
            if column in BINARY:
                out.append(1.0 if r[column] != '0' else 0.0)
            else:
                out.append(float(r[column]))

    arr = np.asarray(out, dtype=np.float32)
    if column in CONTINUOUS and log1p:
        arr = np.log1p(np.maximum(arr, 0.0)).astype(np.float32)
    return arr


def positive_rate(labels):
    """Share of positives, for weighting an auxiliary head against the main one."""
    return float(np.mean(labels > 0))
