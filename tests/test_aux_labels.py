"""The alignment kit/aux_labels.py claims has to be asserted, not asserted-in-a-comment.

A misaligned auxiliary target does not crash. It trains, produces a plausible score, and teaches
the shared embeddings noise -- which is the failure this file exists to make impossible.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kit'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent import config
from aux_labels import BINARY, CONTINUOUS, positive_rate, train_aux_labels
from data import load

_TRAIN = load(config.DATA_DIR)['train']


def test_length_matches_the_train_split_exactly():
    assert len(train_aux_labels(config.DATA_DIR, 'is_click')) == len(_TRAIN)


def test_is_click_lines_up_row_for_row_with_load():
    """The claim that makes the file safe: index i of the labels is the same impression as
    index i of load()['train']. Checked against the raw file the split is built from."""
    import csv
    clicks = train_aux_labels(config.DATA_DIR, 'is_click')
    path = os.path.join(config.DATA_DIR, 'log_standard_4_08_to_4_21_pure.csv')
    with open(path) as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if i % 9973:                      # sample; a full pass is a minute of no extra signal
                continue
            assert r['user_id'] == _TRAIN[i][1]
            assert r['video_id'] == _TRAIN[i][2]
            assert clicks[i] == (1.0 if r['is_click'] != '0' else 0.0)


def test_is_click_carries_the_signal_that_motivates_the_helper():
    """P(long_view | click) ~ 0.72 against ~0.003 without. If this ever stops holding, the
    reason for a multi-task head has gone with it."""
    clicks = train_aux_labels(config.DATA_DIR, 'is_click')
    lv = np.asarray([r[6] for r in _TRAIN], dtype=np.float32)
    with_click = lv[clicks > 0].mean()
    without = lv[clicks == 0].mean()
    assert with_click > 0.70, with_click
    assert without < 0.01, without


def test_continuous_target_is_log_scaled_by_default():
    raw = train_aux_labels(config.DATA_DIR, 'play_time_ms', log1p=False)
    scaled = train_aux_labels(config.DATA_DIR, 'play_time_ms')
    assert raw.max() > 1000
    assert scaled.max() < 25          # log1p of even a very long watch stays small
    assert np.allclose(scaled, np.log1p(np.maximum(raw, 0.0)), atol=1e-4)


def test_the_main_label_is_rejected_as_an_auxiliary_target():
    for bad in ('long_view', 'user_id', 'nonsense'):
        try:
            train_aux_labels(config.DATA_DIR, bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad!r} should not be accepted as an auxiliary target')


def test_every_declared_column_actually_loads():
    for col in BINARY + CONTINUOUS:
        arr = train_aux_labels(config.DATA_DIR, col)
        assert len(arr) == len(_TRAIN), col
        assert np.isfinite(arr).all(), col


def test_positive_rate_matches_the_documented_balance():
    """is_click is far better balanced than long_view, which is why it regularises well."""
    clicks = train_aux_labels(config.DATA_DIR, 'is_click')
    assert 0.35 < positive_rate(clicks) < 0.55
