"""The unbiased-scoring helper must be provably identical to what data.encode() produces.

The point of kit/unbiased.py is that solutions stop reimplementing this, so its equivalence to
the encoder they would otherwise call has to be asserted, not asserted-in-a-comment.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kit'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent import config
from data import encode, load
from unbiased import encode_like_train, load_random_valid, unbiased_primary

_SPLITS = load(config.DATA_DIR)


def test_random_valid_rows_match_the_documented_count():
    """288,338 is what every prior run independently measured for this window."""
    rows = load_random_valid(config.DATA_DIR)
    assert len(rows) == 288338, len(rows)


def test_rows_are_inside_the_validation_window():
    rows = load_random_valid(config.DATA_DIR)
    assert all(20220422 <= r[0] <= 20220428 for r in rows)


def test_author_is_joined_not_defaulted_to_unk():
    """Defaulting author_id to UNK would put these rows in a different feature space from the
    ones the model trained on, making the metric incomparable with validation."""
    rows = load_random_valid(config.DATA_DIR)
    unk = sum(1 for r in rows if r[3] == 'UNK')
    assert unk / len(rows) < 0.05, f'{unk}/{len(rows)} rows have no author'


def test_encode_like_train_matches_data_encode_exactly():
    """The equivalence the helper's docstring claims, actually checked."""
    tr = _SPLITS['train']
    target = load_random_valid(config.DATA_DIR)[:20000]
    X, y, users, dim = encode_like_train(tr, target)
    ref_enc, ref_dim = encode({'train': tr, 'x': target})
    Xr, yr, ur = ref_enc['x']
    assert dim == ref_dim, (dim, ref_dim)
    assert np.array_equal(X, Xr)
    assert np.array_equal(y, yr)
    assert users == ur


def test_unbiased_primary_scores_and_never_raises_on_empty():
    tr = _SPLITS['train']
    rows = load_random_valid(config.DATA_DIR)
    X, _, _, _ = encode_like_train(tr, rows)
    rng = np.random.default_rng(0)
    v = unbiased_primary(config.DATA_DIR, tr, lambda r: rng.random(len(r)))
    assert 0.0 < v < 1.0, v
    assert unbiased_primary(config.DATA_DIR, tr,
                            lambda r: []) is not None or True   # must not raise
