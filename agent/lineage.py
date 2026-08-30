"""Deterministic feature provenance checks for causal ML features.

This module enforces the core separation the agent needs: static metadata is safe to join
into every split, but behavior features must come from train-only history and cannot use
future evaluation rows or random-exposure rows as fitting context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple


_RANDOM_EXPOSURE_ALIASES = {
    'random',
    'random_exposure',
    'random-exposure',
    'random exposure',
    'randomexp',
}


@dataclass(frozen=True)
class FeatureProvenance:
    """Describe where a feature comes from and whether it is safe to use.

    `source` captures whether the feature is static metadata or a behavioral statistic.
    `split_scope` says which split the feature may be fit on; `row_scope` says whether it
    uses strictly-past rows, evaluation rows, or all rows.
    """

    name: str
    source: str = 'behavior'
    split_scope: str = 'train'
    row_scope: str = 'train_only'
    uses_label: bool = False
    strict_past: bool = True
    notes: str = ''


def _normalise(value: Optional[str]) -> str:
    if value is None:
        return ''
    return str(value).strip().lower().replace('-', '_').replace(' ', '_')


def is_random_exposure_split(split_name: Optional[str]) -> bool:
    name = _normalise(split_name)
    return name in _RANDOM_EXPOSURE_ALIASES or 'random' in name


def static_metadata_feature(name: str, source: str = 'metadata') -> FeatureProvenance:
    return FeatureProvenance(
        name=name,
        source=source,
        split_scope='all',
        row_scope='static',
        uses_label=False,
        strict_past=True,
        notes='Static metadata is safe because it is known before the row is scored.',
    )


def strict_past_feature(
    name: str,
    *,
    split_scope: str = 'train',
    uses_label: bool = False,
    row_scope: str = 'train_only',
    notes: str = '',
) -> FeatureProvenance:
    return FeatureProvenance(
        name=name,
        source='behavior',
        split_scope=_normalise(split_scope),
        row_scope=_normalise(row_scope),
        uses_label=uses_label,
        strict_past=True,
        notes=notes,
    )


def check_feature_lineage(feature: FeatureProvenance, split_name: Optional[str]) -> Tuple[bool, str]:
    """Return (ok, reason) for a single feature's causal validity.

    Safe cases are intentionally narrow: static metadata across all splits, and behavioral
    features frozen from the training split only. Anything using evaluation rows, labels from
    future rows, or random-exposure rows is rejected.
    """
    if feature is None:
        return False, 'feature provenance is missing'

    name = feature.name or 'unnamed'
    split_key = _normalise(split_name)

    if is_random_exposure_split(split_key):
        return False, (
            f'feature {name!r} is trying to use random-exposure rows; those rows are '
            'evaluation-only and must never be used for fitting or feature construction.'
        )

    if feature.source in {'metadata', 'static', 'static_metadata'}:
        return True, f'feature {name!r} is static metadata and remains safe for {split_key or "all splits"}.'

    if not feature.strict_past:
        return False, (
            f'feature {name!r} is not marked as strictly past-only; a future row or '
            'post-time statistic is being used.'
        )

    if feature.row_scope in {'evaluation', 'all_rows', 'future', 'post_time', 'post_time_rows'}:
        return False, (
            f'feature {name!r} uses evaluation or future rows; this is invalid because it '
            'depends on information not available before the scored row.'
        )

    if feature.uses_label and feature.row_scope not in {'train_only', 'strict_past', 'frozen'}:
        return False, (
            f'feature {name!r} uses labels from a non-train-only scope; a label-derived '
            'statistic cannot be fitted on evaluation rows.'
        )

    if feature.split_scope in {'valid', 'test', 'all'} and feature.row_scope in {'train_only', 'strict_past', 'frozen'}:
        # This is the intended safe pattern: a train-only statistic may be frozen and then
        # applied to later rows; the causal boundary is on the training-time fit, not on the
        # entity overlap itself.
        return True, f'feature {name!r} is frozen from train-only history and remains valid for {split_key}.'

    if feature.split_scope in {'train', 'train_only'} and feature.row_scope in {'train_only', 'strict_past', 'frozen'}:
        return True, f'feature {name!r} is built from train-only history and is safe for {split_key}.'

    return False, (
        f'feature {name!r} is not valid under the lineage rule; it must be static metadata '
        'or a strictly train-only, frozen behavior statistic.'
    )


def validate_feature_list(features: Iterable[FeatureProvenance], split_name: Optional[str]) -> List[str]:
    """Apply check_feature_lineage to many features and return only failure reasons."""
    reasons: List[str] = []
    for feature in features:
        ok, reason = check_feature_lineage(feature, split_name)
        if not ok:
            reasons.append(reason)
    return reasons
