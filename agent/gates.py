"""Deterministic validation of a proposed experiment. No LLM involved.

The model proposes; plain code decides what is admissible, because every question here --
protected file, already tried, falsifiable -- is one code answers more reliably.

Gates never stall the run: a rejected spec is replanned once, then proceeds with a warning.
"""
import re
from typing import List, Tuple

from . import config
from .spec import REQUIRED_TEXT_FIELDS, ExperimentSpec

MAX_SCOPE_FILES = 2
_MIN_FIELD_CHARS = 15


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, so wording differences do not hide a repeat."""
    return re.sub(r'[^a-z0-9 ]+', ' ', text.lower()).strip()


def _token_set(text: str) -> set:
    """Content words over three characters, used to compare two proposals."""
    return {t for t in _normalise(text).split() if len(t) > 3}


def check_spec(spec: ExperimentSpec, journal) -> Tuple[bool, List[str]]:
    """Returns (ok, reasons). Reasons are phrased for the replan prompt, so they say what to
    fix rather than merely what is wrong."""
    reasons: List[str] = []

    # completeness
    for field in REQUIRED_TEXT_FIELDS:
        value = (getattr(spec, field, '') or '').strip()
        if not value:
            reasons.append(f'`{field}` is empty and is required')
        elif len(value) < _MIN_FIELD_CHARS and field != 'target_metric':
            reasons.append(f'`{field}` is too short to be meaningful: {value!r}')

    if not spec.tags:
        reasons.append('`tags` is empty; it is needed to detect duplicate experiments')

    # legality -- the starter kit defines the task and must never be edited
    protected = {p.lower() for p in config.PROTECTED_FILES}
    for path in spec.implementation_scope:
        if path.strip().lower().lstrip('./').split('/')[-1] in protected:
            reasons.append(f'`implementation_scope` includes protected file {path!r}; '
                           f'{", ".join(sorted(protected))} are read-only')

    # scope -- a change spanning many files is not one atomic change
    if len(spec.implementation_scope) > MAX_SCOPE_FILES:
        reasons.append(f'`implementation_scope` lists {len(spec.implementation_scope)} files; '
                       f'at most {MAX_SCOPE_FILES} for a single atomic change')

    # duplicate -- do not spend a scarce iteration re-running a settled question
    dup = _find_duplicate(spec, journal)
    if dup is not None:
        reasons.append(
            f'node {dup.id} already ran essentially this change '
            f'(tags {sorted(set(getattr(dup, "spec", {}).get("tags", [])))}). Re-testing the '
            f'same mechanism is legitimate when the *implementation* differs materially -- a '
            f'different approximation, sampling scheme or formulation is a genuinely new '
            f'question. If that is what you intend, say concretely how this implementation '
            f'differs from node {dup.id}. Otherwise propose a different mechanism.')

    # citations -- may only cite identifiers a real search actually returned
    registry = getattr(journal, 'citation_registry', {}) or {}
    if registry:
        for cid in spec.cited_ids:
            if cid.lower() not in {k.lower() for k in registry}:
                reasons.append(f'cited identifier {cid!r} was never returned by a search this '
                               f'run; cite only ids from search_papers results')
    elif spec.cited_ids:
        reasons.append('literature is cited but no search was performed this run; '
                       'call search_papers before citing')

    return (not reasons), reasons


def _jaccard(a: set, b: set) -> float:
    """Overlap of two token sets, 0.0 when either is empty."""
    return len(a & b) / max(len(a | b), 1) if a and b else 0.0


# A repeat is the same mechanism carried out the same way, so the implementation threshold is
# the binding one. Two experiments sharing a mechanism can still differ: a sampled
# approximation versus an exact computation is a real second question, not a repeat.
_IMPL_OVERLAP = 0.60
_TAG_OVERLAP = 0.40
_MECH_OVERLAP = 0.50


def _find_duplicate(spec: ExperimentSpec, journal):
    """A prior node proposing near-identical work, judged on the change rather than the label."""
    mine_impl = _token_set(spec.proposed_change)
    if not mine_impl:
        return None
    tags = {t.lower() for t in spec.tags}
    mine_mech = _token_set(getattr(spec, 'mechanism', '') or '')

    for node in getattr(journal, 'nodes', []):
        prior = getattr(node, 'spec', None)
        if not prior:
            continue
        impl = _jaccard(mine_impl, _token_set(prior.get('proposed_change', '')))
        if impl < _IMPL_OVERLAP:
            continue
        tag_sim = _jaccard(tags, {str(t).lower() for t in (prior.get('tags') or [])})
        mech_sim = _jaccard(mine_mech, _token_set(prior.get('mechanism', '')))
        if tag_sim >= _TAG_OVERLAP or mech_sim >= _MECH_OVERLAP:
            return node
    return None
