"""Deterministic validation of a proposed experiment. No LLM involved.

The model proposes; this decides what is admissible. Everything here is a question plain code
can answer better than a model can -- "does this touch a protected file", "have we tried this
already", "is there a falsification condition" -- so asking a model would be slower, costlier
and less reliable.

Gates never stall the run. A rejected spec is replanned once; if the replan also fails the
orchestrator proceeds and records the warning, because a stuck loop is worse than a flawed
experiment.
"""
import re
from typing import List, Tuple

from . import config
from .spec import REQUIRED_TEXT_FIELDS, ExperimentSpec

MAX_SCOPE_FILES = 2
_MIN_FIELD_CHARS = 15


def _normalise(text: str) -> str:
    return re.sub(r'[^a-z0-9 ]+', ' ', text.lower()).strip()


def _token_set(text: str) -> set:
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
        reasons.append(f'node {dup.id} already tested this '
                       f'(tags {sorted(set(getattr(dup, "spec", {}).get("tags", [])))}); '
                       f'propose a different change or say why the retry is warranted')

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


def _find_duplicate(spec: ExperimentSpec, journal):
    """A prior scoring node whose tags match and whose change is near-identical."""
    tags = {t.lower() for t in spec.tags}
    if not tags:
        return None
    mine = _token_set(spec.proposed_change)
    for node in getattr(journal, 'nodes', []):
        prior = getattr(node, 'spec', None)
        if not prior:
            continue
        prior_tags = {str(t).lower() for t in (prior.get('tags') or [])}
        if not prior_tags or prior_tags != tags:
            continue
        theirs = _token_set(prior.get('proposed_change', ''))
        if not mine or not theirs:
            continue
        overlap = len(mine & theirs) / max(len(mine | theirs), 1)
        if overlap >= 0.7:
            return node
    return None
