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
from .lineage import FeatureProvenance, check_feature_lineage
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

    # exhausted mechanism -- a family with three or more attempts, none accepted and none even
    # inside the noise band. Unlike the duplicate check below this is not a "say why" warning:
    # runs/v6 flagged its tree-model repeats seven times and the planner argued past every one,
    # spending 23% of the run on a mechanism whose every attempt came back worse.
    from . import diagnose
    dead = {f: ids for f, ids in diagnose.exhausted_mechanisms(journal)}
    if dead:
        for family in sorted(diagnose.families_of(spec.tags) & set(dead)):
            ids = ', '.join(f'node {i}' for i in dead[family])
            reasons.append(
                f'`{family}` is closed: {len(dead[family])} attempts ({ids}), none accepted and '
                f'none even inside the noise band. Choose a different mechanism -- a further '
                f'variation on this one is not a valid proposal, and arguing that yours differs '
                f'is not an exception')

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

    # provenance -- every experiment must declare the lineage semantics of the feature it adds
    feature_provenance = getattr(spec, 'feature_provenance', None)
    if feature_provenance:
        if isinstance(feature_provenance, FeatureProvenance):
            ok, reason = check_feature_lineage(feature_provenance, 'valid')
            if not ok:
                reasons.append(f'feature provenance rejected: {reason}')
        elif isinstance(feature_provenance, list):
            for feature in feature_provenance:
                if isinstance(feature, FeatureProvenance):
                    ok, reason = check_feature_lineage(feature, 'valid')
                    if not ok:
                        reasons.append(f'feature provenance rejected: {reason}')

    return (not reasons), reasons


TAG_OVERLAP_THRESHOLD = 0.5

# Tags are written freehand, so the same mechanism arrives under different names. The v7 smoke
# run produced `gbdt, target-encoding-ctr, item-level-statistics` and `lightgbm,
# train-window-ctr-features, target-encoding` on consecutive drafts -- the same idea, sharing no
# tag string, so exact matching saw nothing. Folding the model-family and feature-family
# synonyms together and also indexing each hyphen-separated token brings those two to 0.500
# while leaving genuinely distinct pairs below the bar (measured on runs/v5: nodes 6 and 7,
# which are the same experiment, score 1.000; nodes 6 and 15, GBDT-over-CTR against FM-over-CTR,
# score 0.429 and correctly do not match).
_TAG_SYNONYMS = {
    'lightgbm': 'gbdt', 'xgboost': 'gbdt', 'gbm': 'gbdt', 'gradient-boosting': 'gbdt',
    'boosted-trees': 'gbdt', 'catboost': 'gbdt',
    'ctr': 'target-encoding', 'ctr-features': 'target-encoding',
    'target-encoding-ctr': 'target-encoding', 'rate-features': 'target-encoding',
    'item-level-statistics': 'target-encoding', 'popularity-features': 'target-encoding',
}


def _normalise_tags(tags) -> set:
    """A tag set expanded to its synonyms and constituent tokens, for overlap comparison."""
    out = set()
    for t in tags or ():
        t = str(t).lower().strip()
        if not t:
            continue
        t = _TAG_SYNONYMS.get(t, t)
        out.add(t)
        for tok in t.split('-'):
            if len(tok) > 2:
                out.add(_TAG_SYNONYMS.get(tok, tok))
    return out


def _find_duplicate(spec: ExperimentSpec, journal):
    """A prior scoring node testing the same mechanism.

    Tags are matched by overlap rather than exact equality. Requiring identical tag sets let
    `user-category-affinity`, `user-author-affinity` and `user-tag-affinity` through as three
    distinct experiments in runs/v2 when they were one mechanism -- a sparse per-(user, X) label
    rate -- wearing three join keys, and all three overfit identically.

    Tags decide alone. There used to be a second condition, a >=0.7 bag-of-words overlap on
    `proposed_change`, and it is what let runs/v5 run "LightGBM over marginal CTR rates" seven
    times (nodes 2, 3, 4, 6, 7, 9, 14) for 37% of the run, every one a regression. Nodes 6 and 7
    carry *identical* tag sets and were the same experiment, but "Replace FM with LightGBM
    trained on..." and "Fix the LightGBM free_raw_data bug..." share too few tokens to clear the
    bar. Measuring the overlap across v5 shows why no threshold rescues it: for pairs that are
    the same idea the median overlap is 0.239, and for pairs that are different ideas the
    maximum is 0.281. The distributions do not separate, so the condition was contributing noise
    and vetoing real matches, and it is gone.

    A duplicate is a soft finding -- it triggers a replan, which proceeds if the planner can
    justify the difference -- so an over-eager match costs an explanation, not an iteration.
    """
    tags = _normalise_tags(spec.tags)
    if not tags:
        return None
    for node in getattr(journal, 'nodes', []):
        prior_tags = _normalise_tags(((getattr(node, 'spec', None) or {}).get('tags') or []))
        if not prior_tags:
            continue
        if len(prior_tags & tags) / max(len(prior_tags | tags), 1) >= TAG_OVERLAP_THRESHOLD:
            return node
    return None
