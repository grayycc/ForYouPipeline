"""Deterministic post-mortem on a finished node. No LLM involved.

Why this exists: in runs/v2 the agent could not distinguish "the model never trained" from "the
idea was wrong", and answered both the same way -- by abandoning the direction. Node 3's BPR ran
37 epochs with the loss stuck near chance and a train_primary *below the baseline's own*, which
means the hypothesis was never actually tested; the agent recorded it as refuted and cited it as
settled twice more. Meanwhile nodes 8, 11 and 16 were one mechanism (a sparse per-(user, X) label
rate) wearing three different join keys, each overfitting in the same way.

Both are questions plain arithmetic answers better than a model does, from numbers already on the
Node. The verdict then goes into the prompt, so the planner reasons from a diagnosis rather than
from a bare score.
"""
import re
import statistics
from typing import List, Optional

from . import config

UNDER_TRAINED = 'under_trained'
OVERFIT = 'overfit'
IMPROVEMENT = 'improvement'
REGRESSION = 'regression'
NOISE = 'noise'
BUGGY = 'buggy'
NO_SCORE = 'no_score'

# What each verdict means for what to do next. Kept here so the prompt and the log agree.
MEANING = {
    UNDER_TRAINED: 'the model never fit the training data, so the hypothesis was NOT tested; '
                   'this is an optimisation or implementation failure, not a refutation',
    OVERFIT: 'fit the training data and failed to generalise; the mechanism may be sound but '
             'the feature or capacity is too sparse for 1.14M rows',
    IMPROVEMENT: 'a real gain, outside the noise floor',
    REGRESSION: 'genuinely worse than its parent, outside the noise floor',
    NOISE: 'inside the noise floor; not evidence either way',
    BUGGY: 'did not run to completion',
    NO_SCORE: 'produced no score by design',
}


def _baseline_node(journal):
    for n in getattr(journal, 'nodes', []):
        if n.operation == 'baseline' and not n.is_buggy:
            return n
    return None


def classify(node, journal) -> str:
    """One verdict for a resolved node. Never raises."""
    if node.is_buggy:
        return BUGGY
    if node.val_primary is None:
        return NO_SCORE

    base = _baseline_node(journal)

    # Under-training is checked first: a model that did not fit cannot be judged on its
    # validation score at all, whatever that score happens to be. The bar is the baseline's
    # *validation* score -- scoring worse on your own training data than the reference scores on
    # data it never saw is not a close call. Comparing against the baseline's training score
    # instead flags ordinary well-fitted models; see the note in config.
    if (node.train_primary is not None and base is not None
            and base.val_primary is not None
            and node.train_primary < base.val_primary - config.UNDER_TRAINED_MARGIN):
        return UNDER_TRAINED

    if (node.train_primary is not None and base is not None
            and base.train_primary is not None and base.val_primary is not None):
        base_gap = base.train_primary - base.val_primary
        gap = node.train_primary - node.val_primary
        if base_gap > 0 and gap > config.OVERFIT_GAP_RATIO * base_gap:
            return OVERFIT

    parent = journal.get(node.parent_id) if node.parent_id is not None else None
    ref = parent if (parent is not None and parent.val_primary is not None) else base
    if ref is None or ref.val_primary is None:
        return NOISE
    delta = node.val_primary - ref.val_primary
    if abs(delta) < config.EPSILON:
        return NOISE
    return IMPROVEMENT if delta > 0 else REGRESSION


# Two experiments count as the same mechanism at this much tag overlap. Matches the duplicate
# gate's threshold in gates.py so the two agree about what "the same idea" means.
TAG_OVERLAP = 0.5


def _tags(node) -> set:
    spec = node.spec or {}
    return {str(t).lower().strip() for t in (spec.get('tags') or []) if str(t).strip()}


def _mechanism_key(node) -> str:
    return ', '.join(sorted(_tags(node)))


def _cluster(nodes) -> List[List]:
    """Group nodes whose tags overlap enough to be the same idea.

    Exact tag-set equality is too strict to be useful here: runs/v2's nodes 8, 11 and 16 were
    one mechanism -- a sparse per-(user, X) label rate -- tagged `user-category-affinity`,
    `user-author-affinity` and `user-tag-affinity`, so keying on the full set put each in its
    own group and the repeat went unreported. Greedy single-link clustering on Jaccard overlap
    catches them while keeping genuinely different experiments apart.
    """
    clusters: List[List] = []
    for n in nodes:
        tn = _tags(n)
        if not tn:
            continue
        for c in clusters:
            if any(len(tn & _tags(m)) / max(len(tn | _tags(m)), 1) >= TAG_OVERLAP for m in c):
                c.append(n)
                break
        else:
            clusters.append([n])
    return clusters


# The stages of the loop in the problem statement's Figure 1, and tag fragments that indicate
# an experiment touched one. Keyword matching on the planner's own tags -- coarse by design,
# and reported as "what your tags say you have touched", never as a list of things to try.
# Innovation is judged on covering the full stack, and runs/v3 spent every one of its twelve
# improve iterations inside just two of these six.
_STAGE_MARKERS = (
    ('data & sampling', ('sampling', 'negative-sampling', 'hard-negative', 'weighting',
                         'time-decay', 'recency', 'drift', 'augment', 'subsample',
                         'reweight', 'temporal')),
    ('feature engineering', ('feature', 'features', 'encoding', 'embedding-field', 'category',
                             'tag', 'content', 'item-features', 'statistics', 'affinity',
                             'cross', 'popularity')),
    ('model architecture', ('fm', 'lightgbm', 'xgboost', 'gbdt', 'tree', 'architecture',
                            'deep', 'attention', 'sequence', 'din', 'nn', 'two-stage',
                            'stacking')),
    ('training objective', ('loss', 'bpr', 'lambdarank', 'listwise', 'pairwise', 'pointwise',
                            'ranking-objective', 'objective', 'multi-task', 'auxiliary',
                            'listnet', 'softmax')),
    ('hyperparameters & schedule', ('lr', 'learning-rate', 'regulari', 'epoch', 'tuning',
                                    'training-intensity', 'dropout', 'patience', 'schedule')),
    ('evaluation & post-processing', ('ensemble', 'calibration', 'blend', 'post-process',
                                      'seed', 'rerank', 'aggregation')),
)


def stack_coverage(journal) -> str:
    """Which stages of the loop the run has exercised, derived from its own tags."""
    tags = set()
    for n in getattr(journal, 'nodes', []):
        if n.spec:
            tags |= {str(t).lower().strip() for t in (n.spec.get('tags') or [])}
    if not tags:
        return ''

    touched, untouched = [], []
    for stage, markers in _STAGE_MARKERS:
        hits = sorted({t for t in tags if any(m in t for m in markers)})
        (touched if hits else untouched).append((stage, hits))

    lines = ['# Which parts of the pipeline you have exercised',
             '',
             'Derived from the tags on your own prior experiments, not a suggestion list.',
             '']
    for stage, hits in touched:
        lines.append(f'- **{stage}** — {len(hits)} tag(s): {", ".join(hits[:6])}')
    if untouched:
        lines.append('')
        lines.append('No experiment so far has carried a tag matching: '
                     + ', '.join(f'**{s}**' for s, _ in untouched) + '.')
        lines.append('That is an observation about coverage, not a recommendation — a stage '
                     'may be genuinely irrelevant here, and you are the one judging that.')
    return '\n'.join(lines)


def ruled_out_block(journal, min_repeats: int = 2) -> str:
    """A prompt section naming what the evidence has settled and what it has not.

    Two distinct jobs. First, group repeated failures by mechanism so a third attempt at the same
    idea has to argue against the record instead of rediscovering it. Second, list the directions
    whose only trial was under-trained, because those look refuted in the score table while
    actually being untested -- exactly the trap that cost runs/v2 its ranking-loss experiment.
    """
    resolved = [n for n in getattr(journal, 'nodes', [])
                if not n.is_buggy and n.val_primary is not None and n.operation != 'baseline']
    if not resolved:
        return ''

    verdicts = {n.id: classify(n, journal) for n in resolved}

    lines: List[str] = []
    for nodes in _cluster(resolved):
        if len(nodes) < min_repeats:
            continue
        vs = {verdicts[n.id] for n in nodes}
        if vs <= {OVERFIT, REGRESSION, NOISE}:
            ids = ', '.join(f'node {n.id}' for n in nodes)
            # Tags shared by every node in the cluster name the mechanism better than any one
            # node's full tag list, which carries the join key that made them look distinct.
            shared = set.intersection(*(_tags(n) for n in nodes)) or _tags(nodes[0])
            worst = OVERFIT if OVERFIT in vs else sorted(vs)[0]
            lines.append(f'- `{", ".join(sorted(shared))}` -- tried {len(nodes)}x ({ids}), every '
                         f'attempt {"/".join(sorted(vs))}. {MEANING[worst]}. A further variation '
                         f'on this mechanism needs a reason the previous ones do not cover.')

    untested = [n for n in resolved if verdicts[n.id] == UNDER_TRAINED]
    for n in untested:
        key = _mechanism_key(n) or (n.spec or {}).get('proposed_change', '')[:60]
        lines.append(f'- `{key}` (node {n.id}) scored {n.val_primary:.4f} but its train_primary '
                     f'was {n.train_primary:.4f}, below the baseline\'s. {MEANING[UNDER_TRAINED]}. '
                     f'Its low score is NOT evidence against the idea.')

    if not lines:
        return ''
    return ('# What the evidence has and has not settled\n\n'
            'Derived mechanically from prior results, not opinion.\n\n' + '\n'.join(lines))


def _stated_effect(spec) -> Optional[float]:
    """The largest number in `expected_result` that reads as a metric delta.

    The planner writes these freehand -- "0.005-0.015", "+0.01 to 0.02", "roughly 0.003" -- so
    this takes the biggest plausible delta it can find and treats that as the claim.

    The upper bound is 0.05, not 0.1. Absolute scores ("bringing the score to 0.612") have to be
    excluded, and so do train/val gaps -- v5 node 8 wrote "gap narrows from 0.090 to ~0.05" and
    a 0.1 ceiling read that as a forecast of +0.09. 0.05 is still 25x the noise floor and far
    beyond any single change's plausible effect here, where the whole baseline-to-oracle
    headroom is about 0.25.
    """
    if not spec:
        return None
    text = str((spec or {}).get('expected_result', '') or '')
    vals = [float(m) for m in re.findall(r'\d*\.\d+', text)]
    vals = [v for v in vals if 0 < v < 0.05]
    return max(vals) if vals else None


def calibration_block(journal, min_rows: int = 3) -> str:
    """The planner's own effect-size predictions against what actually happened.

    Every prediction in runs/v5 was between +0.003 and +0.02; every measured delta was inside
    +/-0.006, and the accepted best moved +0.0003. The planner never saw that gap, so each
    iteration re-proposed a small change with a confident double-digit-basis-point forecast and
    the run converged at noise. Showing the record turns a systematic bias into something the
    planner can price in -- if its own last several forecasts were an order of magnitude high,
    an idea it forecasts at +0.003 is one it should expect to be indistinguishable from noise.
    """
    from . import config
    rows = []
    for n in getattr(journal, 'nodes', []):
        if n.is_buggy or n.val_primary is None or not n.spec:
            continue
        want = _stated_effect(n.spec)
        parent = journal.get(n.parent_id) if n.parent_id is not None else None
        base_val = parent.val_primary if parent and parent.val_primary is not None else None
        if want is None or base_val is None:
            continue
        rows.append((n.id, want, n.val_primary - base_val))
    if len(rows) < min_rows:
        return ''

    lines = ['# How accurate your effect-size forecasts have been',
             '',
             'Your own `expected_result` against the measured delta versus that node\'s parent.',
             '',
             '| node | you predicted | actually got |',
             '|---|---|---|']
    for nid, want, got in rows[-8:]:
        lines.append(f'| {nid} | +{want:.4f} | {got:+.4f} |')

    beat = sum(1 for _, _, got in rows if got > config.EPSILON)
    ratio = statistics.median([w / max(abs(g), 1e-4) for _, w, g in rows])
    lines += ['',
              f'{beat} of {len(rows)} predictions cleared the {config.EPSILON} noise floor. '
              f'Your forecasts have run about {ratio:.0f}x the size of the effects you measured.',
              '',
              'Price that in. An experiment you honestly forecast at under '
              f'{config.EPSILON} is one whose result you will not be able to distinguish from '
              'noise, and it costs the same iteration as one that could settle something. This '
              'is a statement about the size of your forecasts, not a suggestion about which '
              'direction to take.']
    return '\n'.join(lines)
