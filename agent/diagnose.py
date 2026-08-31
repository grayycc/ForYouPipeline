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
import json
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
    """Tags with the duplicate gate's synonym map applied, but *not* its token expansion.

    The gate also splits each hyphenated tag into its tokens, which is right for asking "is this
    one spec the same as that one node" and wrong here. Clustering is single-link, so the
    generic tokens that expansion produces -- `features`, `objective`, `user`, `loss` -- act as
    bridges: on runs/v6 they chained nineteen nodes spanning four unrelated mechanisms into one
    cluster. Synonyms alone still fold `lightgbm`/`xgboost` into `gbdt`, which is the part that
    matters, without the bridging.
    """
    from . import gates
    spec = node.spec or {}
    return {gates._TAG_SYNONYMS.get(t, t)
            for t in (str(x).lower().strip() for x in (spec.get('tags') or []))
            if t}


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


# Mechanism families, matched on substrings of the planner's own tags. A binding gate has to be
# deterministic and auditable, so the axis is declared here rather than inferred from tag
# similarity. Tag-overlap clustering was tried for this and does not survive contact with real
# tags: on runs/v6 no threshold grouped the seven tree-model attempts without also merging
# unrelated mechanisms, because the discriminating token is one word inside tag sets that are
# otherwise unlike each other. A node may belong to several families; it counts in each.
_FAMILY_MARKERS = (
    # `gbm` rather than `lightgbm`: the v8 smoke run tagged a LightGBM ranker `lgbm-ranker`,
    # which does not contain `lightgbm` and so escaped this family entirely. `gbm` is a
    # substring of lightgbm, lgbm-ranker and gbm alike.
    ('gradient-boosted trees', ('gbdt', 'gbm', 'xgboost', 'catboost', 'boosted', 'tree',
                                'forest')),
    ('factorization machine', ('fm', 'factorization')),
    ('neural architecture', ('neural', 'mlp', 'deep', 'attention', 'din', 'transformer')),
    ('ranking objective', ('bpr', 'pairwise', 'listwise', 'listnet', 'lambdarank', 'softmax',
                           'ranking-objective', 'ndcg-optimization')),
    ('ensembling', ('ensemble', 'blend', 'rank-aggregation', 'variance-reduction')),
    ('item-side statistics', ('item-ctr', 'video-ctr', 'author-ctr', 'content-features',
                              'video-metadata', 'video-categories', 'video-popularity')),
    ('user-side interaction features', ('affinity', 'user-conditional', 'user-history',
                                        'sequence', 'user-tab', 'personalized')),
)


def families_of(tags) -> set:
    """Which mechanism families a raw tag list belongs to. One definition, used by both the
    closed-mechanism report here and the gate that enforces it, so the two cannot drift."""
    from . import gates
    norm = {gates._TAG_SYNONYMS.get(t, t)
            for t in (str(x).lower().strip() for x in (tags or [])) if t}
    return {name for name, markers in _FAMILY_MARKERS
            if any(m in t for t in norm for m in markers)}


def _families(node) -> set:
    return families_of((node.spec or {}).get('tags') or [])


def exhausted_mechanisms(journal, min_attempts: int = 3):
    """Families the evidence has actually closed, as [(family, node_ids)].

    Three conditions, all required:

    - at least `min_attempts` scoring nodes in the family,
    - **not one of them accepted**, and
    - **not one diagnosed `noise` or `improvement`.**

    The accepted-count condition is what keeps this honest. runs/v6's largest family was FM
    feature work, and blocking it would have removed three of the run's seven improvements. The
    noise condition matters just as much: `noise` means the result could not be resolved against
    the floor, which is not evidence the mechanism is wrong -- v6's ranking-objective family is
    0-for-6 but one attempt was noise, so it stays open.

    What the rule does close on v6's log is gradient-boosted trees: nodes 2, 3, 4, 11, 18, 23,
    30 -- seven attempts, zero accepted, every one `regression` or `overfit`. That is 23% of the
    run spent re-refuting a settled question, and it is the case this exists for.
    """
    resolved = [n for n in getattr(journal, 'nodes', [])
                if not n.is_buggy and n.val_primary is not None and n.operation != 'baseline']
    verdicts = {n.id: classify(n, journal) for n in resolved}

    out = []
    for family, _ in _FAMILY_MARKERS:
        members = [n for n in resolved if family in _families(n)]
        if len(members) < min_attempts:
            continue
        if any(n.accepted for n in members):
            continue
        vs = {verdicts[n.id] for n in members}
        if vs & {NOISE, IMPROVEMENT}:
            continue
        out.append((family, [n.id for n in members]))
    return out


def exhausted_block(journal) -> str:
    """Prompt section naming families that may not be proposed again."""
    dead = exhausted_mechanisms(journal)
    if not dead:
        return ''
    lines = ['# Mechanisms this run has closed',
             '',
             'Each was tried at least three times, was never once accepted, and never once even '
             'landed inside the noise band -- every single attempt came back measurably worse '
             'than what it branched from. These are settled. A further variation will be '
             'rejected rather than run, so proposing one costs the iteration and returns '
             'nothing.',
             '']
    for family, ids in dead:
        lines.append(f'- **{family}** -- {len(ids)} attempts '
                     f'({", ".join(f"node {i}" for i in ids)}), all worse than their parent.')
    lines.append('')
    lines.append('Spend this iteration on a mechanism that is still open.')
    return '\n'.join(lines)


def cross_run_yield(runs_dir='runs', exclude_run=None, min_attempts=3):
    """What each mechanism family has actually returned, pooled over every prior run.

    `exhausted_mechanisms` and `ruled_out_block` both read one journal, so everything a run
    learns dies with it. Measured over v2-v9: gradient-boosted trees is **0 accepted from 15
    attempts spread across 7 separate runs**, and every one of those runs spent an iteration
    rediscovering it -- in v9, which converged after 5 iterations, that was 20% of the budget.
    Meanwhile 73% of all attempts went to families whose median delta is at or below +0.0001,
    and the one family whose median clears the noise floor (ensembling, +0.00187 median, 44%
    accepted) was tried 9 times against factorization-machine tweaks' 52.

    That is an allocation problem, not a knowledge problem, and it is fixed by showing the
    planner the table rather than by forbidding anything. Deliberately reported, never enforced:
    the task description has changed repeatedly across these runs, so a family that failed under
    an older prompt has not necessarily been refuted under the current one. Cross-run closure
    would freeze in conclusions drawn against instructions that no longer exist.

    Nodes the reviewer flagged LEAK are excluded -- runs/v7 node 3 read the label through
    `play_time_ms` and scored 0.8482, which would otherwise dominate every statistic it touches.
    """
    import glob
    import os
    import statistics as _st

    by_family = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, '*', 'log.jsonl'))):
        run = os.path.basename(os.path.dirname(path))
        if run.startswith('smoke') or run == exclude_run:
            continue
        try:
            lines = open(path).read().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not d.get('val_primary') or not d.get('tags'):
                continue
            if d.get('review_verdict') == 'LEAK':
                continue
            for fam in families_of(d['tags']) or ['(other)']:
                rec = by_family.setdefault(fam, {'deltas': [], 'accepted': 0, 'runs': set()})
                rec['deltas'].append(d.get('delta_vs_baseline') or 0.0)
                rec['accepted'] += 1 if d.get('accepted') else 0
                rec['runs'].add(run)

    out = []
    for fam, rec in by_family.items():
        n = len(rec['deltas'])
        if n < min_attempts:
            continue
        out.append({'family': fam, 'n': n, 'accepted': rec['accepted'],
                    'median': _st.median(rec['deltas']), 'best': max(rec['deltas']),
                    'runs': len(rec['runs'])})
    out.sort(key=lambda r: -r['median'])
    return out


def cross_run_block(runs_dir='runs', exclude_run=None) -> str:
    """Prompt section: the standing record of each mechanism family across all prior runs."""
    from . import config
    rows = cross_run_yield(runs_dir, exclude_run)
    if len(rows) < 2:
        return ''
    lines = ['# What each kind of change has actually returned, across every prior run',
             '',
             'Pooled over previous runs of this agent on this split, so it carries evidence '
             'this run has not paid for. Delta is against the official baseline; the noise '
             f'floor is {config.EPSILON}.',
             '',
             '| mechanism | attempts | accepted | median delta | best delta | seen in N runs |',
             '|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['family']} | {r['n']} | {r['accepted']} "
                     f"({100 * r['accepted'] // r['n']}%) | {r['median']:+.5f} | "
                     f"{r['best']:+.5f} | {r['runs']} |")
    lines += ['',
              'This is a record, not a rule. The task description has changed between these '
              'runs, so a family that did badly under older instructions has not necessarily '
              'been refuted under the current ones — but a family with many attempts across '
              'many runs and no accepts is asking you for a reason why this attempt differs.']
    return '\n'.join(lines)
