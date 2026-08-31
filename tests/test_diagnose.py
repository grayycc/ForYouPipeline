"""Failure diagnosis, and the accept-bar it feeds.

Every number below is taken from runs/v2, which is what these fixes were written against:
the run spent ~19% of its budget on three attempts at one mechanism, wrote off its
ranking-loss experiment on a model that never trained, and let a gate-rejected node set the
bar for ten iterations.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import diagnose
from agent.journal import Journal, Node

EPS, N = 0.002, 3


def node(i, *, train=None, val=None, parent=0, op='improve', accepted=False,
         buggy=False, tags=(), unbiased=None):
    n = Node(id=i, parent_id=parent, operation=op)
    n.is_buggy = buggy
    n.train_primary, n.val_primary = train, val
    n.unbiased_val_primary = unbiased
    n.accepted = accepted
    if tags:
        n.spec = {'tags': list(tags), 'proposed_change': f'change {i}'}
    return n


def build(nodes):
    j = Journal()
    for n in nodes:
        j.append(n)
    return j


# runs/v2 node 0: the reproduced FM baseline everything is measured against.
BASE = lambda: node(0, train=0.6918, val=0.6015, parent=None, op='baseline', accepted=True,
                    unbiased=0.3638)


def test_bpr_that_never_trained_is_not_a_refutation():
    """runs/v2 node 3: val 0.5546 looks like a decisive negative, but train_primary 0.5614 is
    below the baseline's own 0.6918 -- the model never fit, so the idea was never tested."""
    j = build([BASE(), node(3, train=0.5614, val=0.5546, tags=['bpr', 'ranking-objective'])])
    assert diagnose.classify(j.get(3), j) == diagnose.UNDER_TRAINED


def test_sparse_user_cross_is_diagnosed_as_overfit():
    """runs/v2 node 11: train 0.9727 vs val 0.5854, a gap of 0.387 against the baseline's 0.090."""
    j = build([BASE(), node(11, train=0.9727, val=0.5854, tags=['user-author-affinity'])])
    assert diagnose.classify(j.get(11), j) == diagnose.OVERFIT


def test_small_move_is_noise_not_a_result():
    """runs/v2 node 9: 0.60177 against the baseline is well inside the 0.002 floor."""
    j = build([BASE(), node(9, train=0.6863, val=0.60177, tags=['category-features'])])
    assert diagnose.classify(j.get(9), j) == diagnose.NOISE


def test_real_gain_is_an_improvement():
    j = build([BASE(), node(9, train=0.6900, val=0.6100, tags=['x'])])
    assert diagnose.classify(j.get(9), j) == diagnose.IMPROVEMENT


def test_repeated_mechanism_is_named_in_the_ruled_out_block():
    """Nodes 8/11/16 were one mechanism with three join keys. The block must group them so a
    fourth attempt has to argue against the record instead of rediscovering it."""
    j = build([BASE(),
               node(8, train=0.8058, val=0.5698, tags=['personalization', 'affinity']),
               node(11, train=0.9727, val=0.5854, tags=['personalization', 'affinity']),
               node(16, train=0.8184, val=0.5693, tags=['personalization', 'affinity'])])
    block = diagnose.ruled_out_block(j)
    assert 'tried 3x' in block
    assert 'node 8' in block and 'node 11' in block and 'node 16' in block


def test_under_trained_node_is_flagged_as_untested_in_the_block():
    j = build([BASE(), node(3, train=0.5614, val=0.5546, tags=['bpr'])])
    block = diagnose.ruled_out_block(j)
    assert 'NOT evidence against the idea' in block


def test_gate_rejected_node_never_becomes_best():
    """runs/v2 node 6 was rejected by the unbiased gate yet became journal.best, raising the bar
    for ten iterations and serving as their branch point."""
    j = build([BASE(), node(6, train=0.6863, val=0.6019, accepted=False, unbiased=0.3619)])
    assert j.best is not None and j.best.id == 0, 'a rejected node must not become best'


def test_best_history_ignores_rejected_scores():
    """A rejected node's score is not progress and must not be able to fire convergence."""
    j = build([BASE()] + [node(i, train=0.68, val=0.6019, accepted=False) for i in (1, 2, 3)])
    assert j.best_history() == [0.6015, 0.6015, 0.6015, 0.6015]
    assert j.has_converged(EPS, N), 'three rejected nodes are a genuine plateau'


def test_a_well_fitted_node_is_not_called_under_trained():
    """Regression: anchoring `under_trained` a fixed 0.02 below the baseline's *training* score
    flagged runs/v2 node 15 (train 0.6631) -- the best node in the whole run -- and node 10
    (train 0.6699). The bar is the baseline's validation score, which neither falls below."""
    j = build([BASE(),
               node(10, train=0.6699, val=0.6003, tags=['click-signal']),
               node(15, train=0.6631, val=0.6022, accepted=True, tags=['author-quality'])])
    assert diagnose.classify(j.get(10), j) != diagnose.UNDER_TRAINED
    assert diagnose.classify(j.get(15), j) != diagnose.UNDER_TRAINED


def test_mechanism_clustering_survives_a_changed_join_key():
    """The real runs/v2 tags differed by join key, so exact-set grouping reported nothing."""
    j = build([BASE(),
               node(8, train=0.8058, val=0.5698,
                    tags=['user-category-affinity', 'personalization', 'feature-engineering', 'FM']),
               node(11, train=0.9727, val=0.5854,
                    tags=['user-author-affinity', 'personalization', 'feature-engineering', 'FM']),
               node(16, train=0.8184, val=0.5693,
                    tags=['user-tag-affinity', 'personalization', 'feature-engineering', 'FM'])])
    block = diagnose.ruled_out_block(j)
    assert 'tried 3x' in block and 'node 8' in block and 'node 16' in block


# --- the unbiased gate, referenced to the parent rather than the global best ---

def test_the_higher_validation_score_wins_even_on_a_worse_unbiased_score():
    """The challenge scores "the validation-best checkpoint", so nothing else may outrank it.

    runs/v5 banded validation into EPSILON-wide bins and broke ties on the random-exposure
    score. Band 301 spans 0.6010-0.6030, which was the entire range of every good node the run
    produced, so the tie-break became the only thing that mattered -- and it points the wrong
    way, r(val, unbiased) = -0.33, because the hidden test set is a logged-exposure split like
    validation. Nodes 12, 13, 15 and 18 all beat the incumbent on validation and all four were
    rejected on it. These are node 11 and node 18's real numbers.
    """
    j = build([node(11, val=0.60187, parent=None, op='baseline',
                    accepted=True, unbiased=0.36530)])
    eighteen = node(18, val=0.60260, parent=11, unbiased=0.35987)
    eighteen.accepted = True
    j.append(eighteen)
    assert j.best.id == 18, 'the validation-best node must take the incumbency'


def test_tie_break_cannot_ratchet_validation_downwards():
    """A node more than one epsilon band below the incumbent must never take the incumbency,
    however good its unbiased score."""
    j = build([node(1, val=0.6013, parent=None, op='baseline', accepted=True, unbiased=0.3059)])
    low = node(2, val=0.5900, parent=1, unbiased=0.9000)
    low.accepted = True
    j.append(low)
    assert j.best.id == 1


def test_a_failed_one_shot_op_is_not_debugged_once_it_has_succeeded():
    """runs/v3 spent iteration 12 debugging the baseline that crashed at iteration 0, eleven
    iterations after iteration 1 had already produced a working one."""
    j = build([node(0, parent=None, op='baseline', buggy=True),
               node(1, val=0.6013, parent=None, op='baseline', accepted=True)])
    assert [n.id for n in j.buggy_leaves()] == [], 'baseline is settled; nothing to debug'


def test_a_failed_draft_is_still_worth_debugging():
    """Drafts are exploration, not a one-shot job, so other drafts succeeding does not settle
    a crashed one."""
    j = build([node(1, val=0.6013, parent=None, op='baseline', accepted=True),
               node(3, val=0.5988, parent=None, op='draft'),
               node(5, parent=None, op='draft', buggy=True)])
    assert [n.id for n in j.buggy_leaves()] == [5]


def test_accepted_always_means_new_incumbent():
    """accept and rank must be the same question, or _promote overwrites the best submission
    on disk with a node that is not actually best. runs/v3 node 16 (0.6028, +0.011 unbiased)
    against node 15 (0.6035) is that case: a different epsilon band, so it must not be
    accepted however good its unbiased score."""
    j = build([node(1, val=0.6013, parent=None, op='baseline', accepted=True, unbiased=0.3059)])
    fifteen = node(15, val=0.6035, parent=1, unbiased=0.3120); fifteen.accepted = True
    j.append(fifteen)
    sixteen = node(16, val=0.6028, parent=15, unbiased=0.3234)
    assert not j.outranks_best(sixteen), 'a full band below best must never take incumbency'


# --- escaping a plateau ---

def _agent_with(nodes):
    import random
    from agent.orchestrator import Agent
    a = Agent.__new__(Agent)
    a.journal = build(nodes)
    a.rng = random.Random(0)
    return a


def test_four_noise_results_trigger_a_draft():
    """runs/v3 nodes 8-11 were four consecutive noise results; the search kept editing the
    incumbent for five more iterations instead of starting fresh."""
    ns = [node(0, train=0.6918, val=0.6015, parent=None, op='baseline', accepted=True)]
    ns += [node(i, parent=None, op='draft', val=0.59) for i in (1, 2, 3)]
    ns += [node(i, train=0.65, val=0.6016, parent=0) for i in (4, 5, 6, 7)]
    assert _agent_with(ns)._stalled()


def test_an_informative_failure_breaks_the_streak():
    """overfit and regression point somewhere; only a result indistinguishable from the
    incumbent says the neighbourhood is exhausted."""
    ns = [node(0, train=0.6918, val=0.6015, parent=None, op='baseline', accepted=True)]
    ns += [node(i, parent=None, op='draft', val=0.59) for i in (1, 2, 3)]
    ns += [node(i, train=0.65, val=0.6016, parent=0) for i in (4, 5, 6)]
    ns += [node(7, train=0.97, val=0.5854, parent=0)]          # overfit
    assert not _agent_with(ns)._stalled()


def test_a_recent_draft_suppresses_another():
    """The cooldown stops a long plateau spending the whole budget on drafts."""
    ns = [node(0, train=0.6918, val=0.6015, parent=None, op='baseline', accepted=True)]
    ns += [node(i, parent=None, op='draft', val=0.59) for i in (1, 2, 3)]
    ns += [node(i, train=0.65, val=0.6016, parent=0) for i in (4, 5, 6, 7)]
    ns += [node(8, parent=None, op='draft', val=0.59)]
    assert not _agent_with(ns)._stalled()


def test_stack_coverage_names_untouched_stages_only():
    j = build([node(5, val=0.6029, tags=['bpr', 'pairwise-loss', 'FM']),
               node(8, val=0.6029, tags=['content-features', 'video-category', 'FM'])])
    out = diagnose.stack_coverage(j)
    assert 'training objective' in out and 'feature engineering' in out
    assert 'data & sampling' in out.split('No experiment so far')[1]
    assert 'recommend' not in out.lower().replace('not a recommendation', '')


# --- the v5 fixes -----------------------------------------------------------------------

def test_a_draft_is_visible_to_the_duplicate_gate_in_both_directions():
    """runs/v5 ran 'LightGBM over marginal CTR rates' seven times -- nodes 2, 3, 4, 6, 7, 9, 14,
    every one a regression, 37% of the run. Drafts carried no tags, and the gate skipped any
    node whose tags were empty both when checking and when scanning prior work, so four of
    those seven were drafts that neither saw the others nor were seen."""
    from agent import gates
    from agent.spec import ExperimentSpec

    prior = node(2, val=0.5962, parent=None, op='draft')
    prior.spec = {'tags': ['gbdt', 'marginal-ctr-features'], 'proposed_change': 'LightGBM on CTR'}
    j = build([BASE(), prior])

    same = ExperimentSpec(tags=['gbdt', 'marginal-ctr-features', 'pointwise-loss'],
                          proposed_change='Fix the free_raw_data bug and retrain')
    assert gates._find_duplicate(same, j) is prior, 'rewording must not defeat the gate'

    other = ExperimentSpec(tags=['listwise-loss', 'listnet', 'ranking-objective'],
                           proposed_change='softmax cross-entropy over each user list')
    assert gates._find_duplicate(other, j) is None, 'a genuinely different mechanism must pass'


def test_unbiased_tolerance_pools_every_seeded_node_not_just_the_baseline():
    """A 3-seed estimate of sigma is far too noisy to gate on. The baseline drew a narrow
    triple (sigma 0.00134) while runs/v3+v5 pooled over 18 nodes give 0.00215, so the gate ran
    60% tighter than the noise and vetoed v5 node 18 on a 0.0035 drop against a 0.0034 bar."""
    import statistics
    from agent import config

    baseline_only = statistics.pstdev([0.3638, 0.3667, 0.3639])
    groups = [[0.3638, 0.3667, 0.3639], [0.3605, 0.3594, 0.3578], [0.3628, 0.361, 0.3621],
              [0.364, 0.3668, 0.3651], [0.3632, 0.3658, 0.3631], [0.3616, 0.355, 0.3646],
              [0.3628, 0.3629, 0.3644], [0.3592, 0.3609, 0.3595]]
    dof = sum(len(g) - 1 for g in groups)
    ss = sum((x - statistics.mean(g)) ** 2 for g in groups for x in g)
    pooled = (ss / dof) ** 0.5

    assert pooled > baseline_only
    assert config.UNBIASED_TOLERANCE_SIGMAS * pooled > 0.0035, (
        'the pooled tolerance must clear the 0.0035 drop that wrongly vetoed node 18')


def test_calibration_block_reports_the_forecast_bias():
    """v5 predicted +0.003..+0.02 every iteration and measured nothing outside +/-0.006, and
    the planner was never shown the gap."""
    ns = [BASE()]
    for i, (want, val) in enumerate([('0.005-0.015', 0.5984), ('+0.008', 0.6002),
                                     ('0.003-0.010', 0.6019)], start=1):
        n = node(i, val=val, parent=0)
        n.spec = {'tags': [f't{i}'], 'expected_result': want}
        ns.append(n)
    out = diagnose.calibration_block(build(ns))
    assert 'forecasts have run about' in out
    assert '0 of 3 predictions cleared' in out, out


def test_a_train_val_gap_is_not_read_as_a_forecast_delta():
    """v5 node 8 wrote 'gap narrows from 0.090 to ~0.05'; a 0.1 ceiling read that as +0.09."""
    assert diagnose._stated_effect({'expected_result': 'gap narrows from 0.090 to ~0.05'}) is None
    assert diagnose._stated_effect({'expected_result': 'a gain of 0.003-0.008'}) == 0.008


def test_a_leak_whose_reason_denies_leakage_is_not_a_leak():
    """v5 node 2 was blocked on a reason that itself read 'This is not strictly label leakage
    in the time dimension but rather a data mismatch'."""
    from agent.roles import reviewer
    assert reviewer._concedes_not_leakage(
        'This is not strictly label leakage in the time dimension but rather a data mismatch.')
    # hedging alone must still block -- real leaks are often reported with "may"/"could"
    assert not reviewer._concedes_not_leakage(
        'Lines 46-48 may aggregate splits["valid"] labels into a feature used to score valid.')


# --- closed mechanisms: binding, not advisory ------------------------------------------

def _v6_journal():
    """runs/v6's real log, which is what the closed-mechanism rule was measured against."""
    import json, os
    from agent.journal import Journal, Node
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'runs', 'v6', 'log.jsonl')
    j = Journal()
    for line in open(path):
        d = json.loads(line)
        n = Node(id=d['iter'], parent_id=d.get('parent_id'), operation=d['operation'])
        n.is_buggy = d.get('is_buggy', False)
        n.val_primary = d.get('val_primary')
        n.train_primary = d.get('train_primary')
        n.accepted = d.get('accepted', False)
        if d.get('tags'):
            n.spec = {'tags': d['tags'], 'proposed_change': d.get('proposed_change', '')}
        j.append(n)
    return j


def test_only_the_tree_family_is_closed_on_v6():
    """7 attempts, 0 accepted, every one regression/overfit -- 23% of the run. Everything else
    must stay open, and for reasons that are not "it happened to be under the threshold"."""
    j = _v6_journal()
    closed = dict(diagnose.exhausted_mechanisms(j))
    assert list(closed) == ['gradient-boosted trees'], closed
    assert len(closed['gradient-boosted trees']) == 7


def test_a_family_that_ever_produced_an_accept_is_never_closed():
    """FM feature work was v6's largest family and produced 3 of its 7 improvements; ensembling
    produced 3 more. A rule that closed those would remove everything that ever worked."""
    j = _v6_journal()
    closed = dict(diagnose.exhausted_mechanisms(j))
    assert 'factorization machine' not in closed
    assert 'ensembling' not in closed


def test_a_family_with_a_noise_verdict_stays_open():
    """v6's ranking-objective family is 0-for-6, but one attempt landed in the noise band.
    `noise` means the result could not be resolved, which is not evidence the idea is wrong."""
    j = _v6_journal()
    assert 'ranking objective' not in dict(diagnose.exhausted_mechanisms(j))


def test_closed_mechanism_is_a_hard_gate_reason():
    from agent import gates
    from agent.spec import ExperimentSpec
    j = _v6_journal()
    spec = ExperimentSpec(
        hypothesis='h' * 40, mechanism='m' * 40, proposed_change='p' * 40,
        expected_result='0.004', falsification_condition='f' * 40, target_metric='both',
        risks={'leakage': 'none', 'overfitting': 'x', 'runtime': 'y'},
        implementation_scope=['solution.py'],
        tags=['xgboost', 'rank-ndcg', 'pairwise-ranking-objective'],
        evidence={'task_structure': 'a' * 30, 'previous_experiments': 'b' * 30, 'literature': ''})
    ok, reasons = gates.check_spec(spec, j)
    assert not ok
    assert any('is closed:' in r for r in reasons), reasons

    spec.tags = ['user-history', 'sequence-features', 'within-user-ranking']
    _, reasons2 = gates.check_spec(spec, j)
    assert not any('is closed:' in r for r in reasons2), reasons2


def test_tree_family_matches_however_the_planner_spells_it():
    """The v8 smoke run tagged a LightGBM ranker `lgbm-ranker`, which does not contain the
    string `lightgbm`, so it escaped the family and was counted as a ranking-objective
    experiment only. Every spelling the planner has actually produced must land in the family,
    or the closed-mechanism count silently under-reports."""
    for tag in ('gbdt', 'lightgbm', 'lgbm-ranker', 'xgboost', 'tree-model', 'catboost',
                'gradient-boosted-trees'):
        assert 'gradient-boosted trees' in diagnose.families_of([tag]), tag
    # and the families it must not swallow
    for tag in ('fm', 'listnet', 'ensemble', 'user-history'):
        assert 'gradient-boosted trees' not in diagnose.families_of([tag]), tag
