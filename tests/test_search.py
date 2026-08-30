"""The search policy in Agent.select(), offline -- no model calls, no training.

Covers the phase order (baseline -> EDA -> drafts -> improve) and the two properties that make
a draft a draft: it has no parent, and the planner prompt it produces does not carry the
current best solution's source.
"""
import os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config
from agent.journal import Journal, Node
from agent.orchestrator import Agent
from agent.roles import planner

RUN_ID = '_test_search'


def _agent():
    """An agent with an empty journal, for exercising select() alone."""
    a = Agent(RUN_ID)
    a.journal = Journal()
    return a


def _scored(nid, op, score, parent=None, accepted=False):
    """A node that ran and scored, optionally accepted."""
    return Node(id=nid, parent_id=parent, operation=op, is_buggy=False, val_primary=score,
                accepted=accepted, code=f'# solution {nid}\n', hypothesis=f'h{nid}')


def test_phase_order():
    """baseline, then EDA, then drafting."""
    a = _agent()
    assert a.select()[0] == 'baseline'

    a.journal.append(_scored(0, 'baseline', 0.6024, accepted=True))
    assert a.select()[0] == 'eda'

    a.journal.append(Node(id=1, parent_id=0, operation='eda', is_buggy=False))
    op, parent = a.select()
    assert op == 'draft', op
    assert parent is None, 'a draft must have no parent'


def test_drafting_runs_until_min_drafts_then_goes_greedy():
    """Drafting stops at MIN_DRAFTS, counting the baseline as the first."""
    a = _agent()
    a.journal.append(_scored(0, 'baseline', 0.6024, accepted=True))
    a.journal.append(Node(id=1, parent_id=0, operation='eda', is_buggy=False))

    # the baseline is the first draft, so MIN_DRAFTS - 1 more are expected
    expected = config.MIN_DRAFTS - 1
    for i in range(expected):
        op, parent = a.select()
        assert op == 'draft', f'draft {i}: got {op}'
        assert parent is None
        a.journal.append(_scored(2 + i, 'draft', 0.60 + i / 1000))

    op, parent = a.select()
    assert op == 'improve', op
    assert parent is not None and parent.id == 0, 'improve branches from the accepted best'


def test_implausible_is_flagged_before_seeds_are_spent():
    """A node under the popularity rung is broken code. Confirming it across three seeds costs
    two more full training runs to measure a number already known to be wrong -- in one run
    that was two 20-epoch passes over 2.28M pairs."""
    import inspect
    from agent import orchestrator
    src = inspect.getsource(orchestrator.Agent.step)
    flag = src.index('_flag_implausible')
    confirm = src.index('_confirm_seeds')
    assert flag < confirm, 'seeds are confirmed before the node is checked for being broken'


def test_flag_implausible_marks_and_spares():
    """Only scored nodes below the rung are flagged."""
    a = _agent()
    below = _scored(9, 'draft', config.SANITY_FLOOR - 0.01)
    a._flag_implausible(below)
    assert below.is_buggy and 'below the item-popularity rung' in below.buggy_reason

    above = _scored(10, 'draft', config.SANITY_FLOOR + 0.01)
    a._flag_implausible(above)
    assert not above.is_buggy

    unscored = Node(id=11, parent_id=None, operation='eda', is_buggy=False)
    a._flag_implausible(unscored)
    assert not unscored.is_buggy, 'a node with no score must not be flagged'


def test_improve_branches_from_best_not_last():
    """Greedy search branches from the best node, not the most recent."""
    a = _agent()
    a.journal.append(_scored(0, 'baseline', 0.6024, accepted=True))
    a.journal.append(Node(id=1, parent_id=0, operation='eda', is_buggy=False))
    a.journal.append(_scored(2, 'draft', 0.6038, accepted=True))
    a.journal.append(_scored(3, 'draft', 0.5900))          # worse, and not accepted

    op, parent = a.select()
    assert op == 'improve' and parent.id == 2, (op, parent.id if parent else None)


def test_draft_prompt_withholds_the_best_solution():
    """A draft that is shown the winning file will return a variation on it."""
    j = Journal()
    j.append(_scored(0, 'baseline', 0.6024, accepted=True))
    j.eda_findings = 'VALID rows/user 5.4'

    drafting = planner._build_prompt(2, j, drafting=True)
    improving = planner._build_prompt(2, j, drafting=False)

    assert '# solution 0' in improving, 'improve must show the code being changed'
    assert '# solution 0' not in drafting, 'draft must not show the best solution'
    assert 'Current best solution' not in drafting
    assert 'independent' in drafting.lower()
    # shared context both phases need
    for p in (drafting, improving):
        assert 'Prior attempts' in p and 'VALID rows/user' in p


def test_draft_still_sees_what_was_already_tried():
    """Independence is about the code, not amnesia -- a draft must not repeat a tried idea."""
    j = Journal()
    j.append(_scored(0, 'baseline', 0.6024, accepted=True))
    n = _scored(2, 'draft', 0.6038, accepted=True)
    n.spec = {'proposed_change': 'BPR pairwise loss', 'tags': ['bpr', 'ranking-loss']}
    j.append(n)

    p = planner._build_prompt(3, j, drafting=True)
    assert 'BPR pairwise loss' in p, 'the summary table must reach a drafting planner'


if __name__ == '__main__':
    try:
        fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
        for name, fn in fns:
            fn()
            print(f'  PASS  {name}')
        print(f'\nall {len(fns)} search tests passed')
    finally:
        shutil.rmtree(os.path.join(config.RUNS_DIR, RUN_ID), ignore_errors=True)
