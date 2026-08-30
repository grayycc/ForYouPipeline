"""Convergence must be driven by experiments that actually scored.

The bug this guards against: best_history() used to append an entry for every node, so three
consecutive crashes after a successful baseline produced a flat window and ended the run at
iteration 3 with 46 iterations unused.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.journal import Journal, Node

EPS, N = 0.002, 3


def scoring(i, primary, parent=0, op='improve', accepted=True):
    """A node that ran and scored.

    `accepted` defaults True because best_history tracks the accepted best: in the real loop a
    node that raises the best-so-far is accepted by definition, and marking a non-improving one
    accepted changes nothing, since the history is a running max. Rejection is exercised
    explicitly in tests/test_diagnose.py.
    """
    n = Node(id=i, parent_id=parent, operation=op)
    n.is_buggy, n.val_primary, n.accepted = False, primary, accepted
    return n


def crashed(i, parent=0):
    return Node(id=i, parent_id=parent, operation='improve')   # is_buggy defaults True


def eda(i):
    n = Node(id=i, parent_id=0, operation='eda')
    n.is_buggy = False          # ran fine, just produced no score
    return n


def build(nodes):
    j = Journal()
    for n in nodes:
        j.append(n)
    return j


def test_crashes_do_not_converge():
    j = build([scoring(0, 0.6014, None, 'baseline')] + [crashed(i) for i in (1, 2, 3)])
    assert not j.has_converged(EPS, N), 'three crashes must not look like convergence'


def test_eda_does_not_converge():
    j = build([scoring(0, 0.6014, None, 'baseline'), eda(1),
               scoring(2, 0.6020), scoring(3, 0.6100)])
    assert not j.has_converged(EPS, N), 'EDA must not count toward the window'


def test_genuine_stagnation_converges():
    j = build([scoring(0, 0.6014, None, 'baseline'),
               scoring(1, 0.6015), scoring(2, 0.6016), scoring(3, 0.6018)])
    assert j.has_converged(EPS, N), 'four flat scoring nodes are real convergence'


def test_real_improvement_does_not_converge():
    j = build([scoring(0, 0.6014, None, 'baseline'),
               scoring(1, 0.6100), scoring(2, 0.6200), scoring(3, 0.6300)])
    assert not j.has_converged(EPS, N)


def test_too_few_nodes():
    assert not build([scoring(0, 0.6014, None, 'baseline')]).has_converged(EPS, N)


def test_crashes_interleaved_with_progress():
    """Crashes between improving experiments must not drag the window flat."""
    j = build([scoring(0, 0.6014, None, 'baseline'), crashed(1),
               scoring(2, 0.6100), crashed(3), scoring(4, 0.6200)])
    assert not j.has_converged(EPS, N)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'  PASS  {name}')
    print('\nall journal tests passed')


def test_floor_blocks_convergence_during_warm_up():
    """The same four flat nodes that are genuine stagnation late in a run are just the
    ordinary early state of a search. The floor is what separates the two cases, and it was
    ending every real run at iteration 4 of 50 before it existed."""
    j = build([scoring(0, 0.6014, None, 'baseline'),
               scoring(1, 0.6015), scoring(2, 0.6016), scoring(3, 0.6018)])
    assert j.has_converged(EPS, N), 'unchanged with no floor'
    assert not j.has_converged(EPS, N, min_scoring_nodes=15), 'floor must suppress it'


def test_floor_still_allows_convergence_once_met():
    nodes = [scoring(0, 0.6014, None, 'baseline')]
    nodes += [scoring(i, 0.6015) for i in range(1, 16)]
    j = build(nodes)
    assert j.has_converged(EPS, N, min_scoring_nodes=15), 'floor cleared, rule applies'
