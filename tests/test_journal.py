"""Convergence must be driven by experiments that actually scored.

The bug this guards against: best_history() used to append an entry for every node, so three
consecutive crashes after a successful baseline produced a flat window and ended the run at
iteration 3 with 46 iterations unused.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.journal import Journal, Node

EPS, N = 0.002, 3


def scoring(i, primary, parent=0, op='improve'):
    n = Node(id=i, parent_id=parent, operation=op)
    n.is_buggy, n.val_primary = False, primary
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
