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
    """A node that ran and produced a score."""
    n = Node(id=i, parent_id=parent, operation=op)
    n.is_buggy, n.val_primary, n.accepted = False, primary, accepted
    return n


def crashed(i, parent=0):
    """A node that failed before scoring."""
    return Node(id=i, parent_id=parent, operation='improve')   # is_buggy defaults True


def eda(i):
    """The one analysis pass, which never competes for best."""
    n = Node(id=i, parent_id=0, operation='eda')
    n.is_buggy = False          # ran fine, just produced no score
    return n


def build(nodes):
    """A journal containing the given nodes, in order."""
    j = Journal()
    for n in nodes:
        j.append(n)
    return j


def test_crashes_do_not_converge():
    """Three failures in a row are not three flat results."""
    j = build([scoring(0, 0.6014, None, 'baseline')] + [crashed(i) for i in (1, 2, 3)])
    assert not j.has_converged(EPS, N), 'three crashes must not look like convergence'


def test_eda_does_not_converge():
    """The analysis pass produces no score, so it cannot flatten the window."""
    j = build([scoring(0, 0.6014, None, 'baseline'), eda(1),
               scoring(2, 0.6020), scoring(3, 0.6100)])
    assert not j.has_converged(EPS, N), 'EDA must not count toward the window'


def test_genuine_stagnation_converges():
    """Real experiments that stop improving must end the run."""
    j = build([scoring(0, 0.6014, None, 'baseline'),
               scoring(1, 0.6015), scoring(2, 0.6016), scoring(3, 0.6018)])
    assert j.has_converged(EPS, N), 'four flat scoring nodes are real convergence'


def test_real_improvement_does_not_converge():
    """A gain above epsilon keeps the run going."""
    j = build([scoring(0, 0.6014, None, 'baseline'),
               scoring(1, 0.6100), scoring(2, 0.6200), scoring(3, 0.6300)])
    assert not j.has_converged(EPS, N)


def test_too_few_nodes():
    """Convergence cannot fire before the window is full."""
    assert not build([scoring(0, 0.6014, None, 'baseline')]).has_converged(EPS, N)


def test_crashes_interleaved_with_progress():
    """Crashes between improving experiments must not drag the window flat."""
    j = build([scoring(0, 0.6014, None, 'baseline'), crashed(1),
               scoring(2, 0.6100), crashed(3), scoring(4, 0.6200)])
    assert not j.has_converged(EPS, N)


# Run smoke3 scored node 4 highest (0.6019) but the unbiased-exposure gate rejected it, and
# summary.json still named it the winner. A node the gates threw out must not be reported as
# best, and must not become the bar later nodes are measured against.

def test_rejected_node_is_not_best():
    """Scoring highest is not enough; a rejected node is not the winner."""
    j = build([scoring(0, 0.6014, None, 'baseline'),
               scoring(3, 0.6017),
               scoring(4, 0.6019, accepted=False)])
    assert j.best.id == 3, 'a gate-rejected node must not be reported as best'
    assert j.best.val_primary == 0.6017


def test_rejected_nodes_do_not_raise_the_bar():
    """Higher-scoring nodes the gates rejected leave the run stalled, not advancing."""
    j = build([scoring(0, 0.6014, None, 'baseline')] +
              [scoring(i, 0.60 + 0.01 * i, accepted=False) for i in (1, 2, 3)])
    assert j.best_history() == [0.6014] * 4, 'rejected scores must not enter the history'
    assert j.has_converged(EPS, N), 'nothing accepted in three tries is genuine stagnation'


def test_all_rejected_leaves_no_best():
    """If nothing was accepted there is no best node."""
    j = build([scoring(0, 0.6014, None, 'baseline', accepted=False)])
    assert j.best is None


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'  PASS  {name}')
    print('\nall journal tests passed')
