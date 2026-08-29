"""The solution tree. Every attempt is a Node; the Journal is the tree plus the bookkeeping
that makes greedy search and convergence detection possible.

The point of the tree (vs. one long conversation) is recovery: a bad idea costs one node,
not the run. Improve always branches from the current best, so the search can never get stuck.
"""
import dataclasses, json, time
from typing import Optional, List

WORST = -1.0  # buggy nodes get this so they can never become "best"


@dataclasses.dataclass
class Node:
    id: int
    parent_id: Optional[int]
    operation: str                      # baseline | draft | debug | improve
    hypothesis: str = ''
    code: str = ''
    plan: str = ''

    # metrics parsed from the executed script
    val_gauc: Optional[float] = None
    val_ndcg5: Optional[float] = None
    val_primary: Optional[float] = None
    unbiased_val_primary: Optional[float] = None
    seeds_averaged: int = 1
    seed_scores: List[float] = dataclasses.field(default_factory=list)

    # execution outcome
    is_buggy: bool = True               # guilty until proven otherwise
    buggy_reason: str = ''
    exception_type: Optional[str] = None
    stdout_tail: str = ''
    exec_time: float = 0.0
    submission_ok: bool = False

    # loop bookkeeping
    accepted: bool = False
    reflection: str = ''
    recovery_action: Optional[str] = None
    debug_depth: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_seconds: float = 0.0

    @property
    def score(self) -> float:
        """Comparable score for greedy selection. Buggy nodes sort to the bottom."""
        if self.is_buggy or self.val_primary is None:
            return WORST
        return self.val_primary


class Journal:
    def __init__(self):
        self.nodes: List[Node] = []
        self.t0 = time.time()

    def __len__(self):
        return len(self.nodes)

    def append(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def get(self, node_id: int) -> Optional[Node]:
        return next((n for n in self.nodes if n.id == node_id), None)

    @property
    def good_nodes(self) -> List[Node]:
        return [n for n in self.nodes if not n.is_buggy and n.val_primary is not None]

    @property
    def drafts(self) -> List[Node]:
        return [n for n in self.nodes if n.operation == 'draft']

    @property
    def best(self) -> Optional[Node]:
        good = self.good_nodes
        return max(good, key=lambda n: n.score) if good else None

    def buggy_leaves(self) -> List[Node]:
        """Buggy nodes nobody has tried to fix yet, still under the debug-depth cap."""
        parents = {n.parent_id for n in self.nodes}
        return [n for n in self.nodes
                if n.is_buggy and n.id not in parents and n.debug_depth < 3]

    def best_history(self) -> List[float]:
        """Best-so-far validation primary after each node, in order."""
        out, best = [], WORST
        for n in self.nodes:
            best = max(best, n.score)
            out.append(best)
        return out

    def has_converged(self, epsilon: float, n_required: int) -> bool:
        """Converged when best-so-far hasn't improved by > epsilon over N consecutive iterations."""
        hist = [h for h in self.best_history() if h > WORST]
        if len(hist) < n_required + 1:
            return False
        window = hist[-(n_required + 1):]
        return (window[-1] - window[0]) <= epsilon

    def summary_table(self, limit: int = 25) -> str:
        """Compact prior-attempt table for prompts. Never send the whole tree (context cost)."""
        if not self.nodes:
            return '(no attempts yet)'
        lines = ['| id | parent | op | val_primary | status | hypothesis -> reflection |',
                 '|---|---|---|---|---|---|']
        for n in self.nodes[-limit:]:
            score = f'{n.val_primary:.4f}' if n.val_primary is not None else '--'
            status = f'BUGGY({n.buggy_reason})' if n.is_buggy else ('ACCEPTED' if n.accepted else 'rejected')
            hyp = (n.hypothesis or n.plan or '')[:150].replace('\n', ' ')
            ref = (n.reflection or '')[:200].replace('\n', ' ')
            lines.append(f'| {n.id} | {n.parent_id} | {n.operation} | {score} | {status} | {hyp} -> {ref} |')
        return '\n'.join(lines)

    def to_jsonl(self, path: str):
        with open(path, 'w') as fh:
            for n in self.nodes:
                fh.write(json.dumps(dataclasses.asdict(n)) + '\n')
