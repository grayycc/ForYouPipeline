"""The solution tree. Every attempt is a Node; the Journal is the tree plus the bookkeeping
that makes greedy search and convergence detection possible.

The point of the tree (vs. one long conversation) is recovery: a bad idea costs one node,
not the run. Improve always branches from the current best, so the search can never get stuck.
"""
import dataclasses, json, time
from typing import Dict, Optional, List

WORST = -1.0  # buggy nodes get this so they can never become "best"


@dataclasses.dataclass
class Node:
    id: int
    parent_id: Optional[int]
    operation: str                      # baseline | eda | improve | debug
    hypothesis: str = ''
    code: str = ''
    plan: str = ''

    # what the planner decided, and what the deterministic checks made of it
    spec: Optional[Dict] = None
    gate_result: str = ''
    review_verdict: str = ''
    review_reason: str = ''

    # metrics parsed from the executed script
    train_primary: Optional[float] = None
    val_gauc: Optional[float] = None
    val_ndcg5: Optional[float] = None
    val_primary: Optional[float] = None
    unbiased_val_primary: Optional[float] = None
    seeds_averaged: int = 1
    seed_scores: List[float] = dataclasses.field(default_factory=list)
    # True when val_primary is the score of the seeds' combined prediction rather than the mean
    # of their separate scores. The two differ: the metric ranks rows, so averaging predictions
    # cancels each model's random error and beats every individual seed.
    seeds_ensembled: bool = False

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
        # Findings from the one EDA pass, injected into every later planner call.
        self.eda_findings: str = ''
        # Every paper any search returned this run: id -> {title, outcome of prior use}.
        # Citations are checked against this, so the planner cannot cite a paper that no
        # real search returned.
        self.citation_registry: Dict[str, Dict] = {}

    def register_citation(self, record: Dict):
        """Record a paper a search returned. Keeps the first sighting; outcome is filled in
        later, once a node that cited it has a result."""
        pid = record.get('id')
        if not pid:
            return
        entry = self.citation_registry.setdefault(
            pid, {'title': record.get('title', ''), 'doi': record.get('doi', ''),
                  'used_in': ''})
        if record.get('doi') and not entry.get('doi'):
            entry['doi'] = record['doi']

    def note_citation_outcome(self, node: 'Node'):
        """After a node resolves, annotate the papers it cited with what happened, so a later
        search shows whether that idea worked rather than merely that it was seen."""
        if not node.spec:
            return
        from .spec import _extract_ids
        lit = (node.spec.get('evidence') or {}).get('literature', '')
        if node.is_buggy:
            outcome = f'node {node.id}, failed to run'
        elif node.val_primary is None:
            outcome = f'node {node.id}, no score'
        else:
            delta = node.val_primary - (self.best.val_primary if self.best else node.val_primary)
            verdict = ('accepted' if node.accepted
                       else 'noise' if abs(delta) < 0.002 else 'rejected')
            outcome = f'node {node.id}, primary {node.val_primary:.4f}, {verdict}'
        for pid in _extract_ids(str(lit)):
            for key in self.citation_registry:
                if key.lower() == pid.lower():
                    self.citation_registry[key]['used_in'] = outcome

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

    def has_operation(self, op: str) -> bool:
        return any(n.operation == op and not n.is_buggy for n in self.nodes)

    @property
    def accepted_nodes(self) -> List[Node]:
        return [n for n in self.good_nodes if n.accepted]

    @property
    def best(self) -> Optional[Node]:
        """The best *accepted* node.

        Scoring highest is not enough. A node can be rejected after the fact -- by the leakage
        reviewer, or by the unbiased-exposure gate when its validation gain does not survive on
        randomly-exposed traffic -- and a node rejected for those reasons must not become the
        bar that later nodes are measured against, nor be reported as the run's winner.
        """
        acc = self.accepted_nodes
        return max(acc, key=lambda n: n.score) if acc else None

    def buggy_leaves(self) -> List[Node]:
        """Buggy nodes nobody has tried to fix yet, still under the debug-depth cap.

        A node with no code is skipped: it failed before any was written, so there is nothing
        to repair and the debugger can only invent a fresh experiment while pretending to fix
        one. That wastes the iteration and misattributes the result.
        """
        parents = {n.parent_id for n in self.nodes}
        return [n for n in self.nodes
                if n.is_buggy and n.code and n.id not in parents and n.debug_depth < 3]

    def best_history(self) -> List[float]:
        """Best-so-far validation primary after each *scoring* node, in order.

        Nodes that produced no score are skipped: crashes, and the one-off EDA pass. Counting
        them flattens the window, so three consecutive failures would otherwise look identical
        to three experiments that failed to improve and fire convergence on their own.
        """
        out, best = [], WORST
        for n in self.nodes:
            if n.is_buggy or n.val_primary is None:
                continue
            # One entry per scoring experiment, so a run of failures still advances the
            # convergence window -- but only an accepted node may raise the bar, matching
            # `best`. A node the gates rejected has not moved the run forward.
            if n.accepted:
                best = max(best, n.score)
            out.append(best)
        return out

    def has_converged(self, epsilon: float, n_required: int) -> bool:
        """Converged when best-so-far hasn't improved by > epsilon over N scoring iterations."""
        hist = self.best_history()
        if len(hist) < n_required + 1:
            return False
        window = hist[-(n_required + 1):]
        return (window[-1] - window[0]) <= epsilon

    def summary_table(self, limit: int = 25) -> str:
        """Compact prior-attempt table for prompts. Never send the whole tree (context cost)."""
        if not self.nodes:
            return '(no attempts yet)'
        lines = ['| id | parent | op | val_primary | status | what was tried |',
                 '|---|---|---|---|---|---|']
        for n in self.nodes[-limit:]:
            score = f'{n.val_primary:.4f}' if n.val_primary is not None else '--'
            status = (f'BUGGY({n.buggy_reason})' if n.is_buggy
                      else 'ACCEPTED' if n.accepted else 'rejected')
            spec = n.spec or {}
            what = (spec.get('proposed_change') or n.hypothesis or n.plan or '')
            tags = ','.join(spec.get('tags') or [])
            desc = f'[{tags}] {what}'[:170].replace('\n', ' ') if tags else what[:170].replace('\n', ' ')
            lines.append(f'| {n.id} | {n.parent_id} | {n.operation} | {score} | {status} | {desc} |')
        return '\n'.join(lines)

    def to_jsonl(self, path: str):
        with open(path, 'w') as fh:
            for n in self.nodes:
                fh.write(json.dumps(dataclasses.asdict(n)) + '\n')
