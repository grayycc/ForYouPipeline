"""The solution tree. Every attempt is a Node; the Journal is the tree plus the bookkeeping
that makes greedy search and convergence detection possible.

The point of the tree (vs. one long conversation) is recovery: a bad idea costs one node,
not the run. Improve always branches from the current best, so the search can never get stuck.
"""
import dataclasses, json, time
from typing import Dict, Optional, List

WORST = -1.0  # buggy nodes get this so they can never become "best"


def rank_key(node):
    """How two candidates are ordered: by validation score, full stop.

    The challenge defines the selection rule for us -- "the submission scored for ranking is the
    validation-best checkpoint" -- so ranking by anything else is scoring a different contest
    than the one being judged.

    This used to band validation into EPSILON-wide bins and break ties on the random-exposure
    score, on the theory that within one band validation is only resolving noise. Two things
    were wrong with that. The bands are absolute, so band 301 spans 0.6010-0.6030 -- which in
    runs/v5 was the entire range of every good node the agent produced, making the tie-break the
    *only* thing that mattered. And the tie-break points the wrong way: across v5's 15 scoring
    nodes r(val, unbiased) = -0.33, because the hidden test set is a standard logged-exposure
    split like validation, not a random-exposure one. Preferring the higher unbiased score
    therefore prefers the model that will score worse on the metric being judged. Nodes 12, 13,
    15 and 18 all beat the incumbent on validation and all four were rejected on that tie-break;
    the run shipped +0.000267 instead of node 18's +0.00113.

    The unbiased score keeps its job as a *veto* in Agent._accept -- a candidate whose
    random-exposure score collapses relative to its parent is rejected outright. Guarding
    against a distribution-shift collapse is what it was introduced for. Expressing a preference
    between two scores that validation cannot separate is not.
    """
    return node.val_primary if node.val_primary is not None else WORST


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
    seed_unbiased_scores: List[float] = dataclasses.field(default_factory=list)

    # execution outcome
    is_buggy: bool = True               # guilty until proven otherwise
    buggy_reason: str = ''
    exception_type: Optional[str] = None
    stdout_tail: str = ''
    # Declared, not merely assigned by the orchestrator. Every write to `stderr_tail` was a
    # dynamic attribute set inside _apply_result, so any node that never reached it -- one whose
    # generation raised, or whose submissions failed validation first -- had no such attribute,
    # and both _validate_submissions and the planner's _describe read it unconditionally. That
    # is an AttributeError in the middle of a long run, thrown by the error path.
    stderr_tail: str = ''
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
        # Monotonic: a laptop sleeping mid-run must not be counted as elapsed work.
        self.t0 = time.monotonic()
        # Findings from the one EDA pass, injected into every later planner call.
        self.eda_findings: str = ''
        # How far the unbiased score may fall before a change is called overfitting. Measured
        # from the baseline's own seed spread rather than borrowed from the validation metric's
        # std -- see config.UNBIASED_TOLERANCE_SIGMAS. None until the baseline resolves.
        self.unbiased_tolerance: Optional[float] = None
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
        """Every node that produced a score. Used for counting work done, not for selection."""
        return [n for n in self.nodes if not n.is_buggy and n.val_primary is not None]

    @property
    def accepted_nodes(self) -> List[Node]:
        """Nodes whose gain survived every gate. This is what `best` is drawn from.

        A node rejected by the unbiased-exposure gate scored well on validation but was judged
        not to be a real gain. Letting it into `best` would make it both the bar every later
        node must clear and the branch point they grow from -- so one phantom gain suppresses
        every genuine one behind it. That is what happened for ten iterations of runs/v2.
        """
        return [n for n in self.nodes
                if n.accepted and not n.is_buggy and n.val_primary is not None]

    @property
    def drafts(self) -> List[Node]:
        return [n for n in self.nodes if n.operation == 'draft']

    def has_operation(self, op: str) -> bool:
        return any(n.operation == op and not n.is_buggy for n in self.nodes)

    @property
    def best(self) -> Optional[Node]:
        """Highest-ranking accepted node -- see `rank_key`."""
        acc = self.accepted_nodes
        return max(acc, key=rank_key) if acc else None

    def outranks_best(self, node: Node) -> bool:
        """Would this node take the incumbency? Asked before the node is appended.

        The accept decision and the incumbency ordering have to be the same question, or a node
        can be accepted, promoted over the best submission on disk, and then not actually be
        `best` -- which is how runs/v3's node 16 (0.6028) would have overwritten node 15's
        0.6035 submission.
        """
        best = self.best
        return True if best is None else rank_key(node) > rank_key(best)

    # Operations that only need to succeed once. Debugging a failed one after another attempt
    # has already succeeded re-derives a result the run already holds.
    ONE_SHOT_OPS = ('baseline', 'eda')

    def buggy_leaves(self) -> List[Node]:
        """Buggy nodes nobody has tried to fix yet, still under the debug-depth cap.

        A failed node is only worth an iteration if its job is still undone. runs/v3 spent
        iteration 12 debugging the baseline that had crashed at iteration 0 -- eleven iterations
        after iteration 1 produced a working baseline -- and duly re-derived 0.6016, a number
        the run already had. Drafts are exempt: they are exploration, so a crashed one is still
        worth fixing even once other drafts have run.
        """
        parents = {n.parent_id for n in self.nodes}
        settled = {n.operation for n in self.nodes
                   if not n.is_buggy and n.operation in self.ONE_SHOT_OPS}
        return [n for n in self.nodes
                if n.is_buggy and n.id not in parents and n.debug_depth < 3
                and n.operation not in settled]

    def best_history(self) -> List[float]:
        """Best-so-far *accepted* validation primary after each scoring node, in order.

        Nodes that produced no score are skipped: crashes, and the one-off EDA pass. Counting
        them flattens the window, so three consecutive failures would otherwise look identical
        to three experiments that failed to improve and fire convergence on their own.

        The running best tracks accepted nodes only, matching `best`. A rejected node's score
        is not progress, so it must not be able to end the run by looking like progress.
        """
        out, best = [], WORST
        for n in self.nodes:
            if n.is_buggy or n.val_primary is None:
                continue
            if n.accepted:
                best = max(best, n.score)
            out.append(best)
        return out

    def has_converged(self, epsilon: float, n_required: int,
                      min_scoring_nodes: int = 0) -> bool:
        """Converged when best-so-far hasn't improved by > epsilon over N scoring iterations.

        `min_scoring_nodes` is a floor on how much search must have happened before the rule
        may fire at all. Without it, three consecutive experiments that fail to beat a strong
        baseline -- the ordinary early state of a search -- are indistinguishable from a
        genuine plateau, and the run ends around iteration 4 of 50.
        """
        hist = self.best_history()
        if len(hist) < min_scoring_nodes:
            return False
        if len(hist) < n_required + 1:
            return False
        window = hist[-(n_required + 1):]
        return (window[-1] - window[0]) <= epsilon

    def summary_table(self, limit: int = 25) -> str:
        """Compact prior-attempt table for prompts. Never send the whole tree (context cost)."""
        if not self.nodes:
            return '(no attempts yet)'
        from . import diagnose
        lines = ['| id | parent | op | val_primary | train | status | diagnosis | what was tried |',
                 '|---|---|---|---|---|---|---|---|']
        for n in self.nodes[-limit:]:
            score = f'{n.val_primary:.4f}' if n.val_primary is not None else '--'
            train = f'{n.train_primary:.4f}' if n.train_primary is not None else '--'
            status = (f'BUGGY({n.buggy_reason})' if n.is_buggy
                      else 'ACCEPTED' if n.accepted else 'rejected')
            verdict = diagnose.classify(n, self)
            spec = n.spec or {}
            what = (spec.get('proposed_change') or n.hypothesis or n.plan or '')
            tags = ','.join(spec.get('tags') or [])
            desc = f'[{tags}] {what}'[:170].replace('\n', ' ') if tags else what[:170].replace('\n', ' ')
            lines.append(f'| {n.id} | {n.parent_id} | {n.operation} | {score} | {train} | '
                         f'{status} | {verdict} | {desc} |')
        return '\n'.join(lines)

    def to_jsonl(self, path: str):
        with open(path, 'w') as fh:
            for n in self.nodes:
                fh.write(json.dumps(dataclasses.asdict(n)) + '\n')
