"""The loop. Selects the next operation, sequences the roles, records what happened.

Greedy search over a tree of experiments. Every improvement branches from the current best
node, so a bad idea costs one node rather than the run and the search can never get stuck.

This module never builds a prompt or calls a model directly -- roles do that. What lives here
is the ordering, the accept/reject decision, and the bookkeeping, all of which plain code does
more reliably than a model would.
"""
import json
import os
import random
import shutil
import statistics
import time

from . import config, diagnose, executor, gates
from .journal import Journal, Node
from .llm import LLMClient, LLMError
from .roles import baseline as baseline_role
from .roles import coder as coder_role
from .roles import debugger as debugger_role
from .roles import draft as draft_role
from .roles import eda as eda_role
from .roles import planner as planner_role
from .roles import reviewer as reviewer_role


class Agent:
    def __init__(self, run_id: str, seed: int = 0):
        self.run_dir = os.path.join(config.RUNS_DIR, run_id)
        self.nodes_dir = os.path.join(self.run_dir, 'nodes')
        os.makedirs(self.nodes_dir, exist_ok=True)
        self.journal = Journal()
        self.llm = LLMClient()
        self.rng = random.Random(seed)
        self.t0 = time.monotonic()
        self.converged_at = None
        self.refit_status = 'not attempted'

    def write_dummy_submissions(self):
        """Always keep a valid submission on disk, from before the first iteration."""
        from data import load
        from submit import write_submission
        splits = load(config.DATA_DIR)
        for split in ('valid', 'test'):
            path = os.path.join(self.run_dir, f'best_submission_{split}.csv')
            if not os.path.exists(path):
                write_submission(path, splits[split], [0.0] * len(splits[split]))
        print(f'  [safety] placeholder submissions written to {self.run_dir}')

    def select(self):
        """baseline once, then EDA once, then a few fresh drafts, then debug or improve.

        The drafts matter: `improve` always branches from the incumbent, so without them the
        search can only ever mutate the first working solution. Fourteen consecutive iterations
        of runs/v2 were variations on one FM script for exactly this reason, which no amount of
        better prompting can fix -- a different model class has to start from a blank file.
        """
        j = self.journal
        if not j.has_operation('baseline'):
            return 'baseline', None
        if not j.has_operation('eda'):
            return 'eda', None
        if len(j.drafts) < config.MIN_DRAFTS:
            return 'draft', None
        if self._stalled():
            print(f'  [stall] {config.STALL_NOISE_STREAK} consecutive results inside the noise '
                  f'floor -- drafting instead of editing the incumbent again')
            return 'draft', None
        eligible = [n for n in j.buggy_leaves() if n.debug_depth < config.MAX_DEBUG_DEPTH]
        if eligible and self.rng.random() < config.DEBUG_PROBABILITY:
            return 'debug', self.rng.choice(eligible)
        return 'improve', j.best

    def _stalled(self) -> bool:
        """Have the last few scoring results all landed inside the noise floor?

        Crashes and the EDA pass are skipped rather than breaking the streak -- they carry no
        information about whether the incumbent's neighbourhood is exhausted. A `regression` or
        `overfit` does break it: those are informative failures that point somewhere, unlike a
        result that could not be distinguished from the incumbent at all.
        """
        j = self.journal
        if len(j.drafts) >= config.MAX_DRAFTS:
            return False
        recent_draft = any(n.operation == 'draft'
                           for n in j.nodes[-config.STALL_DRAFT_COOLDOWN:])
        if recent_draft:
            return False
        streak = 0
        for n in reversed(j.nodes):
            if n.is_buggy or n.val_primary is None:
                continue
            if diagnose.classify(n, j) != diagnose.NOISE:
                break
            streak += 1
            if streak >= config.STALL_NOISE_STREAK:
                return True
        return False

    def step(self, iteration: int) -> Node:
        t_iter = time.monotonic()
        op, parent = self.select()
        print(f'\n=== iteration {iteration} | {op}'
              f'{f" from node {parent.id}" if parent else ""} ===')

        node = Node(id=iteration, parent_id=parent.id if parent else None, operation=op,
                    debug_depth=(parent.debug_depth + 1) if op == 'debug' else 0)

        try:
            if op == 'baseline':
                self._generate_baseline(node, iteration)
            elif op == 'eda':
                self._generate_eda(node, iteration)
            elif op == 'draft':
                self._generate_draft(node, iteration)
            elif op == 'debug':
                self._generate_debug(node, iteration, parent)
            else:
                self._generate_experiment(node, iteration, parent)
        except LLMError as e:
            node.is_buggy, node.buggy_reason = True, 'llm_failure'
            node.exception_type, node.stderr_tail = 'LLMError', str(e)
            node.recovery_action = 'fall back to the best node next iteration'
            node.wall_seconds = time.monotonic() - t_iter
            self.journal.append(node)
            self._log(node)
            print(f'  -> LLM FAILURE: {e}')
            return node

        node_dir = os.path.join(self.nodes_dir, f'node_{iteration}')
        os.makedirs(node_dir, exist_ok=True)
        code_path = os.path.join(node_dir, 'solution.py')
        with open(code_path, 'w') as fh:
            fh.write(node.code)

        is_eda = (op == 'eda')
        res = executor.run_solution(code_path, node_dir, seed=config.CONFIRM_SEEDS[0],
                                    require_metrics=not is_eda)
        self._apply_result(node, res)
        with open(os.path.join(node_dir, 'stdout.txt'), 'w') as fh:
            fh.write(res['stdout_tail'] + '\n--- stderr ---\n' + res['stderr_tail'])

        if is_eda:
            if not node.is_buggy:
                self.journal.eda_findings = res['stdout_tail'][-3000:]
                print(f'  -> EDA findings captured ({len(self.journal.eda_findings)} chars)')
            self._finish(node, t_iter)
            return node

        if not node.is_buggy:
            self._validate_submissions(node, node_dir)

        if not node.is_buggy and self._worth_confirming(node):
            self._confirm_seeds(node, code_path, node_dir)

        # Every multi-seed node is another free measurement of the unbiased metric's noise, so
        # the gate's tolerance is re-pooled after each one rather than fixed by the baseline's
        # first three draws.
        if not node.is_buggy and len(node.seed_unbiased_scores or []) > 1:
            self._calibrate_unbiased_tolerance(node)
        elif self.journal.unbiased_tolerance is None:
            self._calibrate_unbiased_tolerance(node)

        node.accepted = self._accept(node)
        if node.accepted:
            self._promote(node, node_dir)

        self._finish(node, t_iter)
        return node

    def _finish(self, node, t_iter):
        node.wall_seconds = time.monotonic() - t_iter
        self.journal.append(node)
        self.journal.note_citation_outcome(node)
        self._log(node)
        self._report(node)

    def _generate_baseline(self, node, iteration):
        node.hypothesis, node.code, ti, to = baseline_role.run(
            self.llm, iteration, self.journal)
        node.tokens_in, node.tokens_out = ti, to

    def _generate_eda(self, node, iteration):
        node.hypothesis, node.code, ti, to = eda_role.run(self.llm, iteration, self.journal)
        node.tokens_in, node.tokens_out = ti, to

    def _generate_draft(self, node, iteration):
        from .spec import ExperimentSpec

        node.hypothesis, tags, node.code, ti, to = draft_role.run(
            self.llm, iteration, self.journal)
        node.tokens_in, node.tokens_out = ti, to

        # A draft that restates a mechanism the run has already refuted is the most expensive
        # way to learn nothing, and it is what runs/v5 did four times over. One redraft, told
        # exactly which node it collided with, is far cheaper than the wasted iteration.
        dup = gates._find_duplicate(ExperimentSpec(tags=tags), self.journal)
        if dup is not None and tags:
            print(f'  [gate] draft repeats the mechanism of node {dup.id} (tags {tags}) '
                  f'-- redrafting once')
            node.gate_result = f'redrafted away from node {dup.id}'
            avoid = (f'Your first attempt proposed {tags}, which is the mechanism node '
                     f'{dup.id} already tested. Choose a different one.')
            h2, t2, c2, ti2, to2 = draft_role.run(self.llm, iteration, self.journal, avoid=avoid)
            node.tokens_in += ti2
            node.tokens_out += to2
            if c2.strip():
                node.hypothesis, tags, node.code = h2, t2, c2
        else:
            node.gate_result = 'passed'

        # The draft's own tags go on the node, so it is visible to the gate as prior work later.
        node.spec = {'tags': tags, 'proposed_change': node.hypothesis or '',
                     'mechanism': '', 'hypothesis': node.hypothesis or ''}

        # Drafts were skipping the leakage review entirely, which is backwards: a draft is
        # written from scratch, so it is *more* likely to reinvent a leaky target encoding than
        # an improve that inherits a reviewed solution. A leaky draft that happened to score
        # well could have become the submission with nothing to stop it.
        from .spec import ExperimentSpec
        stand_in = ExperimentSpec(proposed_change=node.hypothesis or 'a fresh solution',
                                  risks={'leakage': 'not assessed -- this is a draft'})
        verdict, reason = self._review(node, node.code, stand_in)
        node.review_verdict, node.review_reason = verdict, reason
        if verdict == 'LEAK':
            print(f'  [review] leakage confirmed in draft: {reason[:120]}')

    def _generate_debug(self, node, iteration, parent):
        node.spec = parent.spec
        node.hypothesis, node.code, ti, to = debugger_role.run(
            self.llm, iteration, self.journal, parent)
        node.tokens_in, node.tokens_out = ti, to

    def _review(self, node, code, spec, parent_code=None):
        """Leakage verdict, confirmed before it is allowed to block anything.

        The reviewer is a stochastic judge making a decision that costs a whole iteration, and
        measurement says a single sample is not safe in either direction: re-reviewing runs/v4's
        flagged nodes, three false positives repeated across samples while a genuine leak came
        back CLEAN one time in two. Requiring two LEAKs in a row costs one extra call only on
        nodes that were going to be flagged anyway, and turns a coin-flip into a decision.

        Returns (verdict, reason) and accumulates token counts onto the node.
        """
        verdict, reason, ti, to = reviewer_role.run(self.llm, code, spec, parent_code)
        node.tokens_in += ti
        node.tokens_out += to
        if verdict != 'LEAK':
            return verdict, reason
        confirm, reason2, ti2, to2 = reviewer_role.run(self.llm, code, spec, parent_code)
        node.tokens_in += ti2
        node.tokens_out += to2
        if confirm != 'LEAK':
            print('  [review] leakage flagged once but not on confirmation -- treating as clean')
            return 'CLEAN', ''
        return 'LEAK', (reason2 or reason)

    def _generate_experiment(self, node, iteration, parent):
        """Plan, validate the plan, implement it, review it for leakage."""
        spec, err, _, ti, to = planner_role.run(self.llm, iteration, self.journal)
        node.tokens_in, node.tokens_out = ti, to

        reasons = [err] if err else []
        if spec is not None:
            ok, reasons = gates.check_spec(spec, self.journal)
            node.gate_result = 'passed' if ok else 'rejected'
        if reasons:
            print(f'  [gates] rejected: {reasons[0]}')
            spec2, err2, _, ti2, to2 = planner_role.replan(
                self.llm, iteration, self.journal, reasons)
            node.tokens_in += ti2
            node.tokens_out += to2
            if spec2 is not None:
                ok2, reasons2 = gates.check_spec(spec2, self.journal)
                spec = spec2
                node.gate_result = 'passed_after_replan' if ok2 else 'passed_with_warnings'
                if not ok2:
                    # A closed mechanism is recorded distinctly. Everything else the gates raise
                    # is advisory by design, but this one means the planner re-proposed a family
                    # whose every attempt came back worse even after being told it was closed --
                    # which is the pattern worth being able to count in the log afterwards.
                    if any('is closed:' in r for r in reasons2):
                        node.gate_result = 'closed_mechanism_override'
                    print(f'  [gates] still flagged, proceeding anyway: {reasons2[0]}')
            else:
                node.gate_result = 'passed_with_warnings'
                print(f'  [gates] replan unparseable ({err2}), proceeding with original')

        if spec is None:
            raise LLMError('planner produced no usable spec after a replan')

        node.spec = spec.to_dict()
        node.hypothesis = spec.hypothesis
        print(f'  [plan] {spec.summary(110)}')

        parent_code = parent.code if parent else ''
        parent_stdout = parent.stdout_tail if parent else ''
        node.code, ti3, to3 = coder_role.run(self.llm, spec, parent_code, parent_stdout)
        node.tokens_in += ti3
        node.tokens_out += to3

        verdict, reason = self._review(node, node.code, spec, parent_code)
        node.review_verdict, node.review_reason = verdict, reason
        if verdict == 'LEAK':
            print(f'  [review] leakage confirmed: {reason[:120]}')
            node.code, ti5, to5 = coder_role.revise(self.llm, spec, node.code, reason)
            node.tokens_in += ti5
            node.tokens_out += to5
            verdict2, reason2 = self._review(node, node.code, spec, parent_code)
            node.review_verdict, node.review_reason = verdict2, reason2
            if verdict2 == 'LEAK':
                print('  [review] still flagged -- will run, but cannot become best')

    def _validate_submissions(self, node, node_dir):
        ok_v, msg_v = executor.check_submission(
            os.path.join(node_dir, 'submission_valid.csv'), 'valid')
        ok_t, msg_t = executor.check_submission(
            os.path.join(node_dir, 'submission_test.csv'), 'test')
        node.submission_ok = ok_v and ok_t
        if not node.submission_ok:
            node.is_buggy, node.buggy_reason = True, 'invalid_submission'
            node.stderr_tail = (node.stderr_tail + '\n' + msg_v + '\n' + msg_t).strip()

    def _apply_result(self, node: Node, res: dict):
        m = res['metrics']
        node.train_primary = m.get('train_primary')
        node.val_gauc = m.get('val_gauc')
        node.val_ndcg5 = m.get('val_ndcg5')
        node.val_primary = m.get('val_primary')
        node.unbiased_val_primary = m.get('unbiased_val_primary')
        node.is_buggy = res['is_buggy']
        node.buggy_reason = res['buggy_reason']
        node.exception_type = res['exception_type']
        node.stdout_tail = res['stdout_tail']
        node.stderr_tail = res['stderr_tail']
        node.exec_time = res['exec_time']
        if node.val_primary is not None:
            node.seed_scores = [node.val_primary]
        if node.unbiased_val_primary is not None:
            node.seed_unbiased_scores = [node.unbiased_val_primary]

        # A generated solution that stubs UNBIASED_PRIMARY out with the validation score defeats
        # the acceptance gate silently -- runs/v2 node 5 shipped exactly that, comment included,
        # and passed every check. The two are computed on different row sets, so they cannot
        # legitimately be bit-identical.
        if (not node.is_buggy and node.val_primary is not None
                and node.unbiased_val_primary is not None
                and abs(node.unbiased_val_primary - node.val_primary) < 1e-9):
            node.is_buggy, node.buggy_reason = True, 'unbiased_is_stub'
            node.stderr_tail = (
                'UNBIASED_PRIMARY is bit-identical to VAL_PRIMARY, so it was not computed on the '
                'random-exposure log. Compute it on log_random_4_22_to_5_08_pure.csv restricted '
                'to the validation date window, as the task description requires.'
            ).strip()

    def _worth_confirming(self, node: Node) -> bool:
        """Spend the extra seeds on anything that could still be accepted.

        The window is a full epsilon rather than CONFIRM_TRIGGER because the tie-break in
        _accept can promote a node up to epsilon below the incumbent. A node accepted on one
        seed would defeat the point of seed-averaging entirely.
        """
        best = self.journal.best
        if best is None:
            return True
        return node.val_primary > best.val_primary - config.EPSILON

    def _confirm_seeds(self, node: Node, code_path: str, node_dir: str):
        """Re-run a promising candidate on the remaining seeds and average, so a lucky seed
        cannot be mistaken for a gain."""
        print(f'  [confirm] {node.val_primary:.4f} looks promising -- re-running on '
              f'{len(config.CONFIRM_SEEDS) - 1} more seeds')
        for s in config.CONFIRM_SEEDS[1:]:
            r = executor.run_solution(code_path, node_dir, seed=s)
            p = r['metrics'].get('val_primary')
            if r['is_buggy'] or p is None:
                print(f'    seed {s}: failed ({r["buggy_reason"]}) -- excluded')
                continue
            node.seed_scores.append(p)
            # The unbiased score is averaged over the same seeds. Leaving it at the seed-0 value
            # while averaging validation compares a noisy number against a smoothed one, and the
            # gate then rejects on noise -- which is what happened to every candidate in v2.
            u = r['metrics'].get('unbiased_val_primary')
            if u is not None:
                node.seed_unbiased_scores.append(u)
        if len(node.seed_scores) > 1:
            node.val_primary = sum(node.seed_scores) / len(node.seed_scores)
            node.seeds_averaged = len(node.seed_scores)
            print(f'    seeds [{" ".join(f"{x:.4f}" for x in node.seed_scores)}] '
                  f'-> mean {node.val_primary:.4f}')
        if len(node.seed_unbiased_scores) > 1:
            node.unbiased_val_primary = (sum(node.seed_unbiased_scores)
                                         / len(node.seed_unbiased_scores))
            print(f'    unbiased [{" ".join(f"{x:.4f}" for x in node.seed_unbiased_scores)}] '
                  f'-> mean {node.unbiased_val_primary:.4f}')

    def _calibrate_unbiased_tolerance(self, node: Node = None):
        """Pool the within-node seed spread across every node that has run multiple seeds.

        This was calibrated from the baseline node alone, on the reasoning that its three seeds
        are a free measurement of the metric's noise. Three samples is simply too few to
        estimate a standard deviation with: the baseline drew a narrow triple (sigma 0.00134)
        while the pooled estimate over runs/v3 and v5 -- 18 nodes, 36 degrees of freedom -- is
        0.00215. The gate was therefore 60% tighter than the noise it was meant to sit above,
        and in v5 it vetoed node 18 on a drop of 0.0035 against a 0.0034 tolerance, which is a
        coin flip dressed as a decision.

        Pooling within-node deviations (not the spread of node means, which carries real
        between-model differences) re-estimates the same quantity from every seed the run has
        already paid for, and tightens as the run goes on. Recalculated after each multi-seed
        node rather than once, so a run is not stuck with whatever the first three draws said.
        """
        # The current node is not in the journal yet -- _finish appends it after _accept runs --
        # so it is folded in explicitly rather than waiting a whole iteration to count.
        groups = [n.seed_unbiased_scores for n in self.journal.nodes
                  if len(n.seed_unbiased_scores or []) > 1]
        if node is not None and len(node.seed_unbiased_scores or []) > 1:
            groups.append(node.seed_unbiased_scores)
        dof = sum(len(g) - 1 for g in groups)
        if dof >= 2:
            ss = sum((x - statistics.mean(g)) ** 2 for g in groups for x in g)
            sigma = (ss / dof) ** 0.5
            tol = max(config.UNBIASED_TOLERANCE_SIGMAS * sigma, config.SEED_STD)
            got = f'pooled sigma {sigma:.5f} over {len(groups)} nodes, {dof} dof'
        else:
            tol = config.UNBIASED_TOLERANCE_DEFAULT
            got = 'no spread measured yet'
        prev = self.journal.unbiased_tolerance
        self.journal.unbiased_tolerance = tol
        if prev is None or abs(prev - tol) > 1e-6:
            print(f'  [calibrate] unbiased gate tolerance {tol:.5f} ({got})')

    def _accept(self, node: Node) -> bool:
        if node.is_buggy or node.val_primary is None:
            return False
        if node.review_verdict == 'LEAK':
            node.recovery_action = 'blocked from best by the leakage reviewer'
            return False
        best = self.journal.best
        if best is None:
            return True

        # One question, asked once: does this outrank the incumbent? rank_key is the validation
        # score, because that is the checkpoint the challenge says it will score.
        if not self.journal.outranks_best(node):
            return False

        # Then the veto. The unbiased gate asks whether THIS change traded genuine preference
        # signal for logging-policy artifacts, which is a question about the parent->child delta.
        # Measuring it against the global best instead compares across model families, and those
        # differ in unbiased level by far more than any single change does: in runs/v3 the gap
        # between GBDT-style and FM-style solutions was 0.047 against a 0.0092 tolerance, five
        # times too wide. So the gate could only ever fire on a change of family and never within
        # one -- and it rejected nothing at all across 18 iterations. The parent shares the
        # lineage, so the family offset cancels and the tolerance means what it was calibrated
        # to mean. It falls back to the incumbent only when the parent has no unbiased score.
        ref = self.journal.get(node.parent_id) if node.parent_id is not None else None
        if ref is None or ref.unbiased_val_primary is None:
            ref = best
        tol = self.journal.unbiased_tolerance or config.UNBIASED_TOLERANCE_DEFAULT
        if node.unbiased_val_primary is not None and ref.unbiased_val_primary is not None:
            drop = ref.unbiased_val_primary - node.unbiased_val_primary
            if drop > tol:
                node.recovery_action = 'rejected by the unbiased-exposure gate'
                print(f'  [gate] validation rose but random-exposure fell {drop:.4f} vs node '
                      f'{ref.id} (tolerance {tol:.4f}) -- rejected as overfitting to the '
                      f'logging policy')
                return False
        return True

    def _promote(self, node: Node, node_dir: str):
        for split in ('valid', 'test'):
            src = os.path.join(node_dir, f'submission_{split}.csv')
            if os.path.exists(src):
                shutil.copy(src, os.path.join(self.run_dir, f'best_submission_{split}.csv'))
        shutil.copy(os.path.join(node_dir, 'solution.py'),
                    os.path.join(self.run_dir, 'best_solution.py'))

    def _log(self, node: Node):
        spec = node.spec or {}
        rec = {
            'iter': node.id, 'parent_id': node.parent_id, 'operation': node.operation,
            'hypothesis': node.hypothesis or spec.get('hypothesis', ''),
            'mechanism': spec.get('mechanism', ''),
            'evidence': spec.get('evidence', {}),
            'proposed_change': spec.get('proposed_change', ''),
            'expected_result': spec.get('expected_result', ''),
            'falsification_condition': spec.get('falsification_condition', ''),
            'risks': spec.get('risks', {}),
            'tags': spec.get('tags', []),
            'candidates_considered': spec.get('candidates_considered', []),
            'reflection': spec.get('reflection', '') or node.reflection,
            'gate_result': node.gate_result,
            'review_verdict': node.review_verdict, 'review_reason': node.review_reason,
            'train_primary': node.train_primary,
            'val_gauc': node.val_gauc, 'val_ndcg5': node.val_ndcg5,
            'val_primary': node.val_primary,
            'train_val_gap': (round(node.train_primary - node.val_primary, 6)
                              if node.train_primary is not None
                              and node.val_primary is not None else None),
            'unbiased_val_primary': node.unbiased_val_primary,
            'delta_vs_baseline': (round(node.val_primary - config.BASELINE_VALID_PRIMARY, 6)
                                  if node.val_primary is not None else None),
            'diagnosis': diagnose.classify(node, self.journal),
            'seeds_averaged': node.seeds_averaged, 'seed_scores': node.seed_scores,
            'seed_unbiased_scores': node.seed_unbiased_scores,
            'accepted': node.accepted,
            'is_buggy': node.is_buggy, 'buggy_reason': node.buggy_reason,
            'exception_type': node.exception_type, 'recovery_action': node.recovery_action,
            'tokens_in': node.tokens_in, 'tokens_out': node.tokens_out,
            'exec_time': round(node.exec_time, 1),
            'wall_seconds': round(node.wall_seconds, 1),
        }
        with open(os.path.join(self.run_dir, 'log.jsonl'), 'a') as fh:
            fh.write(json.dumps(rec) + '\n')

    def _report(self, node: Node):
        if node.is_buggy:
            print(f'  -> BUGGY ({node.buggy_reason}) in {node.exec_time:.0f}s')
        elif node.val_primary is None:
            print(f'  -> completed in {node.exec_time:.0f}s (no score by design)')
        else:
            tag = 'ACCEPTED - new best' if node.accepted else 'rejected'
            gap = (f', train-val gap {node.train_primary - node.val_primary:+.4f}'
                   if node.train_primary is not None else '')
            unb = (f', unbiased {node.unbiased_val_primary:.4f}'
                   if node.unbiased_val_primary is not None else '')
            print(f'  -> primary {node.val_primary:.4f} '
                  f'(GAUC {node.val_gauc:.4f}, nDCG@5 {node.val_ndcg5:.4f}{gap}{unb}) '
                  f'[{node.seeds_averaged} seed(s)] {tag}')

    def run(self, max_iterations: int = None):
        max_iterations = max_iterations or config.MAX_ITERATIONS
        self.write_dummy_submissions()
        for i in range(max_iterations):
            if time.monotonic() - self.t0 > config.WALL_CLOCK_CEILING_S:
                print('\n[stop] wall-clock ceiling reached')
                break
            self.step(i)
            if self.journal.has_converged(config.EPSILON, config.CONVERGENCE_N,
                                          config.MIN_ITERATIONS_BEFORE_CONVERGENCE):
                self.converged_at = i
                print(f'\n[stop] converged at iteration {i}: validation primary has not '
                      f'improved by >{config.EPSILON} over {config.CONVERGENCE_N} '
                      f'scoring iterations')
                break
        else:
            print(f'\n[stop] iteration cap ({max_iterations}) reached')
        self.final_refit()
        self.write_summary()

    def final_refit(self):
        """Refit the winning configuration on train+valid and regenerate the test submission.

        The test window (04-29 to 05-08) directly follows validation, so the winning config
        trained on train+valid sees a week of more recent data than the one trained on train
        alone. This cannot be measured on validation -- validation is inside the training set
        once it is done -- which is exactly why a validation-driven search will never propose
        it: there is no gradient toward a change the objective cannot see.

        The scored, reported result stays the validation-best checkpoint. Only the test
        submission is regenerated, and only if the refit actually produces a valid one; any
        failure leaves the original in place, since a worse-but-valid submission beats a
        missing one.
        """
        best = self.journal.best
        solution = os.path.join(self.run_dir, 'best_solution.py')
        if best is None or not os.path.exists(solution):
            print('\n[refit] no best solution on disk -- skipped')
            self.refit_status = 'skipped: no best solution'
            return

        print(f'\n[refit] retraining node {best.id} on train+valid for the test submission')
        out_dir = os.path.join(self.run_dir, 'final_refit')
        res = executor.run_solution(solution, out_dir, seed=config.CONFIRM_SEEDS[0],
                                    require_metrics=False,
                                    extra_args=['--train_split', 'train+valid'])
        if res['is_buggy']:
            print(f'  -> refit failed ({res["buggy_reason"]}) -- keeping the train-only '
                  f'submission')
            self.refit_status = f'failed: {res["buggy_reason"]}'
            return

        src = os.path.join(out_dir, 'submission_test.csv')
        ok, msg = executor.check_submission(src, 'test')
        if not ok:
            print(f'  -> refit submission failed --check ({msg[:120]}) -- keeping the '
                  f'train-only submission')
            self.refit_status = 'failed: invalid submission'
            return

        shutil.copy(src, os.path.join(self.run_dir, 'best_submission_test.csv'))
        print('  -> test submission regenerated from the train+valid refit')
        self.refit_status = 'applied'

    def write_summary(self):
        best = self.journal.best
        scoring = self.journal.good_nodes
        summary = {
            'iterations_used': len(self.journal),
            'iteration_cap': config.MAX_ITERATIONS,
            'converged_at_iteration': self.converged_at,
            'convergence_epsilon': config.EPSILON,
            'convergence_n': config.CONVERGENCE_N,
            'min_iterations_before_convergence': config.MIN_ITERATIONS_BEFORE_CONVERGENCE,
            'manual_interventions': 0,
            'total_tokens_in': self.llm.total_in,
            'total_tokens_out': self.llm.total_out,
            'total_tokens': self.llm.total_in + self.llm.total_out,
            'cache_read_tokens': self.llm.cache_reads,
            'cache_write_tokens': self.llm.cache_writes,
            'literature_searches': self.llm.tool_calls,
            'papers_seen': len(self.journal.citation_registry),
            'agent_wall_clock_seconds': round(time.monotonic() - self.t0, 1),
            'gpu_hours': 0.0,
            'buggy_nodes': sum(1 for n in self.journal.nodes if n.is_buggy),
            # Selection pressure: the best of N noisy estimates is partly luck, so the count
            # is recorded to make that visible in the limitations reflection.
            'scoring_candidates_evaluated': len(scoring),
            'baseline_valid_primary': config.BASELINE_VALID_PRIMARY,
            'best_node_id': best.id if best else None,
            'best_valid_primary': best.val_primary if best else None,
            'best_valid_gauc': best.val_gauc if best else None,
            'best_valid_ndcg5': best.val_ndcg5 if best else None,
            'best_train_primary': best.train_primary if best else None,
            'best_unbiased_primary': best.unbiased_val_primary if best else None,
            'unbiased_gate_tolerance': self.journal.unbiased_tolerance,
            'draft_nodes': len(self.journal.drafts),
            # The reported score is the validation-best checkpoint; this says whether the test
            # submission on disk was regenerated from a train+valid refit of that same config.
            'final_refit_train_plus_valid': self.refit_status,
            'delta_vs_baseline': (round(best.val_primary - config.BASELINE_VALID_PRIMARY, 6)
                                  if best else None),
            'best_seeds_averaged': best.seeds_averaged if best else None,
            'accepted_trajectory': [
                {'iter': n.id, 'val_primary': n.val_primary,
                 'unbiased': n.unbiased_val_primary}
                for n in self.journal.nodes if n.accepted],
        }
        with open(os.path.join(self.run_dir, 'summary.json'), 'w') as fh:
            json.dump(summary, fh, indent=2)
        self.journal.to_jsonl(os.path.join(self.run_dir, 'journal.jsonl'))
        print('\n' + json.dumps(summary, indent=2))
