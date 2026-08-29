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
import time

from . import config, executor, gates
from .journal import Journal, Node
from .llm import LLMClient, LLMError
from .roles import baseline as baseline_role
from .roles import coder as coder_role
from .roles import debugger as debugger_role
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
        self.t0 = time.time()
        self.converged_at = None

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
        """baseline once, then EDA once, then debug a broken leaf or improve the best."""
        j = self.journal
        if not j.has_operation('baseline'):
            return 'baseline', None
        if not j.has_operation('eda'):
            return 'eda', None
        eligible = [n for n in j.buggy_leaves() if n.debug_depth < config.MAX_DEBUG_DEPTH]
        if eligible and self.rng.random() < config.DEBUG_PROBABILITY:
            return 'debug', self.rng.choice(eligible)
        return 'improve', j.best

    def step(self, iteration: int) -> Node:
        t_iter = time.time()
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
            elif op == 'debug':
                self._generate_debug(node, iteration, parent)
            else:
                self._generate_experiment(node, iteration, parent)
        except LLMError as e:
            node.is_buggy, node.buggy_reason = True, 'llm_failure'
            node.exception_type, node.stderr_tail = 'LLMError', str(e)
            node.recovery_action = 'fall back to the best node next iteration'
            node.wall_seconds = time.time() - t_iter
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

        node.accepted = self._accept(node)
        if node.accepted:
            self._promote(node, node_dir)

        self._finish(node, t_iter)
        return node

    def _finish(self, node, t_iter):
        node.wall_seconds = time.time() - t_iter
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

    def _generate_debug(self, node, iteration, parent):
        node.spec = parent.spec
        node.hypothesis, node.code, ti, to = debugger_role.run(
            self.llm, iteration, self.journal, parent)
        node.tokens_in, node.tokens_out = ti, to

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

        verdict, reason, ti4, to4 = reviewer_role.run(self.llm, node.code, spec)
        node.tokens_in += ti4
        node.tokens_out += to4
        node.review_verdict, node.review_reason = verdict, reason
        if verdict == 'LEAK':
            print(f'  [review] leakage flagged: {reason[:120]}')
            node.code, ti5, to5 = coder_role.revise(self.llm, spec, node.code, reason)
            node.tokens_in += ti5
            node.tokens_out += to5
            verdict2, reason2, ti6, to6 = reviewer_role.run(self.llm, node.code, spec)
            node.tokens_in += ti6
            node.tokens_out += to6
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

    def _worth_confirming(self, node: Node) -> bool:
        best = self.journal.best
        if best is None:
            return True
        return node.val_primary > best.val_primary - config.CONFIRM_TRIGGER

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
        if len(node.seed_scores) > 1:
            node.val_primary = sum(node.seed_scores) / len(node.seed_scores)
            node.seeds_averaged = len(node.seed_scores)
            print(f'    seeds [{" ".join(f"{x:.4f}" for x in node.seed_scores)}] '
                  f'-> mean {node.val_primary:.4f}')

    def _accept(self, node: Node) -> bool:
        if node.is_buggy or node.val_primary is None:
            return False
        if node.review_verdict == 'LEAK':
            node.recovery_action = 'blocked from best by the leakage reviewer'
            return False
        best = self.journal.best
        if best is None:
            return True
        if node.val_primary <= best.val_primary:
            return False
        # Reject gains that exist only on policy-biased traffic.
        if node.unbiased_val_primary is not None and best.unbiased_val_primary is not None:
            if node.unbiased_val_primary < best.unbiased_val_primary - config.SEED_STD:
                node.recovery_action = 'rejected by the unbiased-exposure gate'
                print('  [gate] validation rose but random-exposure fell -- rejected as '
                      'overfitting to the logging policy')
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
            'seeds_averaged': node.seeds_averaged, 'seed_scores': node.seed_scores,
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
            if time.time() - self.t0 > config.WALL_CLOCK_CEILING_S:
                print('\n[stop] wall-clock ceiling reached')
                break
            self.step(i)
            if self.journal.has_converged(config.EPSILON, config.CONVERGENCE_N):
                self.converged_at = i
                print(f'\n[stop] converged at iteration {i}: validation primary has not '
                      f'improved by >{config.EPSILON} over {config.CONVERGENCE_N} '
                      f'scoring iterations')
                break
        else:
            print(f'\n[stop] iteration cap ({max_iterations}) reached')
        self.write_summary()

    def write_summary(self):
        best = self.journal.best
        scoring = self.journal.good_nodes
        summary = {
            'iterations_used': len(self.journal),
            'iteration_cap': config.MAX_ITERATIONS,
            'converged_at_iteration': self.converged_at,
            'manual_interventions': 0,
            'total_tokens_in': self.llm.total_in,
            'total_tokens_out': self.llm.total_out,
            'total_tokens': self.llm.total_in + self.llm.total_out,
            'cache_read_tokens': self.llm.cache_reads,
            'cache_write_tokens': self.llm.cache_writes,
            'literature_searches': self.llm.tool_calls,
            'papers_seen': len(self.journal.citation_registry),
            'agent_wall_clock_seconds': round(time.time() - self.t0, 1),
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
