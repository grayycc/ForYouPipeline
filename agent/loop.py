"""The agent loop: select -> generate -> execute -> validate -> confirm -> reflect -> record.

Greedy search over the solution tree. Improve always branches from the current best node, so
a bad idea costs one node rather than the run, and the search can never get permanently stuck.
"""
import json
import os
import random
import shutil
import time

from . import config, executor, prompts
from .journal import Journal, Node
from .llm import LLMClient, LLMError


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
            rows = splits[split]
            path = os.path.join(self.run_dir, f'best_submission_{split}.csv')
            if not os.path.exists(path):
                write_submission(path, rows, [0.0] * len(rows))
        print(f'  [safety] placeholder submissions written to {self.run_dir}')

    def select(self):
        j = self.journal
        if not j.good_nodes and not any(n.operation == 'baseline' for n in j.nodes):
            return 'baseline', None
        buggy = j.buggy_leaves()
        eligible = [n for n in buggy if n.debug_depth < config.MAX_DEBUG_DEPTH]
        if len(j.drafts) < config.MIN_DRAFTS and j.best is not None:
            if eligible and self.rng.random() < config.DEBUG_PROBABILITY:
                return 'debug', self.rng.choice(eligible)
            return 'draft', None
        if eligible and self.rng.random() < config.DEBUG_PROBABILITY:
            return 'debug', self.rng.choice(eligible)
        best = j.best
        if best is None:
            return 'draft', None
        return 'improve', best

    def step(self, iteration: int) -> Node:
        t_iter = time.time()
        op, parent = self.select()
        print(f'\n=== iteration {iteration} | {op}'
              f'{f" from node {parent.id}" if parent else ""} ===')

        prompt = {
            'baseline': lambda: prompts.baseline_prompt(iteration, self.journal),
            'draft': lambda: prompts.draft_prompt(iteration, self.journal),
            'improve': lambda: prompts.improve_prompt(iteration, self.journal, parent),
            'debug': lambda: prompts.debug_prompt(iteration, self.journal, parent),
        }[op]()
        use_strong = op in ('baseline', 'draft') or (
            op == 'debug' and parent.debug_depth >= 1)

        node = Node(id=iteration, parent_id=parent.id if parent else None, operation=op,
                    debug_depth=(parent.debug_depth + 1) if op == 'debug' else 0)

        # Ask the model for a hypothesis and a complete solution file.
        try:
            text, ti, to = self.llm.complete(prompts.SYSTEM, prompt, strong=use_strong)
            node.tokens_in, node.tokens_out = ti, to
        except LLMError as e:
            node.is_buggy, node.buggy_reason = True, 'llm_failure'
            node.exception_type, node.stderr_tail = 'LLMError', str(e)
            node.recovery_action = 'fall back to best node next iteration'
            node.wall_seconds = time.time() - t_iter
            return self.journal.append(node)

        node.hypothesis = self._extract_section(text, 'Hypothesis')
        node.code = executor.strip_fences(self._extract_section(text, 'Code') or text)

        node_dir = os.path.join(self.nodes_dir, f'node_{iteration}')
        os.makedirs(node_dir, exist_ok=True)
        code_path = os.path.join(node_dir, 'solution.py')
        with open(code_path, 'w') as fh:
            fh.write(node.code)

        # Run it in a sandboxed subprocess.
        res = executor.run_solution(code_path, node_dir, seed=config.CONFIRM_SEEDS[0])
        self._apply_result(node, res)
        with open(os.path.join(node_dir, 'stdout.txt'), 'w') as fh:
            fh.write(res['stdout_tail'] + '\n--- stderr ---\n' + res['stderr_tail'])

        # A node whose submission is invalid is buggy, whatever it scored.
        if not node.is_buggy:
            ok_v, msg_v = executor.check_submission(
                os.path.join(node_dir, 'submission_valid.csv'), 'valid')
            ok_t, msg_t = executor.check_submission(
                os.path.join(node_dir, 'submission_test.csv'), 'test')
            node.submission_ok = ok_v and ok_t
            if not node.submission_ok:
                node.is_buggy, node.buggy_reason = True, 'invalid_submission'
                node.stderr_tail = (node.stderr_tail + '\n' + msg_v + '\n' + msg_t).strip()

        # Confirm promising candidates across seeds so noise is not mistaken for a gain.
        if not node.is_buggy and self._worth_confirming(node):
            self._confirm_seeds(node, code_path, node_dir)

        # Accept or reject against the current best.
        node.accepted = self._accept(node)
        if node.accepted:
            self._promote(node, node_dir)

        # Record why it worked or failed, to carry into the next prompt.
        node.reflection = self._reflect(node)
        node.wall_seconds = time.time() - t_iter
        self.journal.append(node)
        self._log(node)
        self._report(node)
        return node

    @staticmethod
    def _extract_section(text: str, name: str) -> str:
        """Pull '# Hypothesis' / '# Code' sections out of the response.

        Must be fence-aware: inside a ```python block, '#' starts a Python comment, not a
        markdown header. Treating one as a header truncates the generated file mid-way and
        strips its closing fence.
        """
        cur, buf, in_fence = None, [], False
        for ln in text.splitlines():
            s = ln.strip()
            if s.startswith('```'):
                in_fence = not in_fence
                if cur == name:
                    buf.append(ln)
                continue
            if not in_fence and s.startswith('#'):
                header = s.lstrip('#').strip().lower()
                if header.startswith(name.lower()):
                    cur = name
                    continue
                if cur == name:
                    break                      # next section begins
                continue                       # a header for some other section
            if cur == name:
                buf.append(ln)
        return '\n'.join(buf).strip()

    def _apply_result(self, node: Node, res: dict):
        m = res['metrics']
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
        """Re-run a promising candidate on the remaining seeds and average."""
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
            spread = ' '.join(f'{x:.4f}' for x in node.seed_scores)
            print(f'    seeds [{spread}] -> mean {node.val_primary:.4f}')

    def _accept(self, node: Node) -> bool:
        if node.is_buggy or node.val_primary is None:
            return False
        best = self.journal.best
        if best is None:
            return True
        if node.val_primary <= best.val_primary:
            return False
        # Reject gains that exist only on policy-biased traffic.
        if node.unbiased_val_primary is not None and best.unbiased_val_primary is not None:
            if node.unbiased_val_primary < best.unbiased_val_primary - config.SEED_STD:
                node.recovery_action = 'rejected by unbiased-exposure gate'
                print('  [gate] validation rose but random-exposure score fell -- rejected '
                      'as overfitting to the logging policy')
                return False
        return True

    def _promote(self, node: Node, node_dir: str):
        for split in ('valid', 'test'):
            src = os.path.join(node_dir, f'submission_{split}.csv')
            if os.path.exists(src):
                shutil.copy(src, os.path.join(self.run_dir, f'best_submission_{split}.csv'))
        shutil.copy(os.path.join(node_dir, 'solution.py'),
                    os.path.join(self.run_dir, 'best_solution.py'))

    def _reflect(self, node: Node) -> str:
        try:
            text, ti, to = self.llm.complete(
                prompts.REFLECT_SYSTEM,
                prompts.reflect_prompt(node, self.journal, config.BASELINE_VALID_PRIMARY),
                strong=False)
            node.tokens_in += ti
            node.tokens_out += to
            return text.strip()
        except LLMError as e:
            return f'(reflection unavailable: {e})'

    def _log(self, node: Node):
        rec = {
            'iter': node.id, 'parent_id': node.parent_id, 'operation': node.operation,
            'hypothesis': node.hypothesis, 'val_gauc': node.val_gauc,
            'val_ndcg5': node.val_ndcg5, 'val_primary': node.val_primary,
            'unbiased_val_primary': node.unbiased_val_primary,
            'delta_vs_baseline': (round(node.val_primary - config.BASELINE_VALID_PRIMARY, 6)
                                  if node.val_primary is not None else None),
            'seeds_averaged': node.seeds_averaged, 'seed_scores': node.seed_scores,
            'accepted': node.accepted, 'reflection': node.reflection,
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
        else:
            tag = 'ACCEPTED - new best' if node.accepted else 'rejected'
            print(f'  -> primary {node.val_primary:.4f} '
                  f'(GAUC {node.val_gauc:.4f}, nDCG@5 {node.val_ndcg5:.4f}) '
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
                      f'improved by >{config.EPSILON} over {config.CONVERGENCE_N} iterations')
                break
        else:
            print(f'\n[stop] iteration cap ({max_iterations}) reached')
        self.write_summary()

    def write_summary(self):
        best = self.journal.best
        summary = {
            'iterations_used': len(self.journal),
            'iteration_cap': config.MAX_ITERATIONS,
            'converged_at_iteration': self.converged_at,
            'manual_interventions': 0,
            'total_tokens_in': self.llm.total_in,
            'total_tokens_out': self.llm.total_out,
            'total_tokens': self.llm.total_in + self.llm.total_out,
            'agent_wall_clock_seconds': round(time.time() - self.t0, 1),
            'gpu_hours': 0.0,
            'buggy_nodes': sum(1 for n in self.journal.nodes if n.is_buggy),
            'baseline_valid_primary': config.BASELINE_VALID_PRIMARY,
            'best_node_id': best.id if best else None,
            'best_valid_primary': best.val_primary if best else None,
            'best_valid_gauc': best.val_gauc if best else None,
            'best_valid_ndcg5': best.val_ndcg5 if best else None,
            'delta_vs_baseline': (round(best.val_primary - config.BASELINE_VALID_PRIMARY, 6)
                                  if best else None),
            'best_seeds_averaged': best.seeds_averaged if best else None,
        }
        with open(os.path.join(self.run_dir, 'summary.json'), 'w') as fh:
            json.dump(summary, fh, indent=2)
        self.journal.to_jsonl(os.path.join(self.run_dir, 'journal.jsonl'))
        print('\n' + json.dumps(summary, indent=2))
