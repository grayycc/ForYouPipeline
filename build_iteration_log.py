#!/usr/bin/env python3
"""Render a run's log.jsonl into the per-iteration report the deliverable spec asks for.

Run & Iteration Logs needs, per iteration: the hypothesis, "the code diff applied", the
resulting metrics, and any error/recovery events. log.jsonl already has the first, third and
fourth; the second is where a gap was real. `proposed_change` is a prose description the
planner writes before the code exists, not a diff of what the coder actually produced, and only
the single winning node's code (best_solution.py) is tracked in git -- every intermediate
iteration's file lives only in the gitignored runs/<id>/nodes/ directory, which is disk-only.

This computes an actual unified diff against each node's real parent, from the real files, so
the report says what changed rather than what was intended. It does not edit or regenerate
anything the agent produced -- read-only over an existing run.

    python3 build_iteration_log.py v10 > runs/v10/ITERATION_LOG.md

Requires runs/<id>/nodes/*/solution.py to exist on disk (gitignored, not deleted by a run) --
this has to be built once per run you want documented, from a machine that still has them.
"""
import argparse
import difflib
import json
import os
import sys


def node_dir(run_dir, node_id):
    return os.path.join(run_dir, 'nodes', f'node_{node_id}')


def read_code(run_dir, node_id):
    path = os.path.join(node_dir(run_dir, node_id), 'solution.py')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return fh.read()


def fmt_metric(v, digits=4):
    return f'{v:.{digits}f}' if isinstance(v, (int, float)) else '—'


def render_node(d, run_dir):
    lines = [f"## Node {d['iter']} — `{d['operation']}`"
             + (f" (parent: node {d['parent_id']})" if d.get('parent_id') is not None else '')]

    hyp = (d.get('hypothesis') or '').strip()
    if hyp:
        lines += ['', '**Hypothesis:**', '', hyp]

    if d.get('is_buggy'):
        lines += ['', f"**Error:** `{d.get('buggy_reason')}`"
                 + (f" ({d['exception_type']})" if d.get('exception_type') else ''),
                 f"**Recovery:** {d.get('recovery_action') or '(none recorded)'}"]
    elif d.get('val_primary') is not None:
        lines += ['', '**Metrics:**', '',
                 f"| train_primary | val_gauc | val_ndcg5 | val_primary | unbiased | diagnosis |",
                 f"|---|---|---|---|---|---|",
                 f"| {fmt_metric(d.get('train_primary'))} | {fmt_metric(d.get('val_gauc'))} | "
                 f"{fmt_metric(d.get('val_ndcg5'))} | **{fmt_metric(d.get('val_primary'))}** | "
                 f"{fmt_metric(d.get('unbiased_val_primary'))} | {d.get('diagnosis')} |",
                 '',
                 f"**Accepted:** {'yes — became the new best' if d.get('accepted') else 'no'}"]
        if d.get('review_verdict'):
            lines.append(f"**Leakage review:** {d['review_verdict']}"
                         + (f" — {d['review_reason']}" if d.get('review_reason') else ''))
        if d.get('recovery_action'):
            lines.append(f"**Note:** {d['recovery_action']}")
    else:
        lines += ['', f"**Result:** no score by design ({d['operation']})"]

    code = read_code(run_dir, d['iter'])
    parent_code = read_code(run_dir, d['parent_id']) if d.get('parent_id') is not None else None

    if code is None:
        lines += ['', '_No code file on disk for this node (see error above)._']
    elif parent_code is None:
        n_lines = len(code.splitlines())
        lines += ['', f"**Code diff:** none — fresh file, no parent ({n_lines} lines). "
                     f"See `nodes/node_{d['iter']}/solution.py`."]
    elif code == parent_code:
        lines += ['', '**Code diff:** none — byte-identical to parent.']
    else:
        diff = list(difflib.unified_diff(
            parent_code.splitlines(keepends=True), code.splitlines(keepends=True),
            fromfile=f'node_{d["parent_id"]}/solution.py',
            tofile=f'node_{d["iter"]}/solution.py'))
        added = sum(1 for l in diff if l.startswith('+') and not l.startswith('+++'))
        removed = sum(1 for l in diff if l.startswith('-') and not l.startswith('---'))
        lines += ['', f'**Code diff** (+{added}/-{removed} lines vs. node {d["parent_id"]}):',
                 '', '```diff', ''.join(diff).rstrip('\n'), '```']

    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_id')
    ap.add_argument('--runs_dir', default='runs')
    args = ap.parse_args()

    run_dir = os.path.join(args.runs_dir, args.run_id)
    log_path = os.path.join(run_dir, 'log.jsonl')
    if not os.path.exists(log_path):
        sys.exit(f'no such run: {log_path}')

    nodes = [json.loads(line) for line in open(log_path)]

    print(f'# Iteration log — run `{args.run_id}`\n')
    print(f'{len(nodes)} nodes. Each code diff below is computed from the actual files this '
         f'run wrote, against the actual parent it branched from -- not a description.\n')
    for d in nodes:
        print(render_node(d, run_dir))
        print()


if __name__ == '__main__':
    main()
