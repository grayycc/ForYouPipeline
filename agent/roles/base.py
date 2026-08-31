"""Shared pieces every role needs: the cacheable task prefix, the output contract, and the
fence-aware response parsing.

The section parser has to be fence-aware because inside a ```python block a '#' starts a
Python comment, not a markdown header. Treating one as a header truncates the generated file
mid-way and strips its closing fence, which produced a SyntaxError on line 1 of every solution
until it was fixed.
"""
import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(_HERE, 'task_description.md')) as _fh:
    TASK_DESCRIPTION = _fh.read()

CONTRACT_REMINDER = """
Output contract: accept --data_dir/--out_dir/--seed; print TRAIN_PRIMARY=, VAL_GAUC=,
VAL_NDCG5=, VAL_PRIMARY= and UNBIASED_PRIMARY= each on their own line; write
submission_valid.csv and submission_test.csv into --out_dir; standard library and numpy only.

Feature-generation rule: if you add new signals, create them inside the standalone solution
itself using training-only history and the public metadata files. The agent is responsible for
proposing the feature logic, not just tuning a fixed repo pipeline. Prefer causal preference
signals (user × category affinity, user-topic matching, temporal history) over exposure-driven
popularity proxies. Evaluate any improvement against both validation and the random-exposure set;
features that help only ordinary logged traffic but reduce the unbiased score are rejected.
"""

CODE_SYSTEM = """You write complete, runnable Python for a machine-learning experiment.

Reply with exactly two sections and nothing else:

# Hypothesis
One paragraph: the single change you are making, the mechanism by which you expect it to move
GAUC or nDCG@5, and the size of effect you expect. Name any published method you draw on. Be
specific about the mechanism -- "this might help" is not a hypothesis.

# Code
One complete, standalone Python file in a single ```python block. No prose, no partial
snippets, no diffs -- the file is written to disk and executed exactly as given.
"""


def extract_section(text: str, name: str) -> str:
    """Pull a '# Name' section out of a response, ignoring '#' inside fenced code blocks."""
    cur, buf, in_fence = None, [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('```'):
            in_fence = not in_fence
            if cur == name:
                buf.append(line)
            continue
        if not in_fence and s.startswith('#'):
            header = s.lstrip('#').strip().lower()
            if header.startswith(name.lower()):
                cur = name
                continue
            if cur == name:
                break                      # the next section begins
            continue                       # a header belonging to some other section
        if cur == name:
            buf.append(line)
    return '\n'.join(buf).strip()


def budget_line(iteration, journal, config) -> str:
    best = journal.best
    best_s = f'{best.val_primary:.4f} (node {best.id})' if best else 'none yet'
    return (f'Iteration {iteration} of {config.MAX_ITERATIONS}. '
            f'Best validation primary so far: {best_s}. '
            f'Baseline to beat: {config.BASELINE_VALID_PRIMARY:.4f} validation. '
            f'Noise floor: {config.SEED_STD:.4f} std, so gains under 0.002 are not real.')
