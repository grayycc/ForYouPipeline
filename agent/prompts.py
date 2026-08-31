"""Prompt construction for each operation.

Rules that matter: send a *summary* of prior attempts, never the whole tree; always
demand one atomic change; always demand an explicit hypothesis with a mechanism.
"""
import os

from . import config

_TASK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_description.md')
with open(_TASK_PATH) as _fh:
    TASK_DESCRIPTION = _fh.read()

SYSTEM = """You are an autonomous ML research agent working on a within-user ranking task.

You respond with exactly two sections and nothing else:

# Hypothesis
One paragraph. State the single change you are making, the mechanism by which you expect it
to move GAUC or nDCG@5, and the size of effect you expect. If you are drawing on a known
published method, name it. Be specific about the mechanism -- "this might help" is not a
hypothesis.

# Code
One complete, standalone Python file in a single ```python block. No prose, no partial
snippets, no diffs -- the file is written to disk and executed exactly as given.
"""

_CONTRACT_REMINDER = """
Remember the output contract: accept --data_dir/--out_dir/--seed and the optional
--train_split {train,train+valid} (which trains on both splits and writes ONLY
submission_test.csv -- it must not score validation), print TRAIN_PRIMARY=,
VAL_GAUC=, VAL_NDCG5=, VAL_PRIMARY= and UNBIASED_PRIMARY= on their own lines, write
submission_valid.csv and submission_test.csv into --out_dir. All five metric lines are
required: UNBIASED_PRIMARY drives the acceptance gate and TRAIN_PRIMARY is how overfitting
becomes visible, so a solution that omits them cannot be judged. UNBIASED_PRIMARY must be
computed on the random-exposure log -- a node whose UNBIASED_PRIMARY equals its VAL_PRIMARY is
rejected as a stub.

Importable: numpy, scipy, pandas, scikit-learn, lightgbm, xgboost, torch. Nothing else is
installed and there is no network or GPU. torch must not appear in the same script as lightgbm
or xgboost -- they clash over OpenMP and the process segfaults with no traceback.

Important: generate any new feature logic inside the standalone solution itself. Do not treat
this as a fixed repo pipeline. Any statistic you build from the training split must be usable
by the model at evaluation time in the same form it took at training time.

Use the random-exposure score as an anti-bias gate: a feature that helps normal validation
while hurting the unbiased score is not an improvement.
"""


def _budget_line(iteration, journal):
    best = journal.best
    best_s = f'{best.val_primary:.4f} (node {best.id})' if best else 'none yet'
    return (f'Iteration {iteration} of {config.MAX_ITERATIONS}. '
            f'Best validation primary so far: {best_s}. '
            f'Baseline to beat: {config.BASELINE_VALID_PRIMARY:.4f} validation. '
            f'Noise floor: {config.SEED_STD:.4f} std, so gains under 0.002 are not real.')


def baseline_prompt(iteration, journal):
    """Iteration 0: the agent stands up its own pipeline and verifies it hits the baseline."""
    return f"""{TASK_DESCRIPTION}

---
{_budget_line(iteration, journal)}

# Your task right now

Stand up a working end-to-end pipeline from scratch and confirm it reproduces the official
FM baseline's reported validation score of {config.BASELINE_VALID_PRIMARY:.4f}.

Write the training code yourself rather than importing `run_fm` from `baseline.py` -- you
need your own pipeline to build on in later iterations, and you need to have verified that
it produces the expected number. You may read `baseline.py` for the model definition and
import `FM` from it if that is the cleanest way to match the reference implementation.

Print how close you landed to {config.BASELINE_VALID_PRIMARY:.4f} and treat anything within
the noise floor as a successful reproduction.
{_CONTRACT_REMINDER}"""


def draft_prompt(iteration, journal):
    from . import diagnose
    # A draft is written from a blank file, so it needs the accumulated context *more* than an
    # improve does, not less -- an improve at least inherits a working solution. runs/v3's three
    # drafts got only the prior-attempts table and all three failed hard (train 0.97 / val 0.44
    # in two of them), reinventing the target-encoding overfit the EDA pass had already
    # characterised.
    context = ''
    # Drafts were the one path that never saw the cross-run record, and it showed: runs/v10's
    # only draft opened with gradient-boosted trees, a family that is 0-accepted from 15
    # attempts across 7 runs. A draft picks a mechanism from a blank file, so it needs the
    # standing evidence at least as much as an improve does.
    if getattr(journal, 'cross_run_yield', ''):
        context += f'\n{journal.cross_run_yield}\n'
    if journal.eda_findings:
        context += f'\n# Measured properties of this data\n\n{journal.eda_findings}\n'
    ruled_out = diagnose.ruled_out_block(journal)
    if ruled_out:
        context += f'\n{ruled_out}\n'
    coverage = diagnose.stack_coverage(journal)
    if coverage:
        context += f'\n{coverage}\n'

    return f"""{TASK_DESCRIPTION}

---
{_budget_line(iteration, journal)}

# Prior attempts

{journal.summary_table()}
{context}
# Your task right now

Write a fresh solution from scratch, taking a different angle from the attempts above.
This is a *draft*: you are exploring a genuinely different approach, not tuning an existing
one. Ground your choice in the metric definitions and the structure of the data.

A draft costs a whole iteration and starts from nothing, so it earns its place only by
testing a mechanism the run has not tested. Restating an approach already listed above in
new words is the most expensive way to learn nothing.

Answer in three sections, with these exact headers:

# Hypothesis
The claim being tested, stated so that a result could show it is wrong.

# Tags
Two to four short hyphenated tags naming the *mechanism*, comma-separated on one line --
e.g. `gbdt, marginal-ctr-features, pointwise-loss`. These are how the run detects that an
idea has already been tried, so they must describe the mechanism itself rather than the
framing: two drafts proposing the same thing should collide on tags even when their prose
differs completely.

# Code
The full solution.
{_CONTRACT_REMINDER}"""


def improve_prompt(iteration, journal, node):
    return f"""{TASK_DESCRIPTION}

---
{_budget_line(iteration, journal)}

# Prior attempts

{journal.summary_table()}

# The current best solution (validation primary {node.val_primary:.4f})

```python
{node.code}
```

Its output:
```
{node.stdout_tail}
```

# Your task right now

Propose **a single atomic, actionable improvement** to the solution above, so that its
effect can be experimentally attributed. Change one thing. Do not bundle several ideas into
one iteration -- if you have several, pick the one with the best expected-value-per-iteration
and say why in your hypothesis.

Return the complete modified file, not a diff.
{_CONTRACT_REMINDER}"""


def debug_prompt(iteration, journal, node):
    return f"""{TASK_DESCRIPTION}

---
{_budget_line(iteration, journal)}

# A solution failed and needs fixing

Failure mode: **{node.buggy_reason}**{f' ({node.exception_type})' if node.exception_type else ''}
This is fix attempt {node.debug_depth + 1} of {config.MAX_DEBUG_DEPTH} on this branch; after
that the branch is abandoned.

Its hypothesis was:
{node.hypothesis}

```python
{node.code}
```

stdout tail:
```
{node.stdout_tail}
```

error:
```
{node.stderr_tail}
```

# Your task right now

Diagnose the actual cause and return the complete corrected file. Fix the bug -- do not
redesign the approach and do not quietly drop the idea being tested. If the error shows the
idea itself is unworkable as written, say so explicitly in your hypothesis and implement the
smallest change that makes it run.
{_CONTRACT_REMINDER}"""


REFLECT_SYSTEM = """You analyse the result of one ML experiment. Reply with 2-4 sentences of
plain prose, no headers or lists. Say whether the result matched the hypothesis and by how
much; if it did not, say whether the cause was a bug, a wrong idea, or a difference inside
the noise floor; then say what that implies for what to try next, and what is now ruled out."""


def reflect_prompt(node, journal, baseline_primary):
    best = journal.best
    if node.is_buggy:
        outcome = f'FAILED: {node.buggy_reason} ({node.exception_type})\n\nerror:\n{node.stderr_tail}'
    else:
        delta_base = node.val_primary - baseline_primary
        delta_best = node.val_primary - best.val_primary if best else 0.0
        outcome = (f'validation primary {node.val_primary:.4f} '
                   f'(GAUC {node.val_gauc:.4f}, nDCG@5 {node.val_ndcg5:.4f})\n'
                   f'delta vs official baseline: {delta_base:+.4f}\n'
                   f'delta vs best so far: {delta_best:+.4f}\n'
                   f'seeds averaged: {node.seeds_averaged}'
                   + (f'\nseed spread: {node.seed_scores}' if len(node.seed_scores) > 1 else '')
                   + (f'\nunbiased (random-exposure) primary: {node.unbiased_val_primary:.4f}'
                      if node.unbiased_val_primary is not None else ''))
    return f"""Hypothesis that was tested:
{node.hypothesis}

Outcome:
{outcome}

Noise floor: a difference smaller than 0.002 is not distinguishable from seed noise.
"""
