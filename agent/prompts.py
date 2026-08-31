"""Prompt construction for each operation.

Rules that matter: send a *summary* of prior attempts, never the whole tree; always
demand one atomic change; always demand an explicit hypothesis with a mechanism.
"""
import os

from . import config

_TASK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_description.md')
with open(_TASK_PATH) as _fh:
    TASK_DESCRIPTION = _fh.read()


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
