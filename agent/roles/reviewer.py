"""Check generated code for label leakage before it runs.

Leakage is the quietest way to lose: validation looks excellent, the hidden test score
collapses, and with one scored submission there is no second chance to notice. Nothing else in
the loop can see it -- a leaky solution produces a *better* validation number, so every other
gate waves it through.

A flagged solution still runs; it just cannot become the submission. Blocking execution
outright would let a false positive cost an iteration.
"""
from .. import config
from . import base

SYSTEM = """You review machine-learning code for one specific defect: label leakage.

The rule: for any row at time t, every feature must be computable using only information from
strictly before t.

This is LEAKAGE:
- statistics computed over train+validation+test together, or over all rows before splitting
- a target encoding fitted on the same rows it is then used to predict
- per-item or per-user aggregates that include the row being scored
- reading the label column of an evaluation row, directly or transitively
- features built from `video_features_statistic_pure.csv`, whose counters span the whole
  period including the test window

This is NOT leakage, and you must not flag it:
- statistics computed from the TRAINING split and applied to validation or test rows. This is
  the normal, correct way to build count and target-encoding features. It stays correct even
  though the same users, videos and authors appear in both splits -- what matters is that the
  statistic was computed only from data before the prediction window, not that the entities
  are disjoint.
- expanding-window or as-of-date statistics that exclude the current row
- using the training labels to fit model parameters. That is training, not leakage.
- **auxiliary training targets.** A multi-task model that predicts `long_view` *and* a second
  post-interaction column (`is_click`, `is_like`, `play_time_ms`, ...) from shared embeddings is
  correct, provided the auxiliary column is only ever a *target* and never an input feature. Ask
  what the model receives as input for the row it is scoring: if the answer does not include the
  auxiliary column, there is nothing to flag. Reading `is_click` out of the training rows to
  build a second loss term is training on labels, exactly as the line above allows. It becomes
  leakage only if the value is fed in as a feature, or if the auxiliary head's output for an
  evaluation row is computed from that row's own post-interaction values.
- the model scoring validation rows at evaluation time
- **the `--train_split train+valid` branch.** This mode runs once, after the run has converged,
  and its only output is `submission_test.csv`. Fitting statistics on train+valid and applying
  them to TEST rows is correct: the test window (20220429-0508) follows both. This branch must
  not score validation, and if it does not, there is nothing to flag. Only flag it if that
  branch computes a validation score that is reported as the run's result.
- **`data.encode()` receiving extra splits.** Read its source before judging it. It does
  `tr = splits['train']`, derives `edges` from `tr`, and builds `vocabs` from `tr` -- all
  before the `for name, rws in splits.items()` loop that encodes each split. Every output
  therefore depends on `splits['train']` alone. This was verified by running it: passing an
  extra `'rand'` key gives a byte-identical `dim`, train encoding and valid encoding. So
  `encode({'train': tr, 'valid': va, 'rand': rand})` is the intended, correct way to encode the
  random-exposure rows consistently, and is NOT leakage.

  The one case that IS a real defect: passing a *different* `splits['train']` to a second
  `encode()` call, for example one augmented with extra rows. That does change `dim` and every
  split's encoding (measured: 40260 vs 40696), so the model's weights no longer line up with
  the feature indices. Flag that -- but flag it as the index mismatch it is, and only when the
  train contents genuinely differ between the two calls.
- **reading feature VALUES from evaluation rows** to build a vocabulary, resolve an UNK slot,
  or align an encoding. Only the label column is off limits; feature values of rows that are
  about to be scored are known at scoring time by definition.

The distinction is *when the information comes from*, never *which entities recur*.

Reply in exactly this form, and nothing else -- no preamble, no analysis section, no headings:

VERDICT: CLEAN

or

VERDICT: LEAK
REASON: one or two sentences naming the specific line or computation and why it leaks.

Judge only leakage. Do not comment on style, efficiency, correctness, or whether the idea is
good. A false alarm blocks a legitimate experiment from ever becoming the submission, so flag
LEAK only when you can name the specific computation and say which future information it uses.

Two failure modes to guard against in your own judgement, both measured on real runs of this
harness where they cost half the iterations in a run:

1. Do not dress a correctness complaint as leakage. "This UNK handling looks wrong", "these
   statistics may be inconsistent", "this could break if retrained" are not leakage findings.
   If you cannot complete the sentence "this uses <specific information> from <a row at or
   after the scored row's time>", the verdict is CLEAN.
2. Do not flag on suspicion or on what code "may" or "could" do. If the answer depends on
   behaviour you have not established, read the relevant source in the prompt and settle it.
   Absent a concrete leaking computation you can point to, the verdict is CLEAN.

CLEAN is the correct answer for most solutions. Reserve LEAK for a defect you can name.
"""


def run(llm, code, spec, parent_code=None):
    """Returns (verdict, reason, tokens_in, tokens_out). Verdict is 'CLEAN' or 'LEAK'.

    When `parent_code` is given, the review is scoped to the diff. Reviewing the whole file
    every time meant the same unchanged scaffolding -- data loading, the random-exposure
    scoring, the submission writers -- was re-judged on every iteration, and one persistent
    misreading of it was enough to block a node no matter how many times its parent had been
    cleared. Eight of seventeen nodes in one run were blocked this way, all on scaffolding.
    A change can only introduce leakage through what it changed.
    """
    scope = ''
    if parent_code and parent_code.strip() != code.strip():
        import difflib
        diff = '\n'.join(difflib.unified_diff(
            parent_code.splitlines(), code.splitlines(),
            fromfile='parent.py', tofile='candidate.py', lineterm='', n=6))
        scope = f"""
# What actually changed (this is what you are judging)

The parent solution was already reviewed and cleared. Everything not in this diff is unchanged
from it and is NOT your concern -- do not flag it, however it looks. Judge only whether these
lines introduce leakage.

```diff
{diff[:12000]}
```
"""

    prompt = f"""# The change this code is meant to implement

{spec.proposed_change}

The planner assessed the leakage risk as: {spec.risks.get('leakage', 'not stated')}
{scope}
# The full file, for context

```python
{code}
```
"""
    text, ti, to = llm.complete(SYSTEM, prompt, config.REVIEWER_MODEL,
                                cached_prefix=base.TASK_DESCRIPTION, role='reviewer')

    upper = text.upper()
    verdict = 'LEAK' if 'VERDICT: LEAK' in upper else 'CLEAN'
    reason = ''
    for line in text.splitlines():
        if line.strip().upper().startswith('REASON:'):
            reason = line.split(':', 1)[1].strip()
            break
    if verdict == 'LEAK' and not reason:
        reason = text.strip()[:300]

    # A LEAK whose own reason concedes it is not leakage is a correctness complaint that reached
    # for the only verdict available to it. runs/v5 node 2 was blocked on a reason reading "This
    # is not strictly label leakage in the time dimension but rather a data mismatch" -- the
    # reviewer had already answered the question it was asked and returned LEAK regardless.
    # Blocking a node bars it from ever becoming the submission, so a self-contradicting finding
    # should not carry that cost.
    if verdict == 'LEAK' and _concedes_not_leakage(reason):
        print('  [review] LEAK reason concedes it is not leakage -- treating as clean')
        return 'CLEAN', '', ti, to
    if verdict == 'LEAK' and _is_the_refit_branch(reason):
        print('  [review] LEAK is about the train+valid refit branch -- treating as clean')
        return 'CLEAN', '', ti, to
    return verdict, reason, ti, to


_CONCESSIONS = (
    'not strictly label leakage', 'not strictly leakage', 'not technically leakage',
    'not label leakage', 'not leakage in the time', 'this is not leakage',
    'not a leakage', 'rather than leakage', 'not strictly a leak',
)


def _concedes_not_leakage(reason: str) -> bool:
    """Does the stated reason itself deny that the finding is leakage?

    Deliberately literal. It matches an explicit disavowal, not hedging words like "may" or
    "could" -- those appear in plenty of correctly-reported leaks, and a looser rule here would
    quietly wave real ones through, which is the one error this reviewer exists to prevent.
    """
    low = ' '.join((reason or '').lower().split())
    return any(c in low for c in _CONCESSIONS)


def _is_the_refit_branch(reason: str) -> bool:
    """Is the complaint that the refit branch fits on train+valid and scores test?

    That is the branch working as specified, and the SYSTEM prompt says so outright -- the test
    window (20220429-0508) follows validation (20220422-0428), so train+valid statistics are
    strictly past with respect to every test row. The reviewer contradicts that instruction
    anyway: it cost runs/v4 four of its eight LEAK verdicts, and in the v6 smoke run it blocked
    the best-scoring node with "validation rows must not be included in any aggregate used to
    score test rows", which is simply not the rule.

    Prompt wording has been tried three times against this and does not hold, so it is settled
    here instead. The reasoning is sound independently of the reviewer: this branch runs once
    after convergence and writes only submission_test.csv, so nothing it does can inflate the
    validation score that selects the submission. A finding that also alleges the branch scored
    validation is left alone -- that one is a real defect, and the only one available here.
    """
    low = ' '.join((reason or '').lower().split())
    about_refit = 'train+valid' in low or 'train + valid' in low or 'refit' in low
    about_test = 'test row' in low or 'test window' in low or 'test set' in low \
        or 'submission_test' in low or 'test score' in low
    alleges_val_scoring = ('scores validation' in low or 'scoring validation' in low
                           or 'validation score' in low or 'reports a validation' in low
                           or 'val_primary' in low)
    return about_refit and about_test and not alleges_val_scoring
