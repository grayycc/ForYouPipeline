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
- the model scoring validation rows at evaluation time

The distinction is *when the information comes from*, never *which entities recur*.

Reply in exactly this form, and nothing else -- no preamble, no analysis section, no headings:

VERDICT: CLEAN

or

VERDICT: LEAK
REASON: one or two sentences naming the specific line or computation and why it leaks.

Judge only leakage. Do not comment on style, efficiency, correctness, or whether the idea is
good. A false alarm blocks a legitimate experiment from ever becoming the submission, so flag
LEAK only when you can name the specific computation and say which future information it uses.
"""


def run(llm, code, spec):
    """Returns (verdict, reason, tokens_in, tokens_out). Verdict is 'CLEAN' or 'LEAK'."""
    prompt = f"""# The change this code is meant to implement

{spec.proposed_change}

The planner assessed the leakage risk as: {spec.risks.get('leakage', 'not stated')}

# The code

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
    return verdict, reason, ti, to
