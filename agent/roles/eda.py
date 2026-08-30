"""One pass of data inspection, whose findings ground every later hypothesis.

It runs once, not per iteration: the train split does not change, so re-measuring it would
spend the scarce resource on the same numbers. The findings are cached into every planner
call, and anything this pass missed can be printed alongside a later experiment.

Exempt from the submission and metric contract -- a clean exit is success.
"""
from .. import config
from . import base

SYSTEM = """You are a data analyst. Reply with exactly two sections and nothing else:

# Hypothesis
One paragraph on what you intend to measure and why those measurements would change how a
later experiment is chosen. Not a modelling hypothesis -- an analysis rationale.

# Code
One complete, standalone Python file in a single ```python block that prints its findings.
"""


def run(llm, iteration, journal):
    """Returns (hypothesis, code, tokens_in, tokens_out)."""
    prompt = f"""{base.budget_line(iteration, journal, config)}

# Your task right now

Write a script that inspects the data and prints what you judge most useful for choosing later
experiments. You have one pass; its printed output is the only thing carried forward, and it
will be shown to you at every subsequent iteration.

This script is exempt from the usual contract: it takes --data_dir/--out_dir/--seed, but needs
no submission files, no model, and no VAL_ metrics. Print findings as compact labelled lines,
not raw dumps -- the output is truncated to about 3000 characters, so spend it deliberately.

Keep the runtime modest; this costs one of your {config.MAX_ITERATIONS} iterations.

Importable here: numpy, pandas, scipy, scikit-learn, lightgbm, torch, and the standard
library. Nothing else is installed, and an import outside that list costs the iteration.
"""
    text, ti, to = llm.complete(SYSTEM, prompt, config.EDA_MODEL,
                                cached_prefix=base.TASK_DESCRIPTION, role='eda')
    from ..executor import strip_fences
    code = strip_fences(base.extract_section(text, 'Code') or text)
    code, ti, to = base.retry_if_broken(llm, code, SYSTEM, config.EDA_MODEL, 'eda', ti, to)
    return base.extract_section(text, 'Hypothesis'), code, ti, to
