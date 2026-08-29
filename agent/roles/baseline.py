"""Iteration 0: the agent stands up its own pipeline and confirms it reaches the baseline.

This is an LLM call rather than the harness simply running kit/baseline.py, for two reasons.
Task Requirement #1 asks the *agent* to build the pipeline and verify it, and doing that by
hand forfeits the credit. More practically, kit/ is read-only, so without this the tree has no
root the Coder is allowed to edit -- every later iteration would have to write a full pipeline
from scratch anyway.
"""
from .. import config
from . import base


def run(llm, iteration, journal):
    """Returns (hypothesis, code, tokens_in, tokens_out)."""
    prompt = f"""{base.budget_line(iteration, journal, config)}

# Your task right now

Stand up a working end-to-end pipeline from scratch and confirm it reproduces the official FM
baseline's reported validation score of {config.BASELINE_VALID_PRIMARY:.4f}.

Write the training loop yourself rather than importing `run_fm` from `baseline.py`: you need
your own pipeline to build on in later iterations, and you need to have verified that it
produces the expected number. You may read `baseline.py` for the model definition and import
`FM` from it if that is the cleanest way to match the reference implementation.

Report how close you landed to {config.BASELINE_VALID_PRIMARY:.4f} and treat anything inside
the noise floor as a successful reproduction.
{base.CONTRACT_REMINDER}"""

    text, ti, to = llm.complete(base.CODE_SYSTEM, prompt, config.BASELINE_MODEL,
                                cached_prefix=base.TASK_DESCRIPTION, role='baseline')
    from ..executor import strip_fences
    return (base.extract_section(text, 'Hypothesis'),
            strip_fences(base.extract_section(text, 'Code') or text), ti, to)
