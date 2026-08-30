"""Implement an ExperimentSpec. Does not decide what to test.

Returns a complete file rather than a diff. Diffs would cut tokens, but tokens are not the
binding constraint here and a patch that fails to apply is a wasted iteration.
"""
from .. import config
from ..executor import strip_fences
from . import base

SYSTEM = """You implement a specified machine-learning experiment. You do not choose it.

Reply with a single complete Python file in one ```python block. No prose, no diffs, no
partial snippets -- the file is written to disk and executed exactly as given.

Implement exactly the change the spec describes. Do not substitute a different idea even if
you think it would score better: the experiment must isolate one change so its effect can be
attributed. If the spec cannot be implemented as written, implement the closest thing that
runs and say so in a comment at the top of the file.
""" + base.UNSPECIFIED_CHOICES


def _render(spec) -> str:
    """The spec as prompt text, with the target metric's own definition appended so the
    choices that depend on it are made next to it."""
    lines = [f'hypothesis: {spec.hypothesis}',
             f'mechanism: {spec.mechanism}',
             f'change to make: {spec.proposed_change}',
             f'target metric: {spec.target_metric}',
             f'expected result: {spec.expected_result}']
    if spec.risks:
        lines.append('risks noted by the planner: '
                     + '; '.join(f'{k}: {v}' for k, v in spec.risks.items()))
    if spec.implementation_scope:
        lines.append(f'scope: {", ".join(spec.implementation_scope)}')
    lines.append(f'\nHow that metric aggregates:\n\n{base.METRIC_SECTION}')
    return '\n'.join(lines)


def _retry_if_broken(llm, code, ti, to):
    """Compile check plus one retry, using the coder's own prompt and model."""
    return base.retry_if_broken(llm, code, SYSTEM, config.CODER_MODEL, 'coder', ti, to)


def run(llm, spec, parent_code, parent_stdout=''):
    """Returns (code, tokens_in, tokens_out)."""
    if parent_code:
        context = (f'# The current solution to modify\n\n```python\n{parent_code}\n```'
                   + (f'\n\nIts most recent output:\n```\n{parent_stdout}\n```'
                      if parent_stdout else ''))
        instruction = ('Apply the change to the solution above and return the complete '
                       'modified file.')
    else:
        context = '# There is no prior solution; write one from scratch.'
        instruction = 'Write the complete file.'

    prompt = f"""# The experiment to implement

{_render(spec)}

{context}
{base.CONTRACT_REMINDER}
# Your task right now

{instruction} Implement the specified mechanism and nothing else. Where the spec leaves a
quantity to you, choose it so that it does not change how the target metric weights rows, and
name it in a comment."""

    text, ti, to = llm.complete(SYSTEM, prompt, config.CODER_MODEL,
                                cached_prefix=base.TASK_DESCRIPTION, role='coder')
    code = strip_fences(base.extract_section(text, 'Code') or text)
    return _retry_if_broken(llm, code, ti, to)


def revise(llm, spec, code, reason):
    """One correction pass after the reviewer flags a problem."""
    prompt = f"""# The experiment being implemented

{_render(spec)}

# The code under review

```python
{code}
```

# The reviewer flagged this

{reason}

# Your task right now

Fix the problem the reviewer identified while still implementing the specified change. Return
the complete corrected file. If you believe the reviewer is mistaken, say why in a comment at
the top of the file and return the code unchanged.
{base.CONTRACT_REMINDER}"""

    text, ti, to = llm.complete(SYSTEM, prompt, config.CODER_MODEL,
                                cached_prefix=base.TASK_DESCRIPTION, role='coder')
    code = strip_fences(base.extract_section(text, 'Code') or text)
    return _retry_if_broken(llm, code, ti, to)
