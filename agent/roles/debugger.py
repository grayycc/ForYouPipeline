"""Fix a solution that failed to run. Fires only on failure.

Capped at two attempts per branch by the orchestrator: with 50 iterations, three failed fixes
on one branch would cost 6% of the entire budget. After two, the branch is abandoned and the
search re-improves from the current best, so the run can never get permanently stuck.
"""
from .. import config
from ..executor import strip_fences
from . import base


def run(llm, iteration, journal, node):
    """Returns (hypothesis, code, tokens_in, tokens_out)."""
    prompt = f"""{base.budget_line(iteration, journal, config)}

# A solution failed and needs fixing

Failure mode: **{node.buggy_reason}**{f' ({node.exception_type})' if node.exception_type else ''}
This is fix attempt {node.debug_depth + 1} of {config.MAX_DEBUG_DEPTH} on this branch; after
that the branch is abandoned.

What it was testing:
{node.hypothesis or (node.spec or {}).get('hypothesis', 'not recorded')}

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
idea is unworkable as written, say so explicitly in your hypothesis and make the smallest
change that lets it run.
{base.CONTRACT_REMINDER}"""

    # Second attempt escalates: the cheap fix has already failed once.
    model = config.DEBUGGER_RETRY_MODEL if node.debug_depth >= 1 else config.DEBUGGER_MODEL
    text, ti, to = llm.complete(base.CODE_SYSTEM, prompt, model,
                                cached_prefix=base.TASK_DESCRIPTION, role='debugger')
    code = strip_fences(base.extract_section(text, 'Code') or text)
    code, ti, to = base.retry_if_broken(llm, code, base.CODE_SYSTEM, model,
                                        'debugger', ti, to)
    return base.extract_section(text, 'Hypothesis'), code, ti, to
