"""Pieces every role shares: the cacheable task prefix, the output contract, the rules for
choices nothing specifies, and response parsing.

extract_section is fence-aware because inside a ```python block a '#' starts a comment, not a
markdown header -- treating one as a header truncates the generated file mid-way.
"""
import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(_HERE, 'task_description.md')) as _fh:
    TASK_DESCRIPTION = _fh.read()

# Appended last to every code-writing prompt, for position rather than content. The contract
# itself is not restated: task_description.md is the cached prefix of the same call, so a copy
# here would put the identical instruction in one prompt twice, and the two copies would drift.
CONTRACT_REMINDER = """
Before you return the file, check it against the output contract and the library list in the
task description above, item by item. A file that trains well and violates the contract scores
nothing.

One requirement that is not in the contract and is the most common way a file loses its own
result: if training is iterative, the model you score and submit must be the iteration with the
best validation primary, not whichever one the loop happened to end on. Evaluate validation each
epoch, keep the best parameters, restore them before predicting, and print the epoch you kept.
Nodes are ranked against each other, so a file that trains past its own optimum is not being
compared on its hypothesis -- it is being compared on where its loop stopped.
"""


def _section(title_starts_with: str) -> str:
    """One '## ' section of the task description, by the opening words of its heading.

    Lifted from the file rather than restated here so there is a single definition of
    anything the roles quote. Raises at import if the heading moves, since silently sending
    an empty section is worse than failing to start.
    """
    out, taking = [], False
    for line in TASK_DESCRIPTION.splitlines():
        if line.startswith('## '):
            if taking:
                break
            taking = line[3:].lstrip('0123456789. ').lower().startswith(
                title_starts_with.lower())
            continue
        if taking:
            out.append(line)
    body = '\n'.join(out).strip()
    if not body:
        raise RuntimeError(f'task_description.md has no section starting '
                           f'"{title_starts_with}" -- a role quotes it')
    return body


# How the metrics aggregate. Placed next to the spec in code-writing prompts because the
# choices that go wrong -- a sampling scheme, a truncation, a per-user cap -- are made far
# from the definition, and whether they are right depends on how the metric weights users.
METRIC_SECTION = _section('The metric')

# Shared by every role that writes a training file. The choices a spec or a bug report leaves
# open are where results are decided: a per-user cap and a missing epoch selection each moved
# the same idea by more than 0.04, and both were read afterwards as evidence about the idea.
UNSPECIFIED_CHOICES = """
On the choices nothing specifies for you.

A hypothesis fixes the mechanism, not every number. You will still choose batch sizes, epoch
counts, sampling schemes, caps, truncations, normalisations, initialisations. Two rules govern
those.

**Never let such a choice change what is being optimised.** The most expensive mistake
available to you is a default that quietly reweights the objective -- capping how many examples
a heavy user contributes, truncating a ranked list, normalising away a magnitude the metric
depends on. These feel like hygiene. They are not: they alter the thing being measured, and the
result is then read as evidence about the mechanism, which retires an idea that in fact worked.
Before fixing any quantity that controls *how much weight some group of rows carries*, check it
against how the target metric aggregates. If your choice disagrees with the metric, the metric
wins.

**Name every such choice in a comment.** For each constant, cap, threshold or sampling rule
nothing specified, say in one short comment what you chose and why. An unexplained magic number
is indistinguishable from a bug when the result is being interpreted, and a later reader -- or a
later you, fixing this file -- will not know whether it was reasoned or typed.
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
""" + UNSPECIFIED_CHOICES


def compiles(code: str):
    """Returns (ok, message). Catching a syntax error before the file is executed costs one
    cheap retry; letting it through costs the whole iteration plus a debugger call to fix
    something trivial."""
    try:
        compile(code, 'solution.py', 'exec')
        return True, ''
    except SyntaxError as e:
        return False, f'line {e.lineno}: {e.msg}'
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


def retry_if_broken(llm, code, system, model, role, ti, to):
    """One targeted retry quoting the syntax error.

    Models sometimes write reasoning prose into the middle of the file -- "Wait, I need to
    reconsider..." -- which is a SyntaxError on that line. Quoting the error back fixes it far
    more reliably than rerunning the original prompt, and every role that emits a file needs
    this, not only the coder.
    """
    from ..executor import strip_fences
    ok, msg = compiles(code)
    if ok:
        return code, ti, to
    print(f'  [{role}] generated file does not parse ({msg}) -- retrying once')
    fix_prompt = f"""The file you produced is not valid Python: {msg}

```python
{code}
```

Return the corrected complete file. Emit only Python -- any commentary must be a `#` comment
or a docstring, never bare prose, and never mid-file reconsiderations. Decide on one approach
before writing and write only that.
"""
    text, ti2, to2 = llm.complete(system, fix_prompt, model,
                                  cached_prefix=TASK_DESCRIPTION, role=role)
    fixed = strip_fences(extract_section(text, 'Code') or text)
    ok2, _ = compiles(fixed)
    return (fixed if ok2 else code), ti + ti2, to + to2


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
    """The header every planner prompt opens with: where the run is, what it has to beat, and
    how large a gain has to be before it means anything."""
    best = journal.best
    best_s = f'{best.val_primary:.4f} (node {best.id})' if best else 'none yet'
    return (f'Iteration {iteration} of {config.MAX_ITERATIONS}. '
            f'Best validation primary so far: {best_s}. '
            f'Baseline to beat: {config.BASELINE_VALID_PRIMARY:.4f} validation. '
            f'Noise floor: {config.SEED_STD:.4f} std, so gains under 0.002 are not real.')
