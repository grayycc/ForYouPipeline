"""A fresh solution from a blank file, rather than a mutation of the incumbent.

`improve` always branches from the current best, which makes the search a chain of small edits
to whatever the first working solution happened to be. That is the right shape once a promising
line exists, and the wrong shape for finding one: runs/v2 spent fourteen consecutive iterations
on variants of a single FM script, and no prompt could have produced a gradient-boosted ranker
from there because the operation itself only knows how to edit what it is given.

Drafts branch from nothing, so a bad one costs one node and a good one opens a line the greedy
search can then exploit. They run early, before the improve loop takes over.
"""
from .. import config, prompts
from . import base


def _parse_tags(text):
    """Tags from a '# Tags' section, comma- or bullet-separated, normalised like the planner's.

    A draft used to return no tags at all, and the duplicate gate skips any node whose tags are
    empty -- in both directions. So drafts were neither checked against prior work nor visible
    as prior work themselves, which is how runs/v5 restated one refuted LightGBM-over-CTR idea
    across four separate drafts (nodes 2, 3, 4, 14). Drafts are precisely where a fresh
    restatement of a settled question comes from, so they have to carry the same handle on
    "what mechanism is this" that an improve does.
    """
    raw = base.extract_section(text, 'Tags')
    if not raw:
        return []
    parts = []
    for line in raw.splitlines():
        line = line.strip().lstrip('-*').strip()
        if line.startswith('#'):
            continue
        parts.extend(p.strip().strip('`"\'[],') for p in line.split(','))
    seen, tags = set(), []
    for p in parts:
        t = p.lower().replace(' ', '-')
        if t and t not in seen and len(t) < 40:
            seen.add(t)
            tags.append(t)
    return tags[:4]


def run(llm, iteration, journal, avoid=''):
    """Returns (hypothesis, tags, code, tokens_in, tokens_out).

    `avoid` names a mechanism this draft collided with on a previous attempt, so the redraft is
    steered off it rather than being asked the same open question again.
    """
    prompt = prompts.draft_prompt(iteration, journal)
    if avoid:
        prompt += f'\n\n# Already tested -- do not repeat\n\n{avoid}\n'
    text, ti, to = llm.complete(base.CODE_SYSTEM, prompt, config.CODER_MODEL,
                                cached_prefix=base.TASK_DESCRIPTION, role='draft')
    from ..executor import strip_fences
    return (base.extract_section(text, 'Hypothesis'),
            _parse_tags(text),
            strip_fences(base.extract_section(text, 'Code') or text), ti, to)
