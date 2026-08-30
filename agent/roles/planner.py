"""Reflect on the last result and decide the next experiment. Produces an ExperimentSpec.

Reflection and planning are one call rather than two: a reflection written separately only
reaches the next iteration as a summary line, whereas here the analysis of what just happened
is the reasoning that selects what happens next.

The Planner is asked for three brief candidates before committing to one. That costs little
and stops the first plausible idea from winning by default.

It holds the search tool. It is not told which methods to consider -- identifying what is
worth trying, and why, is the work being graded.
"""
from .. import config, research
from ..spec import parse_spec
from . import base

SYSTEM = """You are an ML researcher deciding the next experiment on a within-user ranking task.

Reply with exactly two sections and nothing else.

# Reasoning
First reflect on the most recent result: did it match its hypothesis, and if not, why?

Attributing a failure correctly is the highest-stakes judgement you make, because it decides
what you stop trying. A mechanism and a particular implementation of it are two different
claims. An experiment that fails may have shown the mechanism is wrong -- or only that one
encoding of it does not work here, which a different formulation, approximation, sampling
scheme or parameterisation might. A result inside the noise floor has shown nothing at all
about either, and a crash has shown nothing about the idea.

So do not retire a mechanism on the strength of a single implementation unless the evidence
points at the mechanism itself. Ruling out a whole family that in fact contained the answer is
far more expensive than one extra experiment, and it is the failure that cannot be recovered
from later in the run. Say explicitly which you are concluding, and what evidence makes you
confident it is that one rather than the other.

Then state what is genuinely ruled out.

Next, decide whether this iteration needs the literature at all.

**Search when you are choosing a direction** -- opening a mechanism nothing in this run has
tried yet. Do it *before* you settle on one, querying from the structure of the problem: the
metric, the data shape, what the last result exposed. Searching to justify a choice already
made yields a citation; searching before choosing can yield a candidate you would not otherwise
have had, which is the only reason the tool is worth its cost.

**Do not search when you are refining** -- tuning a parameter, changing a sampling scheme, or
fixing an implementation of a mechanism already tested here. Say which node you are refining
and move on. A search run out of obligation produces a paper stapled to an unrelated idea.

Then sketch three candidate experiments, in one or two sentences each. Where you searched, name
the retrieved work bearing on each candidate and what it specifically claims. Where you did
not, or where nothing relevant came back, say so plainly. Never attach a paper to an idea it
does not actually support: a loose citation is worse than an honest "the literature was
silent," and it is the kind of thing a reader checks.

Prefer candidates that differ in mechanism from each other, not the same mechanism applied to a
different field. Say which you are choosing and why it has the best expected value for one
iteration.

# Spec
A single JSON object, in a ```json block, with exactly these keys:

{
  "reflection": "what the last result showed and what it rules out",
  "prior_failure_attribution": "if the last experiment failed: 'mechanism' | 'implementation' | 'noise' | 'bug', then one sentence on why it is that one. Empty string if the last experiment succeeded or was the baseline.",
  "candidates_considered": ["one line each for the three you weighed, each naming the retrieved work that bears on it or saying the search surfaced nothing relevant"],
  "hypothesis": "the claim being tested, stated so it could be wrong",
  "mechanism": "why this should move the metric, in terms of how the metric is computed",
  "evidence": {
    "task_structure": "what about this task's structure supports it",
    "previous_experiments": "which prior nodes support it, by number",
    "literature": "the search_papers identifier, plus the specific claim from that abstract that bears on this experiment -- quote or paraphrase what the paper actually says, not that it merely exists. Write 'searched, nothing applicable' when that is the truth; an unsupported citation is worse than none."
  },
  "target_metric": "GAUC | nDCG@5 | both",
  "proposed_change": "the single change, concretely",
  "expected_result": "the size of effect you expect",
  "falsification_condition": "the observation that would show this hypothesis is wrong",
  "risks": {"leakage": "...", "overfitting": "...", "runtime": "..."},
  "implementation_scope": ["solution.py"],
  "tags": ["two-to-four", "short-tags"]
}

Exactly one change per experiment, so its effect can be attributed. If you have several ideas,
pick one and say in the reasoning why it beat the others.
"""


def _seen_map(journal):
    """paper id -> what happened when it was last used, so an already-cited paper carries its
    outcome rather than a bare 'seen' flag."""
    out = {}
    for pid, meta in (getattr(journal, 'citation_registry', {}) or {}).items():
        if meta.get('used_in'):
            out[pid] = meta['used_in']
    return out


def run(llm, iteration, journal, drafting: bool = False):
    """Returns (spec, error, raw_text, tokens_in, tokens_out). `spec` is None when parsing
    failed; the orchestrator replans rather than crashing.

    `drafting` asks for an independent solution rather than a change to the current best.
    """
    searches = {'count': 0}

    def handle_tool(name, args):
        if name != 'search_papers':
            return f'unknown tool {name!r}'
        searches['count'] += 1
        payload = research.search_papers(args.get('query', ''))
        for r in payload.get('results', []):
            journal.register_citation(r)
        return research.format_results(payload, seen=_seen_map(journal))

    prompt = _build_prompt(iteration, journal, drafting=drafting)
    text, ti, to = llm.complete(
        SYSTEM, prompt, config.PLANNER_MODEL, cached_prefix=base.TASK_DESCRIPTION,
        role='planner',
        tools=[research.TOOL_SCHEMA] if config.RESEARCH_ENABLED else None,
        tool_handler=handle_tool,
        max_tool_calls=config.MAX_SEARCHES_PER_CALL if config.RESEARCH_ENABLED else 0)

    spec_text = base.extract_section(text, 'Spec') or text
    spec, err = parse_spec(spec_text)
    return spec, err, text, ti, to


def _build_prompt(iteration, journal, extra: str = '', drafting: bool = False) -> str:
    best = journal.best
    last = journal.nodes[-1] if journal.nodes else None

    parts = [base.budget_line(iteration, journal, config)]

    if journal.eda_findings:
        parts.append(f'\n# Measured properties of this data\n\n{journal.eda_findings}')

    parts.append(f'\n# Prior attempts\n\n{journal.summary_table()}')

    if last is not None:
        parts.append(f'\n# Most recent result\n\n{_describe(last)}')

    # While drafting, the best solution's source is deliberately withheld. Showing it invites
    # a variation on it, and a variation is what the improve phase is for.
    if best is not None and best.code and not drafting:
        parts.append(f'\n# Current best solution (validation primary {best.val_primary:.4f})'
                     f'\n\n```python\n{best.code}\n```')
        if best.stdout_tail:
            parts.append(f'\nIts output:\n```\n{best.stdout_tail}\n```')

    if drafting:
        parts.append("""
# Your task right now

Draft an **independent** solution. This is the exploration phase: you are not modifying
anything above, you are covering a different region of the solution space so that what follows
can build on the best of several distinct approaches instead of the first one that worked.

Choose a mechanism that differs from every row of the table above in kind, not in degree -- a
different training objective, a different model class, a different representation of the
input. Two solutions that share a mechanism and differ in a hyperparameter are one draft, not
two, and the second is wasted.

You are opening a direction, so this is exactly the case where the literature is worth
searching, and worth searching before you commit rather than after.

Say plainly what this draft assumes that the existing attempts do not. If it scores worse than
the current best, that is a useful result: it bounds the alternative rather than wasting the
iteration.""")
    else:
        parts.append("""
# Your task right now

Decide the single next experiment. Reason from how the metrics are computed, the structure of
this data, what previous nodes established, and -- when you are opening a new direction rather
than refining one -- the published literature, which you can search. In that case search before
you commit, not after: the point is to find an approach you did not already have, and a search
run to justify a decision already made cannot do that.

If a measurement would change your choice and the EDA pass did not produce it, you do not need
a separate analysis iteration: ask for the diagnostic to be printed alongside the next
experiment, since its output comes back to you.""")

    if extra:
        parts.append(f'\n# Your previous proposal was rejected\n\n{extra}\n\n'
                     'Address every point above and return a corrected spec.')

    return '\n'.join(parts)


def replan(llm, iteration, journal, reasons, drafting: bool = False):
    """One retry after the gates reject a spec, inside the same iteration."""
    searches = {'count': 0}

    def handle_tool(name, args):
        if name != 'search_papers':
            return f'unknown tool {name!r}'
        searches['count'] += 1
        payload = research.search_papers(args.get('query', ''))
        for r in payload.get('results', []):
            journal.register_citation(r)
        return research.format_results(payload, seen=_seen_map(journal))

    detail = '\n'.join(f'- {r}' for r in reasons)
    prompt = _build_prompt(iteration, journal, extra=detail, drafting=drafting)
    text, ti, to = llm.complete(
        SYSTEM, prompt, config.PLANNER_MODEL, cached_prefix=base.TASK_DESCRIPTION,
        role='planner',
        tools=[research.TOOL_SCHEMA] if config.RESEARCH_ENABLED else None,
        tool_handler=handle_tool,
        max_tool_calls=1 if config.RESEARCH_ENABLED else 0)
    spec, err = parse_spec(base.extract_section(text, 'Spec') or text)
    return spec, err, text, ti, to


def _describe(node) -> str:
    if node.is_buggy:
        return (f'node {node.id} ({node.operation}) FAILED: {node.buggy_reason}'
                + (f' ({node.exception_type})' if node.exception_type else '')
                + (f'\n\n{node.stderr_tail}' if node.stderr_tail else ''))
    # Only val_primary is required of a solution's output, so any of the others can be absent
    # on a node that still scored. Formatting None here would raise inside prompt building,
    # which is outside the loop's LLM error handling and would end the run.
    def num(x):
        return f'{x:.4f}' if isinstance(x, (int, float)) else 'not reported'

    bits = [f'node {node.id} ({node.operation})']
    if node.val_primary is not None:
        bits.append(f'validation primary {num(node.val_primary)} '
                    f'(GAUC {num(node.val_gauc)}, nDCG@5 {num(node.val_ndcg5)})')
        bits.append(f'delta vs baseline {node.val_primary - config.BASELINE_VALID_PRIMARY:+.4f}')
        if node.train_primary is not None:
            bits.append(f'train primary {node.train_primary:.4f} '
                        f'(train-validation gap {node.train_primary - node.val_primary:+.4f})')
        if node.unbiased_val_primary is not None:
            bits.append(f'random-exposure primary {node.unbiased_val_primary:.4f}')
        bits.append(f'seeds averaged {node.seeds_averaged}'
                    + (f', spread {[round(s, 4) for s in node.seed_scores]}'
                       if len(node.seed_scores) > 1 else ''))
        bits.append('accepted as new best' if node.accepted else 'rejected')
    return '\n'.join(bits)
