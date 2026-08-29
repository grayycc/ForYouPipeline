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
First reflect on the most recent result: did it match its hypothesis, and if not was the cause
a bug, a wrong idea, or a difference inside the noise floor? Those call for different
responses. Then state what that rules out. Then sketch three genuinely different candidate
experiments in one or two sentences each, and say which you are choosing and why it has the
best expected value for one iteration.

# Spec
A single JSON object, in a ```json block, with exactly these keys:

{
  "reflection": "what the last result showed and what it rules out",
  "candidates_considered": ["one line each for the three you weighed"],
  "hypothesis": "the claim being tested, stated so it could be wrong",
  "mechanism": "why this should move the metric, in terms of how the metric is computed",
  "evidence": {
    "task_structure": "what about this task's structure supports it",
    "previous_experiments": "which prior nodes support it, by number",
    "literature": "identifiers from search_papers, or empty if you did not search"
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


def run(llm, iteration, journal):
    """Returns (spec, error, raw_text, tokens_in, tokens_out). `spec` is None when parsing
    failed; the orchestrator replans rather than crashing."""
    searches = {'count': 0}

    def handle_tool(name, args):
        if name != 'search_papers':
            return f'unknown tool {name!r}'
        searches['count'] += 1
        payload = research.search_papers(args.get('query', ''))
        for r in payload.get('results', []):
            journal.register_citation(r)
        return research.format_results(payload, seen=_seen_map(journal))

    prompt = _build_prompt(iteration, journal)
    text, ti, to = llm.complete(
        SYSTEM, prompt, config.PLANNER_MODEL, cached_prefix=base.TASK_DESCRIPTION,
        role='planner',
        tools=[research.TOOL_SCHEMA] if config.RESEARCH_ENABLED else None,
        tool_handler=handle_tool,
        max_tool_calls=config.MAX_SEARCHES_PER_CALL if config.RESEARCH_ENABLED else 0)

    spec_text = base.extract_section(text, 'Spec') or text
    spec, err = parse_spec(spec_text)
    return spec, err, text, ti, to


def _build_prompt(iteration, journal, extra: str = '') -> str:
    best = journal.best
    last = journal.nodes[-1] if journal.nodes else None

    parts = [base.budget_line(iteration, journal, config)]

    if journal.eda_findings:
        parts.append(f'\n# Measured properties of this data\n\n{journal.eda_findings}')

    parts.append(f'\n# Prior attempts\n\n{journal.summary_table()}')

    if last is not None:
        parts.append(f'\n# Most recent result\n\n{_describe(last)}')

    if best is not None and best.code:
        parts.append(f'\n# Current best solution (validation primary {best.val_primary:.4f})'
                     f'\n\n```python\n{best.code}\n```')
        if best.stdout_tail:
            parts.append(f'\nIts output:\n```\n{best.stdout_tail}\n```')

    parts.append("""
# Your task right now

Decide the single next experiment. Reason from how the metrics are computed, the structure of
this data, what previous nodes established, and -- if you judge it useful -- the published
literature, which you can search. You are not required to search.

If a measurement would change your choice and the EDA pass did not produce it, you do not need
a separate analysis iteration: ask for the diagnostic to be printed alongside the next
experiment, since its output comes back to you.""")

    if extra:
        parts.append(f'\n# Your previous proposal was rejected\n\n{extra}\n\n'
                     'Address every point above and return a corrected spec.')

    return '\n'.join(parts)


def replan(llm, iteration, journal, reasons):
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
    prompt = _build_prompt(iteration, journal, extra=detail)
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
    bits = [f'node {node.id} ({node.operation})']
    if node.val_primary is not None:
        bits.append(f'validation primary {node.val_primary:.4f} '
                    f'(GAUC {node.val_gauc:.4f}, nDCG@5 {node.val_ndcg5:.4f})')
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
