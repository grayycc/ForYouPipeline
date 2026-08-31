"""Reflect on the last result and decide the next experiment. Produces an ExperimentSpec.

Reflection and planning are one call rather than two: a reflection written separately only
reaches the next iteration as a summary line, whereas here the analysis of what just happened
is the reasoning that selects what happens next.

The Planner is asked for three brief candidates before committing to one. That costs little
and stops the first plausible idea from winning by default.

It holds the search tool. It is not told which methods to consider -- identifying what is
worth trying, and why, is the work being graded.
"""
from .. import config, diagnose, research
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
    "literature": "if you searched: the bracketed ids you actually read, and in one clause what each contributed -- including 'read, did not support this' where that is the honest answer. Empty ONLY if you ran no search this iteration."
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

    # Evidence from prior runs, which this run has not paid for and would otherwise re-derive:
    # gradient-boosted trees is 0-accepted from 15 attempts across 7 separate runs, and every
    # one of those runs spent an iteration finding that out again.
    if getattr(journal, 'cross_run_yield', ''):
        parts.append(f'\n{journal.cross_run_yield}')

    # Before the softer "what the evidence has and has not settled" block, because this one is
    # enforced: a spec matching a closed family is rejected by the gates, not merely flagged.
    exhausted = diagnose.exhausted_block(journal)
    if exhausted:
        parts.append(f'\n{exhausted}')

    ruled_out = diagnose.ruled_out_block(journal)
    if ruled_out:
        parts.append(f'\n{ruled_out}')

    coverage = diagnose.stack_coverage(journal)
    if coverage:
        parts.append(f'\n{coverage}')

    calibration = diagnose.calibration_block(journal)
    if calibration:
        parts.append(f'\n{calibration}')

    # Papers retrieved on earlier iterations, with what came of them. runs/v5 ran 19 searches
    # and surfaced 42 papers, and 18 of its 19 nodes recorded no literature at all: each
    # iteration searched from scratch inside its own tool loop and the results died with it.
    # Carrying the registry forward makes the reading list cumulative, and shows which papers
    # have already been acted on so a search is not silently repeated.
    reg = getattr(journal, 'citation_registry', {}) or {}
    if reg:
        rows = [f'- [{pid}] {m.get("title", "")[:110]}'
                + (f' -- ALREADY USED: {m["used_in"]}' if m.get('used_in') else ' -- not yet used')
                for pid, m in list(reg.items())[:20]]
        parts.append('\n# Papers already retrieved this run\n\n'
                     'Cumulative across iterations. You may cite these by id without searching '
                     'again; search when you need something they do not cover.\n\n'
                     + '\n'.join(rows))

    parts.append("""
# Your task right now

Decide the single next experiment. Reason from how the metrics are computed, the structure of
this data, what previous nodes established, and -- if you judge it useful -- the published
literature, which you can search. You are not required to search.

Read the diagnosis column before concluding anything from a score. A node marked `under_trained`
did not test its hypothesis at all -- its low score says the optimisation failed, not that the
idea is wrong, and treating the two the same way abandons directions that were never tried. A
node marked `overfit` fit the training data and failed to generalise, which points at the
feature's sparsity rather than at the mechanism behind it.

Every iteration costs the same whatever you spend it on, so the question is not "might this
help?" but "if this works as well as I expect, will the result be readable?". A change whose
honest best case is under 0.002 cannot clear the noise floor: it will come back diagnosed
`noise` whether the mechanism was real or not, and you will have bought an unfalsifiable
result at the price of a testable one. State your expected effect honestly and, if it lands
below 0.002, prefer the candidate whose effect would be large enough to see. Being wrong about
a big change teaches more than being right about an invisible one.

**You have fewer iterations than the cap suggests, so weight the early ones accordingly.**
The run ends when the best score stops improving by more than 0.002 across consecutive scoring
iterations, and on this task that has ended every recent run after four to eight of them --
against a cap of fifty and using under a quarter of the wall-clock ceiling. What you have
accumulated by roughly the fourth scoring node is, in practice, the final result. Two runs
illustrate the cost of pacing: one reached +0.0023 at its second scoring node and kept going;
another reached the same total but split across two nodes, stood at +0.00197 at the fourth, and
stopped there — three hundred-thousandths short. If several changes are each individually
below the floor and you have reason to think they compose, proposing them as one change is
both more measurable and leaves the search alive longer than spending an iteration on each.

Whatever you change must live inside the standalone solution file itself, including any feature
logic, computed from train-only history when relevant.

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
        # Each component is formatted only if present. The guard tests val_primary but the line
        # then formatted val_gauc and val_ndcg5 as well, so a node carrying a primary score
        # without its components raised TypeError here -- inside the prompt builder, which runs
        # every iteration.
        parts = [f'validation primary {node.val_primary:.4f}']
        comps = [f'{k} {v:.4f}' for k, v in (('GAUC', node.val_gauc),
                                             ('nDCG@5', node.val_ndcg5)) if v is not None]
        if comps:
            parts.append(f'({", ".join(comps)})')
        bits.append(' '.join(parts))
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
