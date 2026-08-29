"""Spec parsing must never raise, and gates must catch what plain code can catch."""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.spec import parse_spec, ExperimentSpec
from agent.gates import check_spec
from agent.journal import Journal, Node

GOOD = {
    "reflection": "Node 3 matched its hypothesis; ordering-aligned changes look promising.",
    "candidates_considered": ["listwise softmax", "author count features", "lr schedule"],
    "hypothesis": "Replacing pointwise BCE with a within-user listwise softmax will raise "
                  "GAUC and nDCG@5 because both measure intra-user ordering.",
    "mechanism": "Softmax over each user's impression list optimises full-list ordering "
                 "directly, while BCE optimises calibration per row.",
    "evidence": {"task_structure": "~5 impressions per user, judged within-user",
                 "previous_experiments": "node 3 showed capacity is not the bottleneck",
                 "literature": "W2140310134"},
    "target_metric": "both",
    "proposed_change": "Swap the pointwise BCE objective for a within-user listwise softmax.",
    "expected_result": "+0.005 to +0.02 primary",
    "falsification_condition": "primary moves by less than 0.002 across three seeds",
    "risks": {"leakage": "none, objective only", "overfitting": "low", "runtime": "negligible"},
    "implementation_scope": ["solution.py"],
    "tags": ["ranking-loss", "listwise"],
}


def journal_with(spec_dict=None, registry=None):
    j = Journal()
    j.citation_registry = registry or {}
    if spec_dict:
        n = Node(id=3, parent_id=0, operation='improve')
        n.is_buggy, n.val_primary, n.spec = False, 0.61, spec_dict
        j.append(n)
    return j


def test_parses_plain_json():
    spec, err = parse_spec(json.dumps(GOOD))
    assert err == '' and spec.tags == ['ranking-loss', 'listwise'], (err, spec)


def test_parses_fenced_json_with_prose():
    text = "Here is my plan.\n\n```json\n" + json.dumps(GOOD) + "\n```\nThat is the spec."
    spec, err = parse_spec(text)
    assert err == '' and spec.hypothesis.startswith('Replacing pointwise'), (err, spec)


def test_malformed_json_returns_error_not_raise():
    spec, err = parse_spec('{"hypothesis": "x", "mechanism":}')
    assert spec is None and 'not valid JSON' in err, (spec, err)


def test_no_json_returns_error():
    spec, err = parse_spec('I think we should try a ranking loss.')
    assert spec is None and 'no JSON' in err, (spec, err)


def test_empty_returns_error():
    assert parse_spec('')[0] is None and parse_spec('   ')[0] is None


def test_nested_braces_survive():
    spec, _ = parse_spec(json.dumps(GOOD))
    assert spec.evidence['task_structure'].startswith('~5 impressions')


def test_good_spec_passes():
    spec, _ = parse_spec(json.dumps(GOOD))
    ok, reasons = check_spec(spec, journal_with(registry={'W2140310134': {}}))
    assert ok, reasons


def test_protected_file_rejected():
    bad = dict(GOOD, implementation_scope=['evaluate.py'])
    spec, _ = parse_spec(json.dumps(bad))
    ok, reasons = check_spec(spec, journal_with(registry={'W2140310134': {}}))
    assert not ok and any('protected' in r for r in reasons), reasons


def test_missing_falsification_rejected():
    bad = dict(GOOD); bad['falsification_condition'] = ''
    spec, _ = parse_spec(json.dumps(bad))
    ok, reasons = check_spec(spec, journal_with(registry={'W2140310134': {}}))
    assert not ok and any('falsification_condition' in r for r in reasons), reasons


def test_duplicate_rejected():
    spec, _ = parse_spec(json.dumps(GOOD))
    ok, reasons = check_spec(spec, journal_with(spec_dict=GOOD,
                                                registry={'W2140310134': {}}))
    assert not ok and any('already tested' in r for r in reasons), reasons


def test_unregistered_citation_rejected():
    spec, _ = parse_spec(json.dumps(GOOD))
    ok, reasons = check_spec(spec, journal_with(registry={'W999999': {}}))
    assert not ok and any('never returned by a search' in r for r in reasons), reasons


def test_citation_without_any_search_rejected():
    spec, _ = parse_spec(json.dumps(GOOD))
    ok, reasons = check_spec(spec, journal_with(registry={}))
    assert not ok and any('no search was performed' in r for r in reasons), reasons


def test_too_many_files_rejected():
    bad = dict(GOOD, implementation_scope=['a.py', 'b.py', 'c.py'])
    spec, _ = parse_spec(json.dumps(bad))
    ok, reasons = check_spec(spec, journal_with(registry={'W2140310134': {}}))
    assert not ok and any('atomic' in r for r in reasons), reasons


def test_empty_spec_never_raises():
    ok, reasons = check_spec(ExperimentSpec(), journal_with())
    assert not ok and reasons


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'  PASS  {name}')
    print('\nall spec/gate tests passed')
