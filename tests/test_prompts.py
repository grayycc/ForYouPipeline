"""The shared prompt pieces in roles/base.py, and what the coder is actually told.

These are string tests, but the strings are the product: a contradiction between the task
description and the contract reminder is not a typo, it is two different instructions reaching
the same model in the same call.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.roles import base, coder
from agent.spec import ExperimentSpec

LIBRARIES = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'torch')


def _spec(target='GAUC'):
    return ExperimentSpec(
        hypothesis='pairwise loss beats pointwise', mechanism='optimises ordering',
        evidence={}, target_metric=target, proposed_change='swap BCE for BPR',
        expected_result='+0.003', falsification_condition='no gain',
        risks={}, implementation_scope=['solution.py'], tags=['bpr'])


def test_the_library_list_exists_once():
    """It once said 'standard library and numpy only' in the reminder while the task
    description listed six libraries -- two instructions, one call. The list now lives in the
    task description alone and the reminder points at it."""
    for lib in LIBRARIES:
        assert lib in base.TASK_DESCRIPTION, f'{lib} missing from task description'
        assert lib not in base.CONTRACT_REMINDER, f'{lib} restated in the reminder'
    assert 'numpy only' not in base.CONTRACT_REMINDER


def test_contract_reminder_points_rather_than_restates():
    """Both reach the model in the same call, so a restatement is a second copy that drifts."""
    for token in ('TRAIN_PRIMARY=', 'VAL_GAUC=', 'submission_valid.csv', '--data_dir'):
        assert token in base.TASK_DESCRIPTION, token
        assert token not in base.CONTRACT_REMINDER, f'{token} duplicated into the reminder'
    assert 'task description' in base.CONTRACT_REMINDER.lower()


def test_reference_numbers_are_not_transcribed_twice():
    """config reads kit/baseline_scores.json; task_description.md quotes the same figures for
    the agent to read. This is the assertion that keeps the prose honest."""
    from agent import config
    for value, label in ((config.BASELINE_VALID_PRIMARY, 'baseline valid primary'),
                         (config.BASELINE_TEST_PRIMARY, 'baseline test primary'),
                         (config.SANITY_FLOOR, 'item-popularity valid'),
                         (config.ORACLE_PRIMARY, 'oracle test primary')):
        assert f'{value:.4f}' in base.TASK_DESCRIPTION, \
            f'{label} {value:.4f} is in kit/baseline_scores.json but not the task description'
    assert f'{config.EPSILON}' in base.TASK_DESCRIPTION
    assert f'{config.SEED_STD:.4f}' in base.TASK_DESCRIPTION


def test_metric_section_is_lifted_not_restated():
    """One definition of the metric, in task_description.md, quoted wherever it is needed."""
    assert base.METRIC_SECTION and base.METRIC_SECTION in base.TASK_DESCRIPTION
    assert 'sum(npos_u * AUC_u) / sum(npos_u)' in base.METRIC_SECTION
    assert not hasattr(base, 'METRIC_DEFINITIONS'), 'second copy of the definitions'


def test_missing_section_fails_loudly():
    try:
        base._section('a heading that is not there')
    except RuntimeError as e:
        assert 'no section' in str(e)
    else:
        raise AssertionError('silently returning an empty section is worse than raising')


def test_coder_prompt_carries_the_weighting_rule_and_the_metric():
    rendered = coder._render(_spec())
    # The wording must not be readable as "average the per-user AUCs equally" -- a coder read
    # it that way and sampled users uniformly, flattening the weighting the metric is built on.
    assert 'sum(npos_u * AUC_u) / sum(npos_u)' in rendered, 'GAUC formula missing'
    assert 'not* a plain average' in rendered or 'not a plain average' in rendered, \
        'nothing rules out the equal-weight reading'
    # the failure this exists to prevent: an invented per-user cap that reweights the objective
    for phrase in ('caps', 'sampling schemes', 'reweights the objective'):
        assert phrase in coder.SYSTEM, f'missing from coder system prompt: {phrase}'
    assert 'name it in a comment' in coder.SYSTEM.lower() or \
           'name every such choice' in coder.SYSTEM.lower()


def test_metric_section_present_for_any_target():
    for target in ('GAUC', 'nDCG@5', 'both', ''):
        assert 'sum(npos_u * AUC_u) / sum(npos_u)' in coder._render(_spec(target)), target


class _FakeLLM:
    """Returns a valid file, and asserts the retry actually quotes the syntax error."""
    def __init__(self):
        self.calls = 0

    def complete(self, system, prompt, model, cached_prefix=None, role='', **kw):
        self.calls += 1
        assert 'not valid Python' in prompt, 'the retry must quote the error'
        return '```python\nimport numpy as np\nx = 1\n```', 5, 5


def test_every_code_writing_role_recovers_from_prose_in_the_file():
    """A model writing "Wait, I need to reconsider..." mid-file is a SyntaxError, and it cost
    a full iteration in every role that lacked the retry -- which was three of the four."""
    from agent.roles import baseline, debugger, eda
    prose = 'import numpy as np\nWait, I need to reconsider this approach.\nx = 1\n'
    assert not base.compiles(prose)[0]

    for name, system in (('baseline', base.CODE_SYSTEM), ('debugger', base.CODE_SYSTEM),
                         ('eda', eda.SYSTEM), ('coder', coder.SYSTEM)):
        llm = _FakeLLM()
        fixed, ti, to = base.retry_if_broken(llm, prose, system, 'model', name, 0, 0)
        assert base.compiles(fixed)[0], f'{name} did not recover'
        assert llm.calls == 1, f'{name} made {llm.calls} retries, expected exactly 1'
        assert (ti, to) == (5, 5), f'{name} lost the retry token accounting'

    # a file that already parses must not spend a call
    llm = _FakeLLM()
    same, ti, to = base.retry_if_broken(llm, 'x = 1\n', coder.SYSTEM, 'm', 'coder', 1, 2)
    assert llm.calls == 0 and same == 'x = 1\n' and (ti, to) == (1, 2)


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    for name, fn in fns:
        fn()
        print(f'  PASS  {name}')
    print(f'\nall {len(fns)} prompt tests passed')
