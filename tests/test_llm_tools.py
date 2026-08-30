"""The tool loop, against a stubbed Bedrock client so no call leaves the machine.

Run smoke10 lost five of ten iterations to one bug here: the search cap was enforced by
dropping `tools` from the request, which left a conversation holding tool_result blocks with
no schema to explain them, and the model answered with zero content blocks. These tests pin
the two behaviours that prevent it -- the schema is never withdrawn, and the cap is enforced
by the reply the handler gives.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import llm as llm_mod
from agent.llm import LLMClient, LLMError, BUDGET_SPENT


class Block:
    def __init__(self, type, text=None, id=None, name=None, input=None):
        self.type, self.text, self.id, self.name, self.input = type, text, id, name, input


class Usage:
    input_tokens = 10
    output_tokens = 5
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class Response:
    def __init__(self, content, stop_reason='end_turn'):
        self.content, self.stop_reason, self.usage = content, stop_reason, Usage()


class FakeMessages:
    """Replays a scripted list of responses and records whether tools were sent each turn."""
    def __init__(self, script):
        self.script, self.turns, self.tools_sent = list(script), 0, []

    def create(self, **kwargs):
        self.tools_sent.append('tools' in kwargs)
        self.turns += 1
        return self.script.pop(0)


class FakeClient:
    def __init__(self, script):
        self.messages = FakeMessages(script)


def _client(script):
    c = LLMClient()
    c._client = FakeClient(script)
    return c


def _search(n):
    return Block('tool_use', id=f'tu{n}', name='search_papers', input={'query': f'q{n}'})


TOOL = [{'name': 'search_papers', 'description': 'x', 'input_schema': {'type': 'object'}}]


def test_schema_is_never_withdrawn():
    """The regression: tools must be present on every turn once the loop starts."""
    script = [Response([Block('text', 'thinking'), _search(1), _search(2)], 'tool_use'),
              Response([Block('text', '# Spec\n{}')])]
    c = _client(script)
    text, _, _ = c.complete('sys', 'user', 'm', tools=TOOL,
                            tool_handler=lambda n, a: 'results', max_tool_calls=2)
    assert text == '# Spec\n{}', text
    assert c._client.messages.tools_sent == [True, True], c._client.messages.tools_sent


def test_cap_is_enforced_by_the_handler():
    """Past the cap the handler is not called; the model gets BUDGET_SPENT and answers."""
    calls = []
    script = [Response([_search(1), _search(2)], 'tool_use'),
              Response([_search(3)], 'tool_use'),
              Response([Block('text', 'answer')])]
    c = _client(script)

    def handler(name, args):
        calls.append(args['query'])
        return 'results'

    text, _, _ = c.complete('sys', 'user', 'm', tools=TOOL,
                            tool_handler=handler, max_tool_calls=2)
    assert text == 'answer', text
    assert calls == ['q1', 'q2'], calls              # the third never ran
    assert c.tool_calls == 2, c.tool_calls
    assert all(c._client.messages.tools_sent), c._client.messages.tools_sent


def test_a_failing_tool_does_not_kill_the_call():
    script = [Response([_search(1)], 'tool_use'), Response([Block('text', 'answer')])]
    c = _client(script)

    def boom(name, args):
        raise ValueError('openalex down')

    text, _, _ = c.complete('sys', 'user', 'm', tools=TOOL,
                            tool_handler=boom, max_tool_calls=2)
    assert text == 'answer', text


def test_empty_response_reports_why():
    """The old message was 'model returned no text' and diagnosed nothing."""
    c = _client([Response([], 'end_turn')])
    try:
        c.complete('sys', 'user', 'm')
    except LLMError as e:
        assert 'stop_reason=end_turn' in str(e) and 'blocks=' in str(e), str(e)
    else:
        raise AssertionError('expected LLMError')


def test_runaway_tool_use_terminates():
    """A model that ignores BUDGET_SPENT forever must raise, not loop."""
    script = [Response([_search(i)], 'tool_use') for i in range(llm_mod.MAX_TOOL_TURNS + 2)]
    c = _client(script)
    try:
        c.complete('sys', 'user', 'm', tools=TOOL,
                   tool_handler=lambda n, a: 'r', max_tool_calls=1)
    except LLMError as e:
        assert 'still calling tools' in str(e), str(e)
    else:
        raise AssertionError('expected LLMError')


def test_budget_spent_text_reaches_the_model():
    seen = {}
    script = [Response([_search(1)], 'tool_use'),
              Response([_search(2)], 'tool_use'),
              Response([Block('text', 'answer')])]
    c = _client(script)
    real_create = c._client.messages.create

    def spy(**kwargs):
        seen['messages'] = kwargs['messages']
        return real_create(**kwargs)

    c._client.messages.create = spy
    c.complete('sys', 'user', 'm', tools=TOOL, tool_handler=lambda n, a: 'r',
               max_tool_calls=1)
    flat = str(seen['messages'])
    assert BUDGET_SPENT in flat, flat[-300:]


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    for name, fn in fns:
        fn()
        print(f'  PASS  {name}')
    print(f'\nall {len(fns)} tool-loop tests passed')
