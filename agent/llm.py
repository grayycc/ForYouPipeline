"""Bedrock-backed LLM client: prompt caching, tool use, retry/backoff.

Each role passes its own model, set independently in .env, so one role can be tuned without
disturbing the others. Repeated API failure must degrade the run, never crash it.

Two things here exist to control cost. The static task description is identical on every call,
so it sits behind an explicit cache_control breakpoint -- explicit rather than top-level
automatic, because the legacy Bedrock integration rejects the top-level form with a 400. And
the tool loop is capped, since every round-trip resends the whole conversation.
"""
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

from . import config


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self):
        self.total_in = 0
        self.total_out = 0
        self.cache_reads = 0
        self.cache_writes = 0
        self.tool_calls = 0
        self._client = None

    def _ensure(self):
        if self._client is not None:
            return
        from anthropic import AnthropicBedrock
        if not config.BEDROCK_API_KEY:
            raise LLMError('AWS_BEARER_TOKEN is not set in .env')
        if not config.AWS_REGION:
            raise LLMError('AWS_REGION is not set in .env')
        # api_key is mutually exclusive with IAM credentials in this SDK, so pass it
        # explicitly and let region come from .env rather than a boto3 profile.
        self._client = AnthropicBedrock(api_key=config.BEDROCK_API_KEY,
                                        aws_region=config.AWS_REGION)

    @staticmethod
    def _system_blocks(cached_prefix: Optional[str], system: str) -> List[Dict]:
        """Stable prefix first behind a cache breakpoint, volatile part after it. Caching is a
        prefix match, so anything that varies must come last or it invalidates the cache."""
        blocks = []
        if cached_prefix:
            blocks.append({'type': 'text', 'text': cached_prefix,
                           'cache_control': {'type': 'ephemeral'}})
        blocks.append({'type': 'text', 'text': system})
        return blocks

    def _account(self, response):
        u = response.usage
        self.total_in += u.input_tokens
        self.total_out += u.output_tokens
        self.cache_reads += getattr(u, 'cache_read_input_tokens', 0) or 0
        self.cache_writes += getattr(u, 'cache_creation_input_tokens', 0) or 0

    def complete(self, system: str, user: str, model: str,
                 cached_prefix: str = None, tools: List[Dict] = None,
                 tool_handler: Callable[[str, dict], str] = None,
                 max_tool_calls: int = 0, role: str = '') -> Tuple[str, int, int]:
        """Returns (text, tokens_in, tokens_out).

        When `tools` and `tool_handler` are given, tool calls are executed and fed back until
        the model stops asking or `max_tool_calls` is reached. Raises LLMError only after
        exhausting retries, so callers can treat failure as a recoverable node outcome.
        """
        self._ensure()
        if not model:
            var = f'AGENT_{role.upper()}_MODEL' if role else 'the role model'
            raise LLMError(f'{var} is not set in .env -- put an inference profile id there')

        system_blocks = self._system_blocks(cached_prefix, system)
        messages: List[Dict] = [{'role': 'user', 'content': user}]
        used_in = used_out = 0
        calls_made = 0

        while True:
            kwargs = dict(model=model, max_tokens=config.MAX_OUTPUT_TOKENS,
                          system=system_blocks, messages=messages)
            if tools and calls_made < max_tool_calls:
                kwargs['tools'] = tools

            response = self._request(**kwargs)
            self._account(response)
            used_in += response.usage.input_tokens
            used_out += response.usage.output_tokens

            tool_uses = [b for b in response.content if getattr(b, 'type', None) == 'tool_use']
            if not tool_uses or not tool_handler or calls_made >= max_tool_calls:
                text = '\n'.join(b.text for b in response.content
                                 if getattr(b, 'type', None) == 'text').strip()
                if not text:
                    raise LLMError('model returned no text')
                return text, used_in, used_out

            messages.append({'role': 'assistant', 'content': response.content})
            results = []
            for block in tool_uses:
                calls_made += 1
                self.tool_calls += 1
                try:
                    out = tool_handler(block.name, block.input)
                except Exception as e:                       # a tool must never kill the run
                    out = f'tool failed: {type(e).__name__}: {e}'
                results.append({'type': 'tool_result', 'tool_use_id': block.id,
                                'content': out})
            messages.append({'role': 'user', 'content': results})

    def _request(self, **kwargs):
        """One API call with retry. Permanent errors fail fast rather than burning wall-clock."""
        last = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                return self._client.messages.create(**kwargs)
            except Exception as e:
                status = getattr(e, 'status_code', None)
                if status in (400, 401, 403, 404):
                    raise LLMError(f'{type(e).__name__} ({status}), not retryable: {e}')
                last = e
                sleep = config.LLM_BACKOFF_BASE_S * (2 ** attempt) + random.uniform(0, 1)
                print(f'    [llm] {type(e).__name__}: {e} -- retry {attempt + 1}'
                      f'/{config.LLM_MAX_RETRIES} in {sleep:.1f}s')
                time.sleep(sleep)
        raise LLMError(f'LLM failed after {config.LLM_MAX_RETRIES} attempts: {last}')
