"""Bedrock-backed LLM client with tiered models and retry/backoff.

The strong model drafts and rescues hard failures; the fast model handles
routine improve/debug. Repeated API failure must degrade the run, never crash it.
"""
import random
import time

from . import config


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self):
        self.total_in = 0
        self.total_out = 0
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

    def complete(self, system: str, user: str, strong: bool = False) -> tuple:
        """Returns (text, tokens_in, tokens_out). Raises LLMError only after exhausting retries."""
        self._ensure()
        model = config.STRONG_MODEL if strong else config.FAST_MODEL
        if not model:
            var = 'AGENT_STRONG_MODEL' if strong else 'AGENT_FAST_MODEL'
            raise LLMError(f'{var} is not set in .env -- put the inference profile id there')

        last = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                r = self._client.messages.create(
                    model=model, max_tokens=config.MAX_OUTPUT_TOKENS, system=system,
                    messages=[{'role': 'user', 'content': user}],
                )
                text = r.content[0].text
                if not text or not text.strip():
                    raise LLMError('empty response')
                self.total_in += r.usage.input_tokens
                self.total_out += r.usage.output_tokens
                return text, r.usage.input_tokens, r.usage.output_tokens
            except Exception as e:                       # timeouts, rate limits, malformed output
                # 400/401/403/404 are permanent -- retrying just burns wall-clock.
                status = getattr(e, 'status_code', None)
                if status in (400, 401, 403, 404):
                    raise LLMError(f'{type(e).__name__} ({status}), not retryable: {e}')
                last = e
                sleep = config.LLM_BACKOFF_BASE_S * (2 ** attempt) + random.uniform(0, 1)
                print(f'    [llm] {type(e).__name__}: {e} -- retry {attempt + 1}'
                      f'/{config.LLM_MAX_RETRIES} in {sleep:.1f}s')
                time.sleep(sleep)
        raise LLMError(f'LLM failed after {config.LLM_MAX_RETRIES} attempts: {last}')
