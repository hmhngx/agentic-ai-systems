"""
llm_client.py — Raw OpenRouter API calls.

Why raw HTTP instead of the openai SDK?
  The openai SDK abstracts the HTTP layer. Building the call manually forces
  understanding of exactly what is sent and received:
  - We control the stop sequences (critical for parsing Thought/Action blocks)
  - We see the raw response structure before any SDK parsing
  - We add custom headers (OpenRouter requires HTTP-Referer)
  - "temperature=0" is a parameter in an HTTP body, not an SDK abstraction

Why temperature=0 for the agent?
  The agent must make deterministic tool selections. Higher temperature introduces
  randomness into which tool is selected, producing non-reproducible reasoning chains.
  At temperature=0, the model deterministically picks the highest-probability token
  at each step — maximizing adherence to the Thought/Action/FINAL ANSWER format.
  Exception: the web_mock tool internally uses temperature=0.7 to produce varied
  fake search results. The AGENT itself always uses temperature=0.

Why stop=["Observation:"]?
  The model should NEVER generate its own Observation. Observations come from
  tool execution, not LLM generation. If the model hallucinates an Observation,
  it then writes a Thought based on a fake tool result, producing wrong answers.
  The stop sequence forces the model to halt at "Observation:" and wait for us
  to inject the real tool result.
"""

import os
import re

import requests

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")
AGENT_TEMPERATURE = 0  # deterministic — see docstring
WEB_MOCK_TEMPERATURE = 0.7  # varied fake search results
TIMEOUT_SECONDS = 60


def get_max_tokens() -> int:
    """Read at request time so .env / env changes apply after load_dotenv()."""
    return int(os.environ.get("OPENROUTER_MAX_TOKENS", "128"))


def _affordable_tokens_from_402(detail: str) -> int | None:
    match = re.search(r"can only afford (\d+)", detail, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_message_content(choice: dict[str, object]) -> str:
    """Normalize provider-specific response shapes to a single string."""
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if content is not None and str(content).strip():
            return str(content).strip()
        reasoning = message.get("reasoning")
        if reasoning is not None and str(reasoning).strip():
            return str(reasoning).strip()
    text = choice.get("text")
    if text is not None and str(text).strip():
        return str(text).strip()
    return ""


# Backward-compatible name for tests and static checks.
MAX_TOKENS = get_max_tokens()

# Retrying these in the ReAct loop cannot succeed — fail fast instead of burning 10 iterations.
FATAL_API_STATUS_CODES = frozenset({401, 402, 403})


class OpenRouterAPIError(RuntimeError):
    """HTTP error from OpenRouter with status code for fail-fast handling."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        super().__init__(f"OpenRouter API error {status_code}: {detail}")


def is_fatal_api_error(exc: BaseException) -> bool:
    """True when the loop should stop immediately (auth/billing), not self-correct."""
    if isinstance(exc, OpenRouterAPIError):
        return exc.status_code in FATAL_API_STATUS_CODES
    msg = str(exc)
    return any(f"API error {code}" in msg for code in FATAL_API_STATUS_CODES)


def call_llm(
    messages: list[dict[str, str]],
    stop: list[str] | None = None,
    temperature: float = AGENT_TEMPERATURE,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Make a raw HTTP POST to OpenRouter /chat/completions.

    Args:
        messages: OpenAI-compatible message list with role/content keys.
        stop: Stop sequences; agent calls pass ["Observation:"] to prevent
              the model from hallucinating tool results.
        temperature: Sampling temperature; AGENT_TEMPERATURE (0) for agent
                     reasoning, WEB_MOCK_TEMPERATURE (0.7) for web_mock only.
        model: OpenRouter model identifier from OPENROUTER_MODEL env var.

    Returns:
        The assistant message content string from the first choice.

    Raises:
        RuntimeError: If OPENROUTER_API_KEY is missing, HTTP fails, or response is empty.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Copy .env.example to .env and add your OpenRouter API key."
        )

    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/agentic-ai-systems",
        "X-Title": "agentic-ai-systems-day7-react",
    }
    max_tokens = get_max_tokens()
    last_error_detail = ""

    for attempt in range(2):
        body: dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop or [],
        }
        response = requests.post(url, headers=headers, json=body, timeout=TIMEOUT_SECONDS)

        if not response.ok:
            last_error_detail = response.text[:500]
            if response.status_code == 402 and attempt == 0:
                affordable = _affordable_tokens_from_402(last_error_detail)
                if affordable is not None and max_tokens > affordable:
                    max_tokens = affordable
                    continue
            raise OpenRouterAPIError(response.status_code, last_error_detail)

        data = response.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(
                f"OpenRouter returned no choices for model {model}. "
                "Try OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct in .env."
            )

        content = _extract_message_content(choices[0])
        if content:
            return content

        raise RuntimeError(
            f"OpenRouter returned empty content for model {model}. "
            "Some models (e.g. minimax) may not support this ReAct format — "
            "use meta-llama/llama-3.3-70b-instruct or anthropic/claude-sonnet-4-5."
        )

    raise OpenRouterAPIError(402, last_error_detail)
