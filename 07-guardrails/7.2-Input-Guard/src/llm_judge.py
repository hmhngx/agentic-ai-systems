"""
llm_judge.py — Optional OpenRouter escalation for borderline PII and topic verdicts.

Mirrors the pattern from 7.1-Hallucination-Detection/src/llm_judge.py:
  - openai is imported lazily inside the function so the offline path never
    requires it to be installed or a key to be present.
  - Any failure (missing key, API error, timeout) raises LLMJudgeError,
    which guard.py catches to fall back to the offline verdict.
  - Activated only when INPUT_GUARD_USE_LLM=1.
"""
from __future__ import annotations

import os

from src.types import GuardConfig


class LLMJudgeError(RuntimeError):
    """Raised when the OpenRouter call cannot produce a usable verdict."""


_TOPIC_SYSTEM = (
    "You are a strict topic classifier for a RAG assistant that only handles "
    "questions about RAG, LLMs, embeddings, vector databases, and agentic AI. "
    "Given a user query, respond with ONLY one word: 'on_topic' or 'off_topic'. "
    "No explanation, no punctuation."
)

_PII_SYSTEM = (
    "You are a PII detector. Given a user message, respond with ONLY one word: "
    "'pii_detected' if the message contains personally identifiable information "
    "(name, email, phone, SSN, credit card, address, date of birth), "
    "or 'no_pii' otherwise. No explanation, no punctuation."
)


def _call_openrouter(system_prompt: str, user_message: str, model: str) -> str:
    """Call OpenRouter and return the stripped response text. Raises LLMJudgeError."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise LLMJudgeError("OPENROUTER_API_KEY is not set")

    try:
        from openai import APIError, OpenAI
    except ImportError as exc:
        raise LLMJudgeError("the 'openai' package is not installed") from exc

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=16,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
    except APIError as exc:
        raise LLMJudgeError(f"OpenRouter API error: {exc}") from exc

    return (response.choices[0].message.content or "").strip().lower()


def llm_topic_check(query: str, config: GuardConfig) -> str:
    """Returns 'on_topic' or 'off_topic'. Raises LLMJudgeError on any failure."""
    return _call_openrouter(_TOPIC_SYSTEM, query, config.llm_model)


def llm_pii_check(query: str, config: GuardConfig) -> str:
    """Returns 'pii_detected' or 'no_pii'. Raises LLMJudgeError on any failure."""
    return _call_openrouter(_PII_SYSTEM, query, config.llm_model)
