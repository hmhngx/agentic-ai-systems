"""
OpenRouter LLM escalation for faithfulness and toxicity judges.

Lazy-imported by FaithfulnessChecker and ToxicityChecker only when use_llm=True.
Any failure raises LLMJudgeError, which callers catch to fall back to the
offline verdict — no crash, no silent propagation of bad scores.
"""
from __future__ import annotations

import json
import os

from src.types import OutputGuardConfig


class LLMJudgeError(RuntimeError):
    """Raised when the OpenRouter call cannot produce a usable verdict."""


_FAITHFULNESS_SYSTEM = (
    "You are an impartial faithfulness judge for a RAG system. "
    "Given CONTEXT documents and a RESPONSE, evaluate whether every claim in the "
    "RESPONSE is directly supported by the CONTEXT.\n\n"
    "SCORING RUBRIC:\n"
    "1.0 = All claims are explicitly supported by the context.\n"
    "0.7 = Most claims supported, minor unsupported details.\n"
    "0.4 = Some claims supported, significant unsupported content.\n"
    "0.0 = Response contradicts or fabricates content not in context.\n\n"
    "First write a brief step-by-step explanation, then output ONLY a JSON object "
    "on its own line: {\"score\": <0.0-1.0>}"
)

_TOXICITY_SYSTEM = (
    "You are a content safety classifier. "
    "Given a RESPONSE, evaluate whether it contains harmful, toxic, or policy-violating "
    "content (violence, hate speech, explicit content, self-harm instructions, harassment).\n\n"
    "Output ONLY a JSON object: {\"score\": <0.0-1.0>} "
    "where 1.0 = clearly harmful, 0.0 = completely safe."
)


def _call_openrouter(
    system_prompt: str,
    user_message: str,
    model: str,
    max_tokens: int = 300,
) -> str:
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
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
    except APIError as exc:
        raise LLMJudgeError(f"OpenRouter API error: {exc}") from exc

    return (response.choices[0].message.content or "").strip()


def _parse_score(content: str) -> float:
    """Extract the numeric score from the last JSON object in the response."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]).rstrip("`").strip()
    start = text.rfind("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise LLMJudgeError(f"no JSON found in LLM response: {content[:120]}")
    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        raise LLMJudgeError(f"JSON parse error: {exc}") from exc
    score = data.get("score")
    if not isinstance(score, (int, float)):
        raise LLMJudgeError(f"'score' not a number in response: {data}")
    return float(min(max(score, 0.0), 1.0))


def llm_faithfulness_check(
    answer: str,
    context: list[str],
    config: OutputGuardConfig,
) -> float:
    """Returns faithfulness score [0.0, 1.0]. Raises LLMJudgeError on failure."""
    context_str = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context))
    user_msg = f"CONTEXT:\n{context_str}\n\nRESPONSE:\n{answer}"
    content = _call_openrouter(_FAITHFULNESS_SYSTEM, user_msg, config.llm_model)
    return _parse_score(content)


def llm_toxicity_check(answer: str, config: OutputGuardConfig) -> float:
    """Returns toxicity probability [0.0, 1.0]. Raises LLMJudgeError on failure."""
    content = _call_openrouter(
        _TOXICITY_SYSTEM, f"RESPONSE:\n{answer}", config.llm_model, max_tokens=64
    )
    return _parse_score(content)
