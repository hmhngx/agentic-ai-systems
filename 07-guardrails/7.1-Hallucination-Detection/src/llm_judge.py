"""
llm_judge.py - Optional LLM-as-a-judge faithfulness signal (Check 1 variant).

Activated with ``USE_LLM=1``. It asks an OpenRouter model to label each claim in
the answer as entailed / not entailed by the retrieved context (a natural-
language-inference framing of faithfulness), and converts that into the same
``SupportResult`` the string-overlap check produces - so the rest of the
pipeline is identical whether the support signal came from difflib or a model.

Everything here is lazy and defensive:
  - ``openai`` is imported inside the function so the offline default path (and
    the whole test suite) never needs the dependency installed or a key set.
  - Any failure - missing key, API error, malformed JSON - raises
    ``LLMJudgeError``, which the orchestrator catches to fall back to the
    deterministic string-overlap check rather than crashing the guardrail.
"""

from __future__ import annotations

import json
import os

from src.text import split_sentences, strip_citations
from src.types import ClaimSupport, DetectorConfig, RetrievedChunk, SupportResult


class LLMJudgeError(RuntimeError):
    """Raised when the LLM judge cannot produce a usable verdict."""


_SYSTEM_PROMPT = (
    "You are a strict faithfulness judge for a retrieval-augmented generation "
    "system. You are given CONTEXT documents and a list of CLAIMS extracted "
    "from an answer. For each claim, decide whether it is DIRECTLY SUPPORTED by "
    "the CONTEXT - i.e. a reader could verify it from the context alone, with "
    "no outside knowledge. Be conservative: if the context does not state it, "
    "it is NOT supported.\n\n"
    "Respond with ONLY a JSON object of the form:\n"
    '{\"claims\": [{\"index\": 0, \"supported\": true}, ...]}\n'
    "Include one entry per claim, in order. No prose, no code fences."
)


def _build_user_prompt(claims: list[str], chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)
    numbered = "\n".join(f"{i}. {claim}" for i, claim in enumerate(claims))
    return f"CONTEXT:\n{context}\n\nCLAIMS:\n{numbered}"


def _parse_judgement(content: str, n_claims: int) -> list[bool]:
    """Parse the model's JSON into an ordered list of booleans, defensively."""
    text = content.strip()
    # Tolerate accidental code fences despite the instruction not to use them.
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMJudgeError(f"judge returned non-JSON output: {exc}") from exc

    entries = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise LLMJudgeError("judge JSON missing a 'claims' list")

    verdicts = [False] * n_claims
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("index")
        if isinstance(idx, int) and 0 <= idx < n_claims:
            verdicts[idx] = bool(entry.get("supported"))
    return verdicts


def llm_support(
    question: str,
    answer: str,
    chunks: list[RetrievedChunk],
    config: DetectorConfig,
) -> SupportResult:
    """Compute Check 1 via an OpenRouter NLI judge. Raises ``LLMJudgeError``."""
    claims = [c for c in split_sentences(strip_citations(answer)) if c.strip()]
    if not claims:
        return SupportResult(
            supported_ratio=0.0, mean_support=0.0, per_claim=(), method="llm-judge"
        )

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise LLMJudgeError("OPENROUTER_API_KEY is not set")

    try:
        from openai import APIError, OpenAI
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise LLMJudgeError("the 'openai' package is not installed") from exc

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    try:
        response = client.chat.completions.create(
            model=config.llm_model,
            temperature=0,
            max_tokens=512,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(claims, chunks)},
            ],
        )
    except APIError as exc:
        raise LLMJudgeError(f"OpenRouter API error: {exc}") from exc

    content = response.choices[0].message.content or ""
    verdicts = _parse_judgement(content, len(claims))

    per_claim = tuple(
        ClaimSupport(
            claim=claim,
            support=1.0 if verdicts[i] else 0.0,
            containment=1.0 if verdicts[i] else 0.0,
            span_ratio=0.0,                # not measured by the judge
            best_chunk_id=None,            # the judge scores against all context
            supported=verdicts[i],
        )
        for i, claim in enumerate(claims)
    )
    supported_ratio = sum(verdicts) / len(verdicts)
    return SupportResult(
        supported_ratio=supported_ratio,
        mean_support=supported_ratio,      # binary per-claim => mean == ratio
        per_claim=per_claim,
        method="llm-judge",
    )
