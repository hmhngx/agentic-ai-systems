"""Generate an answer from a fused context string and a question.

Offline mode (no API key): prints the full context that would be sent to the LLM.
API mode: calls OpenRouter with a zero-temperature system+user prompt.
"""
from __future__ import annotations

from src import config

_SYSTEM = (
    "You are a precise question-answering assistant. "
    "Answer the question using ONLY the provided context. "
    "If the answer requires connecting multiple facts, show your reasoning chain explicitly. "
    "Be concise — 1-3 sentences."
)


def generate_answer(context: str, question: str) -> str:
    if not config.use_embeddings_api():
        lines = [
            "=== CONTEXT PREVIEW (offline — no LLM call) ===",
            context,
            "=== END CONTEXT ===",
            f"QUESTION: {question}",
            "[Run with OPENROUTER_API_KEY set to get a real answer]",
        ]
        return "\n".join(lines)

    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=config.OPENROUTER_BASE_URL)
    resp = client.chat.completions.create(
        model=config.OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=300,
    )
    return resp.choices[0].message.content or ""
