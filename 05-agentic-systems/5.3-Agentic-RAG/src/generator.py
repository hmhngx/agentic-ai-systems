"""Grounded answer generation.

Offline default: extract the best chunk sentence when grounding is strong;
otherwise simulate the canonical RAG failure (answer from the question itself).
With USE_LLM=1 a real OpenRouter model generates instead, but the topology and
the {answer, citations, confidence} contract are unchanged.

Why simulate a "guess"? RAGAS faithfulness measures whether the answer is grounded
in the retrieved context. A purely extractive generator is faithful by
construction, so it could never demonstrate a faithfulness failure. When grounding
is weak we therefore emit a sentence built from the QUESTION'S OWN words (which
are absent from the chunks) — the deterministic stand-in for a model filling the
gap from parametric memory. The faithfulness proxy then catches it, exactly as it
would catch a real hallucination.
"""
from __future__ import annotations

import os
import re

from src import config
from src.retriever import get_space
from src.text import content_tokens

GROUND_MIN = 1.6   # min idf mass (query ∩ best chunk) to trust extraction
REFUSAL = ("I do not have enough information in the retrieved documents to "
           "answer this question.")

_SENT_RE = re.compile(r"[^.;]+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


def _best_sentence(query: str, chunk_text: str, space) -> str:
    """Sentence in the chunk with the highest idf overlap with the query."""
    qterms = set(content_tokens(query))
    best, best_mass = chunk_text.strip(), -1.0
    for sent in _sentences(chunk_text):
        mass = space.idf_mass(t for t in content_tokens(sent) if t in qterms)
        if mass > best_mass:
            best, best_mass = sent, mass
    return best


def _guess(question: str) -> str:
    """Simulated parametric-memory fabrication: assert the question's own content
    words as if answered. These words are absent from the chunks -> unsupported."""
    core = " ".join(content_tokens(question))
    return f"{core} is described in the retrieved context"


def generate(question: str, query: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {"answer": REFUSAL, "citations": [], "confidence": 0.0}

    if config.use_llm():
        return _generate_llm(question, query, chunks)

    space = get_space()
    best = chunks[0]
    qterms = set(content_tokens(query))
    chunk_terms = set(content_tokens(best["text"]))
    confidence = space.idf_mass(qterms & chunk_terms)

    if confidence >= GROUND_MIN:
        sentence = _best_sentence(query, best["text"], space)
        answer = f"{sentence} [{best['label']}]."
        return {"answer": answer, "citations": [best["label"]], "confidence": confidence}

    # weak grounding -> guess (cites the best chunk, but the claim isn't in it)
    answer = f"{_guess(question)} [{best['label']}]."
    return {"answer": answer, "citations": [best["label"]], "confidence": confidence}


def _generate_llm(question: str, query: str, chunks: list[dict]) -> dict:
    """Real OpenRouter generation (only when USE_LLM=1). Context-only prompt."""
    from langchain_openai import ChatOpenAI

    context = "\n".join(f"[{c['label']}] {c['text']}" for c in chunks)
    system = (
        "Answer ONLY from the CONTEXT. Cite every claim inline as [Doc N]. "
        "If the context lacks the answer, reply exactly: " + REFUSAL)
    llm = ChatOpenAI(model=config.OPENROUTER_MODEL, base_url=config.OPENROUTER_BASE_URL,
                     api_key=os.environ["OPENROUTER_API_KEY"],
                     temperature=0, max_tokens=400, timeout=45)
    text = str(llm.invoke([("system", system),
                           ("human", f"CONTEXT:\n{context}\n\nQUESTION:\n{question}")]).content).strip()
    cites = re.findall(r"Doc \d+", text)
    return {"answer": text, "citations": sorted(set(cites)), "confidence": float(len(cites))}
