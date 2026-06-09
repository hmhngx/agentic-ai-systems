from __future__ import annotations

import re
import time

from src.llm_judge import LLMJudgeError, llm_topic_check
from src.types import CheckResult, GuardConfig, Verdict

# 60-word domain vocabulary for the RAG / LLM / agentic-AI domain.
_DOMAIN_WORDS = frozenset({
    "rag", "retrieval", "augmented", "generation", "embedding", "embeddings",
    "vector", "vectors", "llm", "llms", "agent", "agents", "guardrail",
    "guardrails", "token", "tokens", "prompt", "prompts", "langchain",
    "openai", "anthropic", "chunking", "chunk", "chunks", "inference",
    "hallucination", "hallucinations", "similarity", "cosine", "encoder",
    "decoder", "transformer", "transformers", "attention", "context",
    "pipeline", "pipelines", "tool", "tools", "memory", "graph", "knowledge",
    "node", "edge", "index", "indexing", "semantic", "search", "dense",
    "sparse", "hybrid", "rerank", "reranking", "bm25", "faiss", "qdrant",
    "pinecone", "weaviate", "chroma", "milvus", "bert", "gpt", "claude",
    "mistral", "llama", "openrouter", "huggingface", "nlp", "classification",
    "completion", "chatbot", "assistant", "answering", "document", "documents",
    "corpus", "dataset", "model", "models", "training", "evaluate", "evaluation",
    "benchmark", "grounding", "faithfulness", "fine-tuning", "finetuning",
})

_TOKENIZE_RE = re.compile(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*")

# Scores below this floor are definitively off-topic; LLM escalation is not attempted.
# Scores between this floor and topic_threshold are borderline: escalate to LLM if available.
_BORDERLINE_FLOOR: float = 0.04

_REDIRECT_MSG = (
    "This assistant answers questions about RAG, LLMs, embeddings, "
    "vector databases, and agentic AI systems."
)


def _score(query: str) -> float:
    """Keyword-density score: matched domain words / total tokens, clamped to [0, 1]."""
    tokens = [t.lower() for t in _TOKENIZE_RE.findall(query)]
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in _DOMAIN_WORDS)
    return min(hits / len(tokens), 1.0)


class TopicChecker:
    def __init__(self, config: GuardConfig) -> None:
        self._threshold = config.topic_threshold
        self._config = config

    def check(self, query: str) -> CheckResult:
        t0 = time.perf_counter()
        try:
            score = _score(query)
        except Exception:
            return CheckResult(
                check="topic",
                verdict=Verdict.BLOCK,
                reason="malformed input",
                score=1.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Clearly on-topic — no LLM needed
        if score >= self._threshold:
            return CheckResult(
                check="topic",
                verdict=Verdict.ALLOW,
                reason=f"on-topic (score {score:.3f} ≥ {self._threshold})",
                score=score,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Borderline zone — escalate to LLM when available
        if self._config.use_llm and score >= _BORDERLINE_FLOOR:
            try:
                llm_verdict = llm_topic_check(query, self._config)
                if llm_verdict == "on_topic":
                    return CheckResult(
                        check="topic",
                        verdict=Verdict.ALLOW,
                        reason=f"on-topic (LLM override; keyword score {score:.3f})",
                        score=score,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                # "off_topic" or any unexpected token → block with redirect
                return CheckResult(
                    check="topic",
                    verdict=Verdict.BLOCK,
                    reason=(
                        f"off-topic (LLM confirmed; score {score:.3f} < {self._threshold}). "
                        f"{_REDIRECT_MSG}"
                    ),
                    score=score,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            except LLMJudgeError:
                pass  # conservative: fall through to offline block below

        # Clearly off-topic or LLM unavailable / failed
        return CheckResult(
            check="topic",
            verdict=Verdict.BLOCK,
            reason=f"off-topic (score {score:.3f} < {self._threshold}). {_REDIRECT_MSG}",
            score=score,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
