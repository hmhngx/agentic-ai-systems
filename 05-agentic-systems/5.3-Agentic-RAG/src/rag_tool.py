"""rag_v1_complete: the v1 RAG pipeline (retrieve -> generate) exposed BOTH as a
plain callable (rich RagResult, used by the graph node) and as a LangChain
StructuredTool (rag_tool, the LLM-facing interface a tool-calling agent invokes).

This is the "wire rag_v1_complete as a tool in LangGraph" requirement: the agent
does not hardcode a retrieval step — it holds a tool it can choose to call.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from langchain_core.tools import StructuredTool

from src.generator import generate
from src.retriever import retrieve


@dataclass
class RagResult:
    answer: str
    chunks: list[dict]
    citations: list[str]
    confidence: float

    def asdict(self) -> dict:
        return {"answer": self.answer, "chunks": self.chunks,
                "citations": self.citations, "confidence": self.confidence}


def rag_v1_complete(question: str, query: Optional[str] = None) -> RagResult:
    """Retrieve for `query` (defaults to `question`), then generate a grounded,
    cited answer to `question`."""
    search = query if query is not None else question
    chunks = retrieve(search)
    out = generate(question, search, chunks)
    return RagResult(answer=out["answer"], chunks=chunks,
                     citations=out["citations"], confidence=out["confidence"])


def _rag_tool_fn(query: str) -> str:
    """LLM-facing entry: returns a compact JSON string (answer + citations + doc ids)."""
    res = rag_v1_complete(query)
    return json.dumps({
        "answer": res.answer,
        "citations": res.citations,
        "doc_ids": [c["doc_id"] for c in res.chunks],
    })


rag_tool = StructuredTool.from_function(
    func=_rag_tool_fn,
    name="rag_v1_complete",
    description=("Retrieve grounded, cited answers from the Helios knowledge base. "
                 "Input: a search query string. Use when a question needs facts "
                 "from the knowledge base."),
)
