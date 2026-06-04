"""Each Week-2-Review concept goal -> an explicit assertion, plus the source-note
common mistakes."""
from langchain_core.tools import BaseTool

from src.decision import decide
from src.faithfulness import score
from src.generator import REFUSAL
from src.graph import run_agent
from src.rag_tool import rag_tool
from src.remediation import advise, REMEDIATION


# --- Concept 1: RAG as a TOOL (not a fixed pipeline) ---
def test_concept_rag_is_a_tool():
    assert isinstance(rag_tool, BaseTool) and rag_tool.name == "rag_v1_complete"


# --- Concept 2: when the agent SHOULD vs SHOULDN'T retrieve ---
def test_concept_should_vs_shouldnt_retrieve():
    assert decide("What is the Helios query quota?").route == "retrieve"      # SHOULD
    assert decide("Given that quota is 1000, is 500 under it?").route == "direct"  # SHOULDN'T
    # and the graph honors it: a direct route never touches retrieval
    assert run_agent("Could you rephrase your previous answer?")["attempts_log"] == []


# --- Concept 3: read a RAGAS score table and know what to fix ---
def test_concept_read_ragas_table_to_know_what_to_fix():
    # every failure mode maps to a concrete remediation
    for mode in ("NO_RETRIEVAL", "WRONG_CHUNKS", "UNSUPPORTED_GENERATION", "GROUNDED", "ABSTAINED"):
        assert mode in REMEDIATION
    assert "retriev" in advise("WRONG_CHUNKS").lower()
    assert "generat" in advise("UNSUPPORTED_GENERATION").lower()


# --- Concept 4: trace a hallucination back to its ROOT failure mode ---
def test_concept_trace_retrieval_failure_vs_generation_failure():
    d3 = [{"doc_id": "D3", "label": "Doc 1", "score": 0.9, "rank": 1,
           "text": "Helios query quota: the free tier permits one thousand queries each day."}]
    # retrieval root cause: question grounds weakly, answer guesses
    weak = [{"doc_id": "D6", "label": "Doc 1", "score": 0.3, "rank": 1,
             "text": "The Helios dashboard displays latency charts."}]
    r_retrieval = score("How long does Helios keep documents?",
                        "keep documents deleting is described in the retrieved context [Doc 1].", weak)
    assert r_retrieval.failure_mode == "WRONG_CHUNKS"
    # generation root cause: chunk grounds the question, but the answer adds an
    # unsupported claim as its own sentence (the model drifting beyond the context).
    r_generation = score(
        "What is the Helios query quota each day?",
        "The quota is one thousand queries each day [Doc 1]. "
        "Weekends are unlimited and unmetered for every tier [Doc 1].",
        d3)
    assert r_generation.failure_mode == "UNSUPPORTED_GENERATION"


# --- Common mistake 1: unbounded retry loop ---
def test_mistake_retry_loop_is_bounded():
    final = run_agent("What is the Helios carbon footprint per query?")
    assert final["attempt"] <= 3            # 1 initial + max 2 retries


# --- Common mistake 2: serving the LAST attempt instead of the BEST ---
def test_mistake_serve_best_not_last():
    final = run_agent("What is the Helios carbon footprint per query?")
    assert final["faithfulness"] == max(a["faithfulness"] for a in final["attempts_log"])


# --- Common mistake 3: a correct refusal mis-scored as a hallucination ---
def test_mistake_refusal_is_abstained_not_hallucination():
    chunks = [{"doc_id": "D1", "label": "Doc 1", "score": 0.9, "rank": 1, "text": "Helios retention."}]
    assert score("q", REFUSAL, chunks).failure_mode == "ABSTAINED"
