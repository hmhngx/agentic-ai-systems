import json

from langchain_core.tools import BaseTool

from src.rag_tool import rag_v1_complete, rag_tool, RagResult


def test_rag_v1_complete_returns_result_with_chunks_and_answer():
    res = rag_v1_complete("What is the Helios query quota per day?")
    assert isinstance(res, RagResult)
    assert res.chunks and res.chunks[0]["doc_id"] == "D3"
    assert "quota" in res.answer.lower()
    assert res.citations == ["Doc 1"]


def test_query_defaults_to_question_but_can_differ():
    res = rag_v1_complete("How long does Helios keep documents?",
                          query="Helios documents retained retention purged")
    assert res.chunks[0]["doc_id"] == "D1"   # refined query sharpens onto D1


def test_rag_tool_is_a_langchain_tool():
    assert isinstance(rag_tool, BaseTool)
    assert rag_tool.name == "rag_v1_complete"
    assert "retriev" in rag_tool.description.lower()


def test_rag_tool_invoke_returns_json_string():
    out = rag_tool.invoke({"query": "What is the Helios query quota per day?"})
    payload = json.loads(out)
    assert payload["citations"] == ["Doc 1"]
    assert "quota" in payload["answer"].lower()
