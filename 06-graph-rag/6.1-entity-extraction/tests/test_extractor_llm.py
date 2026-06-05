import json

from src.extractor_llm import build_messages, parse_response
from src.schema import EntityType


def test_build_messages_includes_schema_and_text():
    msgs = build_messages("Some document text about Claude.")
    assert msgs[0]["role"] == "system"
    assert "JSON" in msgs[0]["content"]
    assert "Claude" in msgs[1]["content"]


def test_parse_response_extracts_entities_and_triples():
    raw = json.dumps(
        {
            "entities": [
                {"name": "Claude", "type": "MODEL", "aliases": []},
                {"name": "NaturalQuestions", "type": "DATASET", "aliases": ["NQ"]},
            ],
            "triples": [
                {
                    "source": "Claude",
                    "relation": "evaluated_on",
                    "target": "NaturalQuestions",
                    "evidence": "Claude was evaluated on NQ.",
                }
            ],
        }
    )
    result = parse_response(raw)
    assert {e.name for e in result.entities} == {"Claude", "NaturalQuestions"}
    assert result.entities[1].type == EntityType.DATASET
    assert result.triples[0].relation == "evaluated_on"


def test_parse_response_tolerates_code_fence():
    raw = "```json\n{\"entities\": [], \"triples\": []}\n```"
    result = parse_response(raw)
    assert result.entities == [] and result.triples == []
