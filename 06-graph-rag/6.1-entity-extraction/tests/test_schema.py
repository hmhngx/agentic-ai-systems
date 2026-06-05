from src.schema import Entity, EntityType, Triple, ExtractionResult


def test_entity_normalizes_and_defaults():
    e = Entity(name="  Stanford University  ", type=EntityType.ORG)
    assert e.name == "Stanford University"   # whitespace stripped
    assert e.type == EntityType.ORG
    assert e.aliases == []
    assert e.mentions == 1


def test_triple_holds_relation_and_evidence():
    t = Triple(
        source="Nelson Liu",
        relation="works_at",
        target="Stanford University",
        evidence="Nelson Liu ... at Stanford University.",
    )
    assert t.source == "Nelson Liu"
    assert t.relation == "works_at"
    assert t.target == "Stanford University"
    assert "Stanford" in t.evidence


def test_extraction_result_roundtrips_json():
    r = ExtractionResult(
        entities=[Entity(name="Claude", type=EntityType.MODEL)],
        triples=[Triple(source="Claude", relation="evaluated_on", target="NaturalQuestions")],
    )
    again = ExtractionResult.model_validate_json(r.model_dump_json())
    assert again.entities[0].name == "Claude"
    assert again.triples[0].relation == "evaluated_on"
