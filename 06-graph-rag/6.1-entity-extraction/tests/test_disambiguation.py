from src.disambiguation import resolve
from src.schema import Entity, EntityType, ExtractionResult, Triple


def test_merges_alias_pair_and_rewrites_triples():
    result = ExtractionResult(
        entities=[
            Entity(name="University of California, Berkeley", type=EntityType.ORG, mentions=2),
            Entity(name="UC Berkeley", type=EntityType.ORG, mentions=1),
            Entity(name="Kevin Lin", type=EntityType.PERSON),
        ],
        triples=[Triple(source="Kevin Lin", relation="works_at", target="UC Berkeley")],
    )
    out = resolve(result)
    names = {e.name for e in out.entities}
    # the two Berkeley forms collapse to one canonical node
    assert sum(1 for n in names if "Berkeley" in n) == 1
    canonical = next(e for e in out.entities if "Berkeley" in e.name)
    assert "UC Berkeley" in canonical.aliases or canonical.name == "UC Berkeley"
    # the triple was rewritten to the canonical target
    assert out.triples[0].target == canonical.name


def test_does_not_merge_across_types():
    result = ExtractionResult(
        entities=[
            Entity(name="Claude", type=EntityType.MODEL),
            Entity(name="Claude", type=EntityType.PERSON),
        ],
        triples=[],
    )
    out = resolve(result)
    assert len(out.entities) == 2   # same string, different type -> kept apart
