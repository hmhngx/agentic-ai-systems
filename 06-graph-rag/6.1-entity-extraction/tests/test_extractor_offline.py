from src.extractor_offline import extract_offline
from src.loader import Page
from src.schema import EntityType


def test_extracts_known_entities(sample_text):
    result = extract_offline([Page(page_num=1, text=sample_text)])
    names = {e.name for e in result.entities}
    # people and orgs from spaCy NER
    assert any("Stanford" in n for n in names)
    assert any(n == "Nelson Liu" or "Nelson" in n for n in names)
    # domain-lexicon types
    types = {e.type for e in result.entities}
    assert EntityType.MODEL in types        # GPT-3.5-Turbo / Claude
    assert EntityType.DATASET in types      # NaturalQuestions


def test_emits_triples_between_cooccurring_entities(sample_text):
    result = extract_offline([Page(page_num=1, text=sample_text)])
    assert len(result.triples) > 0
    # every triple references real entity names
    names = {e.name for e in result.entities}
    for t in result.triples:
        assert t.source in names and t.target in names
        assert t.evidence  # carries the sentence


def test_is_deterministic(sample_text):
    r1 = extract_offline([Page(page_num=1, text=sample_text)])
    r2 = extract_offline([Page(page_num=1, text=sample_text)])
    assert [e.name for e in r1.entities] == [e.name for e in r2.entities]
    assert [(t.source, t.relation, t.target) for t in r1.triples] == \
           [(t.source, t.relation, t.target) for t in r2.triples]
