from src import extractor
from src.loader import Page
from src.schema import Entity, EntityType, ExtractionResult


def test_routes_to_offline_by_default(monkeypatch, sample_text):
    monkeypatch.setattr(extractor.config, "use_llm", lambda: False)
    called = {"offline": False}

    def fake_offline(pages):
        called["offline"] = True
        return ExtractionResult(entities=[Entity(name="X", type=EntityType.CONCEPT)], triples=[])

    monkeypatch.setattr(extractor, "extract_offline", fake_offline)
    extractor.extract([Page(page_num=1, text=sample_text)])
    assert called["offline"] is True


def test_routes_to_llm_when_enabled(monkeypatch, sample_text):
    monkeypatch.setattr(extractor.config, "use_llm", lambda: True)
    called = {"llm": False}

    def fake_llm(pages):
        called["llm"] = True
        return ExtractionResult(entities=[Entity(name="Y", type=EntityType.MODEL)], triples=[])

    monkeypatch.setattr(extractor, "extract_llm", fake_llm)
    extractor.extract([Page(page_num=1, text=sample_text)])
    assert called["llm"] is True
