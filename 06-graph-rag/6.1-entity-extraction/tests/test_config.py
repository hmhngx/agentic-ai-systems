import importlib

from src import config


def test_use_llm_requires_flag_and_key(monkeypatch):
    monkeypatch.setenv("USE_LLM", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    importlib.reload(config)
    assert config.use_llm() is False          # flag without key -> offline

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert config.use_llm() is True


def test_use_embeddings_api_tracks_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    importlib.reload(config)
    assert config.use_embeddings_api() is False
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert config.use_embeddings_api() is True
