from src.text import tokenize, content_tokens, STOPWORDS


def test_tokenize_lowercases_and_splits_alnum():
    assert tokenize("Helios keeps 90 Documents!") == ["helios", "keeps", "90", "documents"]


def test_content_tokens_drops_stopwords_keeps_oov_content():
    # "how", "does", "the" are stopwords; "keep"/"documents" are content (kept even if OOV).
    assert content_tokens("How does the Helios keep documents") == ["helios", "keep", "documents"]


def test_stopwords_contains_function_words():
    for w in ("the", "is", "a", "per", "what", "how", "does"):
        assert w in STOPWORDS
