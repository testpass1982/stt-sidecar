"""TDD Red: Ollama embeddings via nomic-embed-text.
Should fail because SearchEngine.embed() doesn't exist yet."""
import pytest


def test_embed_returns_768_dimensions():
    """nomic-embed-text produces 768-dimensional vectors."""
    from stt_sidecar.search_engine import SearchEngine
    engine = SearchEngine(":memory:")
    v = engine.embed("hello world")
    assert isinstance(v, list)
    assert len(v) == 768


def test_embed_returns_floats():
    """All elements of the embedding vector are floats."""
    from stt_sidecar.search_engine import SearchEngine
    engine = SearchEngine(":memory:")
    v = engine.embed("authentication")
    assert all(isinstance(x, float) for x in v)


def test_semantic_similarity():
    """"authentication" and "login" have higher cosine than "authentication" and "weather"."""
    from stt_sidecar.search_engine import SearchEngine
    engine = SearchEngine(":memory:")
    v_auth = engine.embed("authentication")
    v_login = engine.embed("login")
    v_weather = engine.embed("weather")

    def cosine(a, b):
        dot = sum(x*y for x,y in zip(a,b))
        na = sum(x*x for x in a)**0.5
        nb = sum(x*x for x in b)**0.5
        return dot / (na*nb) if na and nb else 0.0

    sim_auth_login = cosine(v_auth, v_login)
    sim_auth_weather = cosine(v_auth, v_weather)

    assert sim_auth_login > 0.4, f"auth↔login cosine={sim_auth_login:.3f} too low"
    assert sim_auth_weather < sim_auth_login, \
        f"auth↔weather={sim_auth_weather:.3f} should be less than auth↔login={sim_auth_login:.3f}"


def test_empty_string_returns_zero():
    """Empty string fallback returns a zero vector (not a crash)."""
    from stt_sidecar.search_engine import SearchEngine
    engine = SearchEngine(":memory:")
    v = engine.embed("")
    assert isinstance(v, list)
    assert len(v) == 768
    assert all(x == 0.0 for x in v)


def test_russian_embedding():
    """Non-ASCII (русский) text produces a valid embedding vector."""
    from stt_sidecar.search_engine import SearchEngine
    engine = SearchEngine(":memory:")
    v = engine.embed("аутентификация и вход в систему")
    assert isinstance(v, list)
    assert len(v) == 768
    assert any(x != 0.0 for x in v), "russian text should produce non-zero embedding"


def test_repeated_embed_is_consistent():
    """Same text twice returns identical (or nearly identical) vectors."""
    from stt_sidecar.search_engine import SearchEngine
    engine = SearchEngine(":memory:")
    v1 = engine.embed("user registration form")
    v2 = engine.embed("user registration form")
    assert v1 == v2  # deterministic model — exact match expected
