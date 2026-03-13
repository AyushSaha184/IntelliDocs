import importlib
import time
import uuid
from typing import List, Optional

import numpy as np
import pytest


@pytest.fixture()
def redis_env(monkeypatch):
    prefix = f"pytest-rag-{uuid.uuid4().hex}"
    monkeypatch.setenv("REDIS_PREFIX", prefix)
    monkeypatch.setenv("REDIS_DEFAULT_TTL_SECONDS", "60")
    monkeypatch.setenv("REDIS_RETRIEVAL_TTL_SECONDS", "60")
    monkeypatch.setenv("REDIS_LLM_TTL_SECONDS", "60")
    monkeypatch.setenv("REDIS_EMBEDDING_TTL_SECONDS", "60")

    import backend.cache.RedisCache as redis_cache_mod

    importlib.reload(redis_cache_mod)
    client = redis_cache_mod.get_redis_client()
    if client is None:
        pytest.skip("redis package unavailable")

    try:
        client.ping()
    except Exception as exc:
        pytest.skip(f"Redis server unavailable for integration tests: {exc}")

    yield redis_cache_mod, client, prefix

    try:
        keys = client.keys(f"{prefix}:*")
        if keys:
            client.delete(*keys)
    except Exception:
        pass


@pytest.fixture()
def cache_modules(redis_env):
    redis_cache_mod, _, _ = redis_env

    import src.modules.QueryCache as query_cache_mod
    import src.modules.Embeddings as embeddings_mod

    importlib.reload(redis_cache_mod)
    importlib.reload(query_cache_mod)
    importlib.reload(embeddings_mod)

    return query_cache_mod, embeddings_mod, redis_cache_mod


def test_retrieval_cache_set_get_and_key_distinctions(cache_modules):
    query_cache_mod, _, _ = cache_modules
    cache = query_cache_mod.RetrievalCache()

    payload = {
        "retrieved_chunks": ["chunk-a"],
        "metadata": [{"chunk_id": "1", "score": 0.88}],
        "retrieval_scores": [0.88],
    }

    cache.set_cache("s1", "What is RAG?", 3, payload, retrieval_params={"mode": "hybrid", "alpha": 0.7})
    hit = cache.get_cache("s1", "What is RAG?", 3, retrieval_params={"mode": "hybrid", "alpha": 0.7})
    assert hit == payload

    cache.set_cache("s1", "What is RAG?", 4, {"retrieved_chunks": ["top4"]})
    assert cache.get_cache("s1", "What is RAG?", 3) != cache.get_cache("s1", "What is RAG?", 4)

    cache.set_cache("s1", "What is RAG?", 3, {"retrieved_chunks": ["params-a"]}, retrieval_params={"mode": "dense"})
    cache.set_cache("s1", "What is RAG?", 3, {"retrieved_chunks": ["params-b"]}, retrieval_params={"mode": "hybrid"})
    assert cache.get_cache("s1", "What is RAG?", 3, retrieval_params={"mode": "dense"}) != cache.get_cache(
        "s1", "What is RAG?", 3, retrieval_params={"mode": "hybrid"}
    )


def test_llm_cache_context_hash_semantics(cache_modules):
    query_cache_mod, _, _ = cache_modules
    cache = query_cache_mod.LLMCache()

    response = {"llm_response": "Answer A", "llm_metadata": {"model": "x"}}
    chunks = ["context line 1", "context line 2"]

    cache.set_cache("sess-1", "Explain hybrid retrieval", chunks, response)
    assert cache.get_cache("sess-1", "Explain hybrid retrieval", chunks) == response

    changed_chunks = ["context line 1", "DIFFERENT"]
    assert cache.get_cache("sess-1", "Explain hybrid retrieval", changed_chunks) is None


class _FakeEmbeddingModel:
    def __init__(self):
        self.single_calls = 0
        self.batch_calls = 0
        self.last_batch_inputs: List[str] = []

    @property
    def dimension(self) -> int:
        return 4

    @property
    def model_name(self) -> str:
        return "fake-embed-model"

    def embed(self, text: str) -> Optional[np.ndarray]:
        self.single_calls += 1
        seed = sum(ord(c) for c in text) % 100
        return np.array([seed, seed + 1, seed + 2, seed + 3], dtype=np.float32)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        self.batch_calls += 1
        self.last_batch_inputs = list(texts)
        rows = [self.embed(t) for t in texts]
        return np.stack(rows).astype(np.float32)


def test_embedding_cache_repeat_hits_and_batch_dedup(cache_modules):
    _, embeddings_mod, _ = cache_modules

    model = _FakeEmbeddingModel()
    service = embeddings_mod.EmbeddingService(
        model=model,
        cache_dir="ignored-in-redis-mode",
        use_cache=True,
        max_cache_size=100,
    )

    first = service.embed_text("alpha")
    second = service.embed_text("alpha")
    assert first.embedding is not None
    assert second.embedding is not None
    assert np.array_equal(first.embedding, second.embedding)
    assert model.single_calls == 1

    texts = ["a", "b", "a", "c", "b"]
    results = service.embed_batch(texts, batch_size=8)
    assert len(results) == len(texts)
    assert sorted(model.last_batch_inputs) == ["a", "b", "c"]

    before_calls = model.batch_calls
    _ = service.embed_batch(texts, batch_size=8)
    assert model.batch_calls == before_calls


def test_ttl_expiry_for_retrieval_cache(redis_env, monkeypatch):
    _, _, _ = redis_env
    monkeypatch.setenv("REDIS_RETRIEVAL_TTL_SECONDS", "1")

    import src.modules.QueryCache as query_cache_mod
    importlib.reload(query_cache_mod)

    cache = query_cache_mod.RetrievalCache()
    cache.set_cache("s-ttl", "q", 1, {"retrieved_chunks": ["x"]})
    assert cache.get_cache("s-ttl", "q", 1) == {"retrieved_chunks": ["x"]}

    time.sleep(1.25)
    assert cache.get_cache("s-ttl", "q", 1) is None


def test_redis_failure_degrades_to_cache_miss(monkeypatch):
    import backend.cache.RedisCache as redis_cache_mod
    import src.modules.QueryCache as query_cache_mod
    import src.modules.Embeddings as embeddings_mod

    importlib.reload(redis_cache_mod)
    importlib.reload(query_cache_mod)
    importlib.reload(embeddings_mod)

    def _raise_client_error():
        raise RuntimeError("redis offline")

    monkeypatch.setattr(redis_cache_mod, "get_redis_client", _raise_client_error)

    retrieval = query_cache_mod.RetrievalCache()
    llm = query_cache_mod.LLMCache()

    retrieval.set_cache("s", "q", 1, {"x": 1})
    llm.set_cache("s", "q", ["ctx"], {"x": 2})

    assert retrieval.get_cache("s", "q", 1) is None
    assert llm.get_cache("s", "q", ["ctx"]) is None

    model = _FakeEmbeddingModel()
    service = embeddings_mod.EmbeddingService(model=model, use_cache=True)
    out = service.embed_text("still-works")
    assert out.embedding is not None


def test_retrieval_and_llm_cache_graceful_without_redis_client(monkeypatch):
    import backend.cache.RedisCache as redis_cache_mod
    import src.modules.QueryCache as query_cache_mod

    importlib.reload(redis_cache_mod)
    importlib.reload(query_cache_mod)

    monkeypatch.setattr(redis_cache_mod, "get_redis_client", lambda: None)

    retrieval = query_cache_mod.RetrievalCache()
    llm = query_cache_mod.LLMCache()

    retrieval.set_cache("s", "q", 1, {"k": "v"})
    llm.set_cache("s", "q", ["ctx"], {"k": "v"})

    assert retrieval.get_cache("s", "q", 1) is None
    assert llm.get_cache("s", "q", ["ctx"]) is None


def test_embedding_service_graceful_without_redis_client(monkeypatch):
    import backend.cache.RedisCache as redis_cache_mod
    import src.modules.Embeddings as embeddings_mod

    importlib.reload(redis_cache_mod)
    importlib.reload(embeddings_mod)

    monkeypatch.setattr(redis_cache_mod, "get_redis_client", lambda: None)

    model = _FakeEmbeddingModel()
    service = embeddings_mod.EmbeddingService(model=model, use_cache=True)

    one = service.embed_text("offline")
    two = service.embed_text("offline")

    assert one.embedding is not None
    assert two.embedding is not None
    assert model.single_calls == 2


def test_singleton_interfaces_still_work(cache_modules):
    query_cache_mod, _, _ = cache_modules

    retrieval = query_cache_mod.get_retrieval_cache()
    llm = query_cache_mod.get_llm_cache()

    retrieval.set_cache("compat", "query", 2, {"retrieved_chunks": ["ok"]})
    assert retrieval.get_cache("compat", "query", 2) == {"retrieved_chunks": ["ok"]}

    llm.set_cache("compat", "query", ["ctx"], {"llm_response": "ok"})
    assert llm.get_cache("compat", "query", ["ctx"]) == {"llm_response": "ok"}


def test_singleton_interfaces_graceful_without_redis(monkeypatch):
    import backend.cache.RedisCache as redis_cache_mod
    import src.modules.QueryCache as query_cache_mod

    importlib.reload(redis_cache_mod)
    importlib.reload(query_cache_mod)

    monkeypatch.setattr(redis_cache_mod, "get_redis_client", lambda: None)

    retrieval = query_cache_mod.get_retrieval_cache()
    llm = query_cache_mod.get_llm_cache()

    retrieval.set_cache("compat", "query", 2, {"retrieved_chunks": ["ok"]})
    llm.set_cache("compat", "query", ["ctx"], {"llm_response": "ok"})

    assert retrieval.get_cache("compat", "query", 2) is None
    assert llm.get_cache("compat", "query", ["ctx"]) is None


def test_auxiliary_caches_roundtrip(redis_env):
    _, _, _ = redis_env

    import backend.cache.AuxiliaryCaches as auxiliary_mod
    importlib.reload(auxiliary_mod)

    auxiliary_mod.set_cached_document_summary("doc-1", "summary text")
    assert auxiliary_mod.get_cached_document_summary("doc-1") == "summary text"

    auxiliary_mod.set_cached_approved_answer("sess-1", "What is RAG?", "Redis answer")
    assert auxiliary_mod.get_cached_approved_answer("sess-1", "What is RAG?") == "Redis answer"


def test_auxiliary_caches_graceful_without_redis(monkeypatch):
    import backend.cache.RedisCache as redis_cache_mod
    import backend.cache.AuxiliaryCaches as auxiliary_mod

    importlib.reload(redis_cache_mod)
    importlib.reload(auxiliary_mod)

    monkeypatch.setattr(redis_cache_mod, "get_redis_client", lambda: None)

    auxiliary_mod.set_cached_document_summary("doc-x", "hello")
    auxiliary_mod.set_cached_approved_answer("sess-x", "q", "a")
    assert auxiliary_mod.get_cached_document_summary("doc-x") is None
    assert auxiliary_mod.get_cached_approved_answer("sess-x", "q") is None


class _FakeStore:
    def __init__(self):
        self.data = {}

    def get_json(self, key):
        return self.data.get(key)

    def set_json(self, key, payload):
        self.data[key] = payload
        return True


def test_chunker_token_cache_semantics_without_live_redis():
    from src.modules.Chunking import TextChunker

    chunker = TextChunker(use_db=False)

    text = "This is a token counting cache test string."
    first = chunker.get_token_count(text)
    second = chunker.get_token_count(text)

    assert first == second
    assert chunker._cache_misses >= 1
    assert chunker._cache_hits >= 1
    assert len(chunker._token_cache) >= 1
