"""Unit tests for QueryEmbeddingCache (no native backends required)."""

import numpy as np
from leann.searcher_base import QueryEmbeddingCache


class TestQueryEmbeddingCache:
    def test_put_get_roundtrip(self):
        cache = QueryEmbeddingCache(max_size=8)
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cache.put("hello", emb)
        got = cache.get("hello")
        assert got is not None
        assert got.shape == (3,)
        assert np.allclose(got, emb)

    def test_template_is_part_of_key(self):
        cache = QueryEmbeddingCache(max_size=8)
        cache.put("q", np.ones(4, dtype=np.float32), query_template="A: ")
        assert cache.get("q", query_template="A: ") is not None
        assert cache.get("q", query_template="B: ") is None
        assert cache.get("q") is None

    def test_cache_miss_returns_none(self):
        cache = QueryEmbeddingCache(max_size=8)
        assert cache.get("missing") is None

    def test_lru_eviction(self):
        cache = QueryEmbeddingCache(max_size=2)
        cache.put("a", np.array([1.0], dtype=np.float32))
        cache.put("b", np.array([2.0], dtype=np.float32))
        cache.put("c", np.array([3.0], dtype=np.float32))  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_put_normalizes_2d_to_1d(self):
        cache = QueryEmbeddingCache(max_size=4)
        cache.put("q", np.array([[9.0, 8.0, 7.0]], dtype=np.float32))
        got = cache.get("q")
        assert got is not None
        assert got.shape == (3,)
        assert np.allclose(got, [9.0, 8.0, 7.0])

    def test_clear(self):
        cache = QueryEmbeddingCache(max_size=4)
        cache.put("q", np.ones(2, dtype=np.float32))
        cache.clear()
        assert cache.get("q") is None
