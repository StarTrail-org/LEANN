#!/usr/bin/env python3
"""
Benchmark to demonstrate cache improvements without requiring full LEANN installation.
Simulates the query embedding computation and caching behavior.
"""

import hashlib
import json
import time
from typing import Optional

import numpy as np


class QueryEmbeddingCache:
    """Hash-based cache for query embeddings to avoid recomputation."""

    def __init__(self, max_size: int = 1000):
        self.cache: dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _hash_query(self, query: str, query_template: Optional[str] = None) -> str:
        """Create hash key for query."""
        key_data = {
            "query": query,
            "template": query_template or "",
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, query: str, query_template: Optional[str] = None) -> Optional[np.ndarray]:
        """Get cached embedding if exists."""
        key = self._hash_query(query, query_template)
        result = self.cache.get(key)
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1
        return result

    def put(self, query: str, embedding: np.ndarray, query_template: Optional[str] = None):
        """Cache embedding."""
        key = self._hash_query(query, query_template)

        # Simple LRU: remove oldest if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        self.cache[key] = embedding.copy()


def simulate_expensive_embedding(query: str, latency_ms: float = 15000) -> np.ndarray:
    """
    Simulate expensive embedding computation.
    Issue #177 reports 13-19s per query, using 15s as average.
    """
    # Scale down for faster testing (use 150ms instead of 15000ms)
    scaled_latency = latency_ms / 100
    time.sleep(scaled_latency / 1000)
    return np.random.rand(384)  # Typical embedding dimension


def benchmark_without_cache(queries: list[str], latency_ms: float = 15000):
    """Benchmark without caching (current behavior from issue #177)."""
    print("\n" + "=" * 60)
    print("BENCHMARK: WITHOUT CACHE (Current Behavior)")
    print("=" * 60)

    total_start = time.time()
    times = []

    for i, query in enumerate(queries, 1):
        start = time.time()
        simulate_expensive_embedding(query, latency_ms)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Query {i} ('{query}'): {elapsed * 1000:.1f}ms")

    total_time = time.time() - total_start
    avg_time = sum(times) / len(times)

    print(f"\n  Total time: {total_time:.2f}s")
    print(f"  Average per query: {avg_time * 1000:.1f}ms")
    print(f"  Estimated real-world (100x scale): {total_time * 100:.1f}s")

    return total_time, times


def benchmark_with_cache(queries: list[str], latency_ms: float = 15000):
    """Benchmark with caching (optimized behavior)."""
    print("\n" + "=" * 60)
    print("BENCHMARK: WITH CACHE (Optimized Behavior)")
    print("=" * 60)

    cache = QueryEmbeddingCache(max_size=1000)
    total_start = time.time()
    times = []

    for i, query in enumerate(queries, 1):
        start = time.time()

        # Check cache first
        cached = cache.get(query)
        if cached is not None:
            embedding = cached
            cache_hit = True
        else:
            embedding = simulate_expensive_embedding(query, latency_ms)
            cache.put(query, embedding)
            cache_hit = False

        elapsed = time.time() - start
        times.append(elapsed)
        status = "CACHE HIT" if cache_hit else "COMPUTED"
        print(f"  Query {i} ('{query}'): {elapsed * 1000:.1f}ms [{status}]")

    total_time = time.time() - total_start
    avg_time = sum(times) / len(times)

    print(f"\n  Total time: {total_time:.2f}s")
    print(f"  Average per query: {avg_time * 1000:.1f}ms")
    print(f"  Cache hits: {cache.hits}/{len(queries)} ({cache.hits / len(queries) * 100:.1f}%)")
    print(f"  Cache misses: {cache.misses}/{len(queries)}")
    print(f"  Estimated real-world (100x scale): {total_time * 100:.1f}s")

    return total_time, times, cache


def main():
    """Run benchmarks to demonstrate cache improvements."""
    print("=" * 60)
    print("LEANN QUERY EMBEDDING CACHE BENCHMARK")
    print("=" * 60)
    print("\nSimulating issue #177 scenario:")
    print("  - Each query takes 13-19s (using 15s average)")
    print("  - Scaled down 100x for faster testing (150ms per query)")
    print("  - Testing with repeated queries to show cache benefit")
    print()

    # Test queries - includes repetitions to show cache benefit
    queries = [
        "hello world",
        "search function",
        "Test query",
        "hello world",  # Repeat
        "another query",
        "search function",  # Repeat
        "hello world",  # Repeat again
        "Test query",  # Repeat
        "final query",
        "hello world",  # Repeat many times
    ]

    print(f"Testing with {len(queries)} queries:")
    unique_queries = set(queries)
    print(f"  Unique queries: {len(unique_queries)}")
    print(f"  Repeated queries: {len(queries) - len(unique_queries)}")
    print()

    # Benchmark without cache
    time_without, _times_without = benchmark_without_cache(queries)

    # Benchmark with cache
    time_with, times_with, cache = benchmark_with_cache(queries)

    # Calculate improvements
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("\nWithout cache:")
    print(f"  Total time: {time_without:.2f}s")
    print(f"  Est. real-world: {time_without * 100:.1f}s ({time_without * 100 / 60:.1f} minutes)")

    print("\nWith cache:")
    print(f"  Total time: {time_with:.2f}s")
    print(f"  Est. real-world: {time_with * 100:.1f}s ({time_with * 100 / 60:.1f} minutes)")
    print(f"  Cache hit rate: {cache.hits}/{len(queries)} ({cache.hits / len(queries) * 100:.1f}%)")

    speedup = time_without / time_with
    time_saved = time_without - time_with
    time_saved_real = time_saved * 100

    print("\nImprovement:")
    print(f"  Speedup: {speedup:.2f}x faster")
    print(f"  Time saved (scaled): {time_saved:.2f}s")
    print(
        f"  Time saved (real-world est.): {time_saved_real:.1f}s ({time_saved_real / 60:.1f} minutes)"
    )

    # Per-query analysis
    print("\nPer-query breakdown:")
    cache_hits = [i for i, q in enumerate(queries) if queries[:i].count(q) > 0]
    cache_misses = [i for i in range(len(queries)) if i not in cache_hits]

    if cache_hits:
        avg_hit_time = sum(times_with[i] for i in cache_hits) / len(cache_hits)
        print(
            f"  Avg cached query: {avg_hit_time * 1000:.3f}ms (est. real: {avg_hit_time * 100 * 1000:.1f}ms)"
        )

    if cache_misses:
        avg_miss_time = sum(times_with[i] for i in cache_misses) / len(cache_misses)
        print(
            f"  Avg uncached query: {avg_miss_time * 1000:.1f}ms (est. real: {avg_miss_time * 100:.0f}s)"
        )

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(
        f"\nFor issue #177 workload with {cache.hits / len(queries) * 100:.0f}% repeated queries:"
    )
    print("  - WITHOUT cache: Every query takes ~15s")
    print("  - WITH cache: Repeated queries are near-instant")
    print(f"  - Overall speedup: {speedup:.1f}x")
    print("\nThis demonstrates the theoretical improvement from PR #226.")
    print("Real-world performance will vary based on:")
    print("  - Cache hit rate (how many queries are repeated)")
    print("  - ZMQ connection reuse overhead reduction (~10-50ms per query)")
    print("  - Model loading and server startup optimizations")


if __name__ == "__main__":
    main()
