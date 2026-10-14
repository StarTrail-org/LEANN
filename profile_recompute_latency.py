#!/usr/bin/env python3
"""
Profile recompute latency to identify bottlenecks in LEANN search.

This script reproduces issue #177 and profiles where time is spent:
- Server startup time
- Model loading time
- Embedding computation time
- ZMQ communication overhead
- Query processing time
"""

import cProfile
import pstats

# Add leann-core to path
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent / "packages" / "leann-core" / "src"))

from leann import LeannSearcher


class ProfiledSearcher:
    """Wrapper around LeannSearcher that profiles each operation."""

    def __init__(self, index_path: str, **kwargs):
        self.index_path = index_path
        self.timings = {}
        self.searcher: Optional[LeannSearcher] = None

    def initialize(self):
        """Initialize searcher and measure time."""
        print("\n" + "=" * 60)
        print("PROFILING: Searcher Initialization")
        print("=" * 60)

        start = time.time()
        self.searcher = LeannSearcher(self.index_path, recompute_embeddings=True)
        init_time = time.time() - start

        self.timings["initialization"] = init_time
        print(f"✓ Initialization: {init_time:.3f}s")
        return self.searcher

    def search_with_profiling(self, query: str, top_k: int = 3):
        """Perform search with detailed profiling."""
        print("\n" + "=" * 60)
        print(f"PROFILING: Search Query '{query}'")
        print("=" * 60)

        if not self.searcher:
            self.initialize()

        # Profile the entire search
        profiler = cProfile.Profile()
        profiler.enable()

        total_start = time.time()

        # Check if server is already running
        server_check_start = time.time()
        has_server = hasattr(self.searcher.backend_impl, "embedding_server_manager")
        if has_server:
            manager = self.searcher.backend_impl.embedding_server_manager
            server_running = (
                manager.server_process is not None and manager.server_process.poll() is None
            )
        else:
            server_running = False
        server_check_time = time.time() - server_check_start

        if not server_running:
            print("  ⚠️ Server not running, will start during search...")

        # Measure query embedding computation
        embedding_start = time.time()
        self.searcher.backend_impl.compute_query_embedding(
            query,
            use_server_if_available=True,
        )
        embedding_time = time.time() - embedding_start

        # Measure actual search
        search_start = time.time()
        results = self.searcher.search(query, top_k=top_k, recompute_embeddings=True)
        search_time = time.time() - search_start

        total_time = time.time() - total_start

        profiler.disable()

        # Print timing breakdown
        print("\n⏱️ TIMING BREAKDOWN:")
        print(f"  Total search time: {total_time:.3f}s")
        print(f"  ├─ Server check: {server_check_time:.6f}s")
        print(
            f"  ├─ Query embedding: {embedding_time:.3f}s ({embedding_time / total_time * 100:.1f}%)"
        )
        print(f"  └─ Graph search: {search_time:.3f}s ({search_time / total_time * 100:.1f}%)")

        # Profile stats
        print("\n📊 PROFILER STATS (top 20 by cumulative time):")
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        stats.print_stats(20)

        # Check for model reloads
        print("\n🔍 MODEL RELOAD CHECK:")
        if has_server:
            print(
                f"  Server process PID: {manager.server_process.pid if manager.server_process else 'None'}"
            )
            print(f"  Server port: {manager.server_port}")
            print(f"  Server running: {server_running}")

        return results, {
            "total_time": total_time,
            "embedding_time": embedding_time,
            "search_time": search_time,
            "server_check_time": server_check_time,
        }


def main():
    """Main profiling function."""
    import argparse

    parser = argparse.ArgumentParser(description="Profile LEANN recompute latency")
    parser.add_argument("index_path", help="Path to LEANN index")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["hello", "Test", "function"],
        help="Queries to test (default: hello Test function)",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of results (default: 3)")

    args = parser.parse_args()

    print("=" * 60)
    print("LEANN RECOMPUTE LATENCY PROFILER")
    print("=" * 60)
    print(f"Index: {args.index_path}")
    print(f"Queries: {args.queries}")
    print(f"Top-K: {args.top_k}")

    profiler = ProfiledSearcher(args.index_path)

    # First search (cold start)
    print("\n" + "=" * 60)
    print("COLD START (First Query)")
    print("=" * 60)
    _results1, timings1 = profiler.search_with_profiling(args.queries[0], args.top_k)

    # Subsequent searches (warm)
    for i, query in enumerate(args.queries[1:], 1):
        print("\n" + "=" * 60)
        print(f"WARM QUERY #{i + 1} (Query: '{query}')")
        print("=" * 60)
        _results, timings = profiler.search_with_profiling(query, args.top_k)

        # Compare with first query
        print("\n📈 COMPARISON WITH COLD START:")
        print(f"  Cold start total: {timings1['total_time']:.3f}s")
        print(f"  Warm query total: {timings['total_time']:.3f}s")
        print(f"  Difference: {timings['total_time'] - timings1['total_time']:.3f}s")
        print(f"  Speedup: {timings1['total_time'] / timings['total_time']:.2f}x")

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
