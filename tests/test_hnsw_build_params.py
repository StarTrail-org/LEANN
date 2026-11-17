"""`leann build --graph-degree/--complexity/--num-threads` must reach the HNSW builder too."""

import os
import platform

import numpy as np
import pytest
from leann.cli import LeannCLI


def _build_and_read_params(tmp_path, **build_kwargs) -> tuple[int, int]:
    """Build a small HNSW index and return the (M, efConstruction) stored in it."""
    pytest.importorskip("leann_backend_hnsw")
    from leann_backend_hnsw import faiss
    from leann_backend_hnsw.hnsw_backend import HNSWBuilder

    data = np.random.default_rng(0).standard_normal((64, 8)).astype(np.float32)
    builder = HNSWBuilder(is_recompute=False, is_compact=False, **build_kwargs)
    builder.build(data, [str(i) for i in range(len(data))], str(tmp_path / "params.leann"))
    index = faiss.read_index(str(tmp_path / "params.index"))
    # FAISS gives level 0 of an HNSW graph 2*M neighbor slots.
    return index.hnsw.nb_neighbors(0) // 2, index.hnsw.efConstruction


@pytest.mark.parametrize(
    ("build_kwargs", "expected"),
    [
        ({}, (32, 200)),
        ({"graph_degree": 8, "complexity": 16}, (8, 16)),
        ({"graph_degree": 8, "complexity": 16, "M": 12, "efConstruction": 40}, (12, 40)),
    ],
)
def test_hnsw_builder_honors_graph_build_options(tmp_path, build_kwargs, expected):
    assert _build_and_read_params(tmp_path, **build_kwargs) == expected


@pytest.mark.parametrize(
    ("build_kwargs", "expected"),
    [({"num_threads": 3}, 3), ({}, os.cpu_count() if platform.system() == "Linux" else None)],
)
def test_hnsw_builder_builds_with_num_threads(tmp_path, monkeypatch, build_kwargs, expected):
    pytest.importorskip("leann_backend_hnsw")
    from leann_backend_hnsw import faiss

    class RecordingIndex(faiss.IndexHNSWFlat):
        def add(self, n, x):
            threads.append(faiss.omp_get_max_threads())
            return super().add(n, x)

    threads: list[int] = []
    monkeypatch.setattr(faiss, "IndexHNSWFlat", RecordingIndex)
    original_threads = faiss.omp_get_max_threads()
    _build_and_read_params(tmp_path, **build_kwargs)

    assert threads == [expected or original_threads]
    assert faiss.omp_get_max_threads() == original_threads


def test_build_cli_forwards_backend_options_only_when_given():
    parser = LeannCLI().create_parser()

    defaults = parser.parse_args(["build", "idx", "--docs", "."])
    assert LeannCLI._backend_build_options(defaults) == {}

    flags = ["--graph-degree", "8", "--complexity", "16", "--num-threads", "4"]
    explicit = parser.parse_args(["build", "idx", "--docs", ".", *flags])
    assert LeannCLI._backend_build_options(explicit) == {
        "graph_degree": 8,
        "complexity": 16,
        "num_threads": 4,
    }
