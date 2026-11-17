"""Cosine HNSW indexes must rank recomputed passages by cosine, not by raw inner product."""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("leann_backend_hnsw")

# Each passage text is its own embedding (see tests/support/vector_text_hnsw_server.py).
PASSAGES = {
    "aligned-short": "0.5,0,0,0",  # cosine 1.0,   dot 0.5
    "near-unit": "1,0.05,0,0",  # cosine 0.999, dot 1.0
    "skewed-long": "6,5,0,0",  # cosine 0.768, dot 6.0
    "diagonal": "2,2,0,0",  # cosine 0.707, dot 2.0
    "orthogonal": "0,0,3,0",  # cosine 0.0,   dot 0.0
}
QUERY = "1,0,0,0"


@pytest.fixture
def vector_text_server(monkeypatch):
    """Serve recompute requests with the real HNSW embedding server and a model-free embedder."""
    from leann import searcher_base
    from leann.embedding_server_manager import EmbeddingServerManager

    support = str(Path(__file__).resolve().parent / "support")
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", support + (os.pathsep + existing if existing else ""))
    monkeypatch.setattr(
        searcher_base,
        "EmbeddingServerManager",
        lambda backend_module_name: EmbeddingServerManager("vector_text_hnsw_server"),
    )


@pytest.mark.parametrize("is_compact", [True, False])
def test_cosine_recompute_ranks_by_cosine(tmp_path, vector_text_server, is_compact):
    from leann.api import LeannBuilder, LeannSearcher

    ids = list(PASSAGES)
    vectors = np.array([[float(v) for v in PASSAGES[i].split(",")] for i in ids], dtype=np.float32)
    index_path = str(tmp_path / "cosine.leann")
    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model="vector-text",
        dimensions=vectors.shape[1],
        distance_metric="cosine",
        is_recompute=True,
        is_compact=is_compact,
    )
    for pid in ids:
        builder.add_text(PASSAGES[pid])
    builder.build_index_from_arrays(index_path, ids, vectors)

    with LeannSearcher(index_path, enable_warmup=False, use_daemon=False) as searcher:
        results = searcher.search(QUERY, top_k=3, recompute_embeddings=True)

    assert [r.id for r in results] == ["aligned-short", "near-unit", "skewed-long"]
    assert [round(r.score, 3) for r in results] == [1.0, 0.999, 0.768]
