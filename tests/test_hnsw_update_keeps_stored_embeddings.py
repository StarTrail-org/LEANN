"""Appending to an HNSW index must keep the storage mode its metadata records.

``leann build --no-recompute`` writes an index that keeps its embeddings and records
``is_pruned: false``. A later ``leann build`` without the flag appends new files through a
builder whose ``is_recompute`` is the CLI default, True. ``update_index()`` let that builder
flag switch the append to recompute mode: it started an embedding server, and after the append
it pruned the embeddings from ``documents.index`` while the metadata still said they were
there. Every search then crashed with a segmentation fault, with or without ``--recompute``.
"""

import json
import sys

import leann.api as api
import leann.embedding_compute as embedding_compute
import numpy as np
import pytest
from leann.api import LeannBuilder, LeannSearcher

VECTORS = {
    text: np.random.default_rng(ord(text)).standard_normal(8).astype(np.float32) for text in "abcd"
}


def _embed(texts, *args, **kwargs):
    return np.stack([VECTORS[text] for text in texts])


def _builder(is_recompute: bool) -> LeannBuilder:
    return LeannBuilder(
        backend_name="hnsw",
        dimensions=8,
        distance_metric="cosine",
        is_compact=False,
        is_recompute=is_recompute,
    )


class _UnavailableEmbeddingServer:
    """An index that keeps its embeddings never needs the recompute server; this one
    cannot start, so the test needs no embedding model."""

    def __init__(self, *args, **kwargs):
        pass

    def start_server(self, port, *args, **kwargs):
        return False, port

    def stop_server(self):
        pass


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="On macOS runners the native ``index.add()`` call that ``update_index()`` makes "
    "aborts inside the FAISS bindings, taking the whole pytest session with it.",
)
def test_append_with_default_builder_keeps_a_no_recompute_index_searchable(tmp_path, monkeypatch):
    index = tmp_path / "docs.leann"
    monkeypatch.setattr(api, "compute_embeddings", _embed)
    monkeypatch.setattr(embedding_compute, "compute_embeddings", _embed)
    monkeypatch.setattr(api, "EmbeddingServerManager", _UnavailableEmbeddingServer)

    builder = _builder(is_recompute=False)
    for text in "abc":
        builder.add_text(text)
    builder.build_index(str(index))

    # What `leann build` hands update_index() when --no-recompute is not repeated.
    updater = _builder(is_recompute=True)
    updater.add_text("d")
    updater.update_index(str(index))

    meta = json.loads((tmp_path / "docs.leann.meta.json").read_text(encoding="utf-8"))
    assert meta["is_pruned"] is False

    from leann_backend_hnsw import faiss

    stored = faiss.read_index(str(tmp_path / "docs.index"))
    assert stored.ntotal == 4
    assert stored.storage is not None and stored.storage.ntotal == 4

    searcher = LeannSearcher(str(index), recompute_embeddings=False, enable_warmup=False)
    try:
        hits = searcher.search("d", top_k=1)
    finally:
        searcher.cleanup()
    assert [hit.text for hit in hits] == ["d"]
