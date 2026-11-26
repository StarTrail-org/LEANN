"""``update_index()`` must keep the index's BM25 artifact in step with its passages.

``build_index()`` writes ``<index>.bm25.sqlite`` for keyword and hybrid search
(``LeannSearcher.search(..., vector_weight=...)``), and the searcher reads it whenever the
metadata names it. ``update_index()`` appended and removed passages without touching it: keyword
search never found an appended passage, and after an IVF removal it still matched the removed
text, so the ID that an appended passage took over returned that passage instead.

As in ``test_update_index_auto_passage_ids.py``, the IVF case stubs ``add_vectors()`` and
``remove_ids()``: keyword search reads only the passage store and the BM25 artifact.
"""

import hashlib
import sys

import leann.api as api
import leann_backend_ivf
import numpy as np
import pytest
from leann.api import LeannBuilder, LeannSearcher


def _embed(texts, *args, **kwargs):
    rows = []
    for text in texts:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rows.append(np.random.default_rng(seed).standard_normal(8).astype(np.float32))
    return np.stack(rows)


def _builder(backend_name):
    if backend_name == "ivf":
        return LeannBuilder(backend_name="ivf", dimensions=8, nlist=2)
    return LeannBuilder(backend_name="hnsw", dimensions=8, is_compact=False, is_recompute=False)


def _keyword_hits(index, query: str) -> list[str]:
    searcher = LeannSearcher(str(index), recompute_embeddings=False, enable_warmup=False)
    try:
        return [hit.text for hit in searcher.search(query, top_k=3, vector_weight=0.0)]
    finally:
        searcher.cleanup()


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="On macOS runners the native ``index.add()`` call that ``update_index()`` makes "
    "aborts inside the FAISS bindings, taking the whole pytest session with it.",
)
def test_keyword_search_finds_a_passage_appended_to_an_hnsw_index(tmp_path, monkeypatch):
    index = tmp_path / "docs.leann"
    monkeypatch.setattr(api, "compute_embeddings", _embed)
    builder = _builder("hnsw")
    for text in ("alpine lakes freeze", "coral reefs grow"):
        builder.add_text(text)
    builder.build_index(str(index))

    updater = _builder("hnsw")
    updater.add_text("desert dunes shift")
    updater.update_index(str(index))

    assert _keyword_hits(index, "dunes") == ["desert dunes shift"]


def test_keyword_search_follows_an_ivf_remove_and_add(tmp_path, monkeypatch):
    index = tmp_path / "docs.leann"
    monkeypatch.setattr(api, "compute_embeddings", _embed)
    builder = _builder("ivf")
    for text in ("alpine lakes freeze", "coral reefs grow", "desert dunes shift"):
        builder.add_text(text)
    builder.build_index(str(index))
    monkeypatch.setattr(leann_backend_ivf, "add_vectors", lambda *args: None)
    monkeypatch.setattr(leann_backend_ivf, "remove_ids", lambda path, ids: len(ids))

    # Passage "0" is removed, and the appended passage takes the freed ID.
    updater = _builder("ivf")
    updater.add_text("kelp forests sway")
    updater.update_index(str(index), remove_passage_ids=["0"])

    assert _keyword_hits(index, "alpine") == []
    assert _keyword_hits(index, "kelp") == ["kelp forests sway"]
