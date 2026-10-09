"""Appending to an existing index must not be blocked by the IDs its own builder chose.

``add_text()`` without an ID stamps ``str(len(self.chunks))`` -- this builder's insertion position
-- and ``update_index()`` then compared that stamp against the *target* index, so the first
appended chunk was always rejected as a duplicate of passage ``0``.  Auto-assigned IDs have to be
resolved against the index being updated; the caller's example for that flow
(``examples/dynamic_update_no_recompute.py``) has to hand every chunk a globally unique ``id`` to
get the same result.

The IVF cases drive ``add_vectors()``/``remove_ids()`` with a stand-in because those two calls go
straight to a FAISS ``IndexIVF``, and the FAISS bindings an installed HNSW wheel brings along do
not expose the NumPy-friendly ``add_with_ids()``/``remove_ids()`` overloads -- the append then
dies in the binding instead of in the code under test.  The passage-ID decision this change fixes
happens before any backend is called, and the last test keeps a real end-to-end append on the
non-compact HNSW path, where that decision is reached through the same code.
"""

import hashlib
import json
import pickle
import platform
import sys
from unittest.mock import patch

import leann.api as api
import leann_backend_ivf
import numpy as np
import pytest
from leann.api import LeannBuilder

VECTORS = {
    text: np.random.default_rng(ord(text)).standard_normal(8).astype(np.float32)
    for text in "abcdefgh"
}


def _embed(texts, *args, **kwargs):
    return np.stack([VECTORS[text] for text in texts])


def _fake_ivf_updates(monkeypatch):
    """Record what ``update_index()`` asks the IVF backend to add and remove."""
    calls: dict[str, list] = {"add": [], "remove": []}

    def add_vectors(index_path, embeddings, passage_ids):
        calls["add"].append(list(passage_ids))

    def remove_ids(index_path, passage_ids):
        calls["remove"].append(list(passage_ids))
        return len(passage_ids)

    monkeypatch.setattr(leann_backend_ivf, "add_vectors", add_vectors)
    monkeypatch.setattr(leann_backend_ivf, "remove_ids", remove_ids)
    return calls


def _builder(backend_name="ivf", **kwargs):
    if backend_name == "ivf":
        return LeannBuilder(backend_name="ivf", dimensions=8, nlist=2, **kwargs)
    return LeannBuilder(backend_name=backend_name, dimensions=8, **kwargs)


def _build(index, texts, backend_name="ivf", **kwargs):
    with patch.object(api, "compute_embeddings", side_effect=_embed):
        builder = _builder(backend_name, **kwargs)
        for text in texts:
            builder.add_text(text)
        builder.build_index(str(index))


def _append(index, texts, remove_passage_ids=None, backend_name="ivf", **kwargs):
    with patch.object(api, "compute_embeddings", side_effect=_embed):
        builder = _builder(backend_name, **kwargs)
        for text in texts:
            builder.add_text(text)
        builder.update_index(str(index), remove_passage_ids=remove_passage_ids)


def _text_of(index, passage_id: str) -> str:
    """Resolve a passage ID the way the searcher does: offset map, then one JSONL line."""
    with open(f"{index}.passages.idx", "rb") as f:
        offsets = pickle.load(f)
    with open(f"{index}.passages.jsonl", "rb") as f:
        f.seek(offsets[passage_id])
        return json.loads(f.readline().decode("utf-8"))["text"]


def _offset_ids(index) -> list[str]:
    with open(f"{index}.passages.idx", "rb") as f:
        return sorted(pickle.load(f))


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def test_append_without_ids_adds_to_the_existing_index(tmp_path, monkeypatch):
    index = tmp_path / "docs.leann"
    _build(index, "abc")
    calls = _fake_ivf_updates(monkeypatch)

    _append(index, "de")

    assert calls["add"] == [["3", "4"]]
    assert _text_of(index, "0") == "a"
    assert _text_of(index, "2") == "c"
    assert _text_of(index, "3") == "d"
    assert _text_of(index, "4") == "e"


def test_append_after_a_removal_reuses_the_freed_id_but_skips_ids_still_there(
    tmp_path, monkeypatch
):
    """Deleting a passage makes the index shorter, so an ID derived from its size is already used."""
    index = tmp_path / "docs.leann"
    _build(index, "abc")
    calls = _fake_ivf_updates(monkeypatch)

    _append(index, "de", remove_passage_ids=["0"])

    assert calls["remove"] == [["0"]]
    assert calls["add"] == [["0", "3"]]
    assert _offset_ids(index) == ["0", "1", "2", "3"]
    assert _text_of(index, "0") == "d"
    assert _text_of(index, "1") == "b"
    assert _text_of(index, "2") == "c"
    assert _text_of(index, "3") == "e"


def test_a_caller_chosen_duplicate_id_is_still_rejected(tmp_path, monkeypatch):
    index = tmp_path / "docs.leann"
    _build(index, "abc")
    _fake_ivf_updates(monkeypatch)

    with patch.object(api, "compute_embeddings", side_effect=_embed):
        builder = _builder()
        builder.add_text("d", metadata={"id": "1"})
        with pytest.raises(ValueError, match="already exists"):
            builder.update_index(str(index))


def test_a_content_hash_id_is_still_the_id_of_that_content(tmp_path, monkeypatch):
    """Content-hash IDs are derived from the text, not from a position, so they stay authoritative."""
    index = tmp_path / "docs.leann"
    scheme = {"passage_id_scheme": "content-hash"}
    _build(index, "abc", **scheme)
    calls = _fake_ivf_updates(monkeypatch)

    _append(index, "d", **scheme)

    assert calls["add"] == [[_hash("d")]]
    assert _text_of(index, _hash("d")) == "d"
    assert _text_of(index, _hash("a")) == "a"


@pytest.mark.skipif(
    sys.platform == "darwin" and platform.machine() == "x86_64",
    reason="On Intel macOS runners the native ``index.add()`` call that ``update_index()`` makes "
    "at ``api.py:1194`` aborts inside the FAISS bindings, taking the whole pytest session with "
    "it; the passage-ID decision under test happens before that call.",
)
def test_a_non_compact_hnsw_index_appends_without_ids_too(tmp_path):
    """The rejection happens in ``update_index`` before any backend runs, so HNSW hits it as well."""
    index = tmp_path / "docs.leann"
    _build(index, "abc", backend_name="hnsw", is_compact=False)

    _append(index, "de", backend_name="hnsw", is_compact=False)

    assert _text_of(index, "3") == "d"
    assert _text_of(index, "4") == "e"
