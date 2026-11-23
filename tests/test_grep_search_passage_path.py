"""Grep search must use the passage shards the index metadata declares.

``_grep_search`` used to look only for ``documents.leann.passages.jsonl``, so grep
search raised ``FileNotFoundError`` for every index not named ``documents``.
``PassageManager`` already resolves the real paths from the index's own
``passage_sources``; these tests pin that behaviour.
"""

import json
import pickle

import numpy as np
import pytest

pytest.importorskip("leann_backend_hnsw")

from leann.api import LeannBuilder, LeannSearcher


def _build_index(tmp_path, name, passages):
    """Build a real HNSW index from precomputed vectors (no model download)."""
    vectors = np.eye(len(passages), dtype=np.float32)
    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model="grep-test",
        embedding_mode="sentence-transformers",
        dimensions=vectors.shape[1],
    )
    for _, text in passages:
        builder.add_text(text)
    index_path = str(tmp_path / name)
    builder.build_index_from_arrays(index_path, [pid for pid, _ in passages], vectors)
    return index_path


def _open_searcher(index_path):
    return LeannSearcher(
        index_path,
        enable_warmup=False,
        recompute_embeddings=False,
        use_daemon=False,
    )


def test_grep_search_reads_index_own_passages(tmp_path):
    """An index named ``notes.leann`` finds its own shard instead of ``documents.*``."""
    index_path = _build_index(
        tmp_path,
        "notes.leann",
        [("note-a", "Boats moor in the harbor at dawn"), ("note-b", "The harbor bell rings")],
    )

    with _open_searcher(index_path) as searcher:
        results = searcher.search("harbor", use_grep=True)

    assert [(result.id, result.text) for result in results] == [
        ("note-a", "Boats moor in the harbor at dawn"),
        ("note-b", "The harbor bell rings"),
    ]


def test_grep_search_works_for_the_default_index_name(tmp_path):
    """``documents`` is the CLI default index name; its shard must still resolve.

    Regression cover for the common case, not red-on-main evidence: the previous
    hardcoded lookup already found this name.
    """
    index_path = _build_index(
        tmp_path, "documents.leann", [("doc-a", "The harbor is calm tonight")]
    )

    with _open_searcher(index_path) as searcher:
        results = searcher.search("harbor", use_grep=True)

    assert [(result.id, result.text) for result in results] == [
        ("doc-a", "The harbor is calm tonight")
    ]


def test_grep_search_aggregates_matches_from_every_shard(tmp_path):
    """Every shard declared in the metadata is searched, not just the first one."""
    index_path = _build_index(tmp_path, "notes.leann", [("note-a", "harbor light on the pier")])

    # A second shard, declared the same way the builder declares its own.
    shard = tmp_path / "archive.leann.passages.jsonl"
    offsets = {}
    with open(shard, "w", encoding="utf-8") as f:
        offsets["archive-a"] = f.tell()
        json.dump({"id": "archive-a", "text": "harbor fog at noon", "metadata": {}}, f)
        f.write("\n")
    with open(tmp_path / "archive.leann.passages.idx", "wb") as f:
        pickle.dump(offsets, f)

    meta_path = tmp_path / "notes.leann.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["passage_sources"].append(
        {
            "type": "jsonl",
            "path": shard.name,
            "index_path": "archive.leann.passages.idx",
        }
    )
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with _open_searcher(index_path) as searcher:
        results = searcher.search("harbor", use_grep=True)

    assert [(result.id, result.text) for result in results] == [
        ("note-a", "harbor light on the pier"),
        ("archive-a", "harbor fog at noon"),
    ]


def test_grep_search_raises_when_passage_jsonl_is_missing(tmp_path):
    """Error semantics are unchanged: no shard on disk still means FileNotFoundError."""
    index_path = _build_index(tmp_path, "notes.leann", [("note-a", "harbor at dawn")])
    (tmp_path / "notes.leann.passages.jsonl").unlink()

    with _open_searcher(index_path) as searcher:
        with pytest.raises(
            FileNotFoundError, match="No .jsonl passages file found for grep search"
        ):
            searcher.search("harbor", use_grep=True)
