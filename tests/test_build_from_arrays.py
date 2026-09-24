"""
Tests for LeannBuilder.build_index_from_arrays and its integration with
build_index_from_embeddings (pickle-based path).
"""

import json
import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest


def test_build_from_arrays_custom_ids_are_searchable(tmp_path, monkeypatch):
    """Search resolves supplied IDs to the text added before the array build."""
    from leann.api import BACKEND_REGISTRY, LeannBuilder, LeannSearcher

    backend = Mock()
    monkeypatch.setitem(BACKEND_REGISTRY, "array-test", backend)
    index_path = str(tmp_path / "custom-ids.leann")
    ids = ["doc-a", "doc-b"]
    embeddings = np.array([[1, 0], [0, 1]], dtype=np.float32)

    builder = LeannBuilder(backend_name="array-test", dimensions=2)
    builder.add_text("Alpha document", metadata={"id": "original-a", "source": "alpha"})
    builder.add_text("Beta document", metadata={"source": "beta"})
    builder.build_index_from_arrays(index_path, ids, embeddings)

    assert backend.builder.return_value.build.call_args.args[1] == ids
    with open(f"{index_path}.passages.jsonl", encoding="utf-8") as f:
        assert [json.loads(line)["id"] for line in f] == ids
    with open(f"{index_path}.passages.idx", "rb") as f:
        assert set(pickle.load(f)) == set(ids)

    backend.searcher.return_value.compute_query_embedding.return_value = embeddings[0]
    backend.searcher.return_value.search.return_value = {
        "labels": [ids],
        "distances": [[0.9, 0.8]],
    }
    with LeannSearcher(index_path, enable_warmup=False, recompute_embeddings=False) as searcher:
        results = searcher.search("document", top_k=2)

    assert [(result.id, result.text, result.metadata["source"]) for result in results] == [
        ("doc-a", "Alpha document", "alpha"),
        ("doc-b", "Beta document", "beta"),
    ]
    assert results[0].metadata["id"] == "original-a"


def test_build_from_arrays_rejects_colliding_ids(tmp_path, monkeypatch):
    """Numeric and text IDs that become identical strings cannot share a passage."""
    from leann.api import BACKEND_REGISTRY, LeannBuilder

    monkeypatch.setitem(BACKEND_REGISTRY, "array-test", Mock())
    builder = LeannBuilder(backend_name="array-test", dimensions=2)
    with pytest.raises(ValueError, match="IDs must be unique"):
        builder.build_index_from_arrays(
            str(tmp_path / "colliding-ids.leann"), [1, "1"], np.zeros((2, 2), dtype=np.float32)
        )
    assert not (tmp_path / "colliding-ids.leann.passages.jsonl").exists()


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Skip model tests in CI to avoid MPS memory issues"
)
def test_build_from_arrays_basic():
    """Generate real embeddings for 5 texts, build via build_index_from_arrays, verify searchable."""
    from leann.api import LeannBuilder, LeannSearcher, compute_embeddings

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a high-level programming language",
        "Neural networks are inspired by the human brain",
        "Natural language processing enables computers to understand text",
    ]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        index_path = str(Path(temp_dir) / "test_arrays.hnsw")

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
        )

        embeddings = compute_embeddings(
            texts,
            model_name="facebook/contriever",
            mode="sentence-transformers",
            use_server=False,
            is_build=True,
        )

        for text in texts:
            builder.add_text(text)

        ids = list(range(len(texts)))
        builder.build_index_from_arrays(index_path, ids, embeddings)

        with LeannSearcher(index_path) as searcher:
            results = searcher.search("artificial intelligence machine learning", top_k=3)
            assert len(results) > 0
            assert any("intelligence" in r.text or "learning" in r.text for r in results)


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Skip model tests in CI to avoid MPS memory issues"
)
def test_build_from_arrays_matches_pickle_path():
    """Build same data via both methods, verify both produce searchable indexes."""
    from leann.api import LeannBuilder, LeannSearcher, compute_embeddings

    texts = [
        "The sun rises in the east",
        "Water flows downhill due to gravity",
        "Birds migrate south in winter",
        "Cats are independent animals",
        "Mathematics is the language of the universe",
    ]

    embeddings = compute_embeddings(
        texts,
        model_name="facebook/contriever",
        mode="sentence-transformers",
        use_server=False,
        is_build=True,
    )
    ids = list(range(len(texts)))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        # Build via arrays method
        arrays_index = str(Path(temp_dir) / "arrays_index.hnsw")
        builder_arrays = LeannBuilder(
            backend_name="hnsw",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
        )
        for text in texts:
            builder_arrays.add_text(text)
        builder_arrays.build_index_from_arrays(arrays_index, ids, embeddings)

        # Build via pickle method
        pickle_index = str(Path(temp_dir) / "pickle_index.hnsw")
        pickle_path = str(Path(temp_dir) / "embeddings.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump((ids, embeddings), f)

        builder_pickle = LeannBuilder(
            backend_name="hnsw",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
        )
        for text in texts:
            builder_pickle.add_text(text)
        builder_pickle.build_index_from_embeddings(pickle_index, pickle_path)

        query = "birds animals nature"

        with LeannSearcher(arrays_index) as searcher:
            arrays_results = searcher.search(query, top_k=3)
            assert len(arrays_results) > 0

        with LeannSearcher(pickle_index) as searcher:
            pickle_results = searcher.search(query, top_k=3)
            assert len(pickle_results) > 0

        # Both should return results (texts may differ slightly due to HNSW non-determinism,
        # but both indexes should be functional)
        assert len(arrays_results) == len(pickle_results)


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Skip model tests in CI to avoid MPS memory issues"
)
def test_build_from_arrays_with_text_chunks():
    """Call add_text first, then build_index_from_arrays; verify passages contain actual text."""
    from leann.api import LeannBuilder, LeannSearcher, compute_embeddings

    texts = [
        "Elephants are the largest land animals",
        "The Amazon rainforest is the world's largest tropical rainforest",
        "Quantum computing uses quantum mechanical phenomena",
    ]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        index_path = str(Path(temp_dir) / "with_chunks.hnsw")

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
        )

        for text in texts:
            builder.add_text(text)

        embeddings = compute_embeddings(
            texts,
            model_name="facebook/contriever",
            mode="sentence-transformers",
            use_server=False,
            is_build=True,
        )
        ids = list(range(len(texts)))
        builder.build_index_from_arrays(index_path, ids, embeddings)

        # Check that the passages JSONL contains real text, not placeholders
        passages_file = Path(index_path).parent / f"{Path(index_path).name}.passages.jsonl"
        assert passages_file.exists()
        import json

        passage_texts = []
        with open(passages_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    passage_texts.append(json.loads(line)["text"])

        assert len(passage_texts) == len(texts)
        # Actual texts, not placeholders like "Document 0"
        for actual_text in texts:
            assert actual_text in passage_texts

        with LeannSearcher(index_path) as searcher:
            results = searcher.search("large animals nature", top_k=2)
            assert len(results) > 0
            assert not any(r.text.startswith("Document ") for r in results)


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Skip model tests in CI to avoid MPS memory issues"
)
def test_build_from_arrays_dimension_mismatch():
    """Set builder dimensions to 100, pass 768-dim embeddings, expect ValueError."""
    from leann.api import LeannBuilder

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        index_path = str(Path(temp_dir) / "dim_mismatch.hnsw")

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
            dimensions=100,
        )

        ids = [0, 1, 2]
        # 768-dim embeddings (contriever default) when builder expects 100
        embeddings = np.random.rand(3, 768).astype(np.float32)

        with pytest.raises(ValueError, match="[Dd]imension"):
            builder.build_index_from_arrays(index_path, ids, embeddings)


def test_build_from_arrays_count_mismatch():
    """Pass 3 ids but 5 embeddings, expect ValueError."""
    from leann.api import LeannBuilder

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        index_path = str(Path(temp_dir) / "count_mismatch.hnsw")

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
        )

        ids = [0, 1, 2]  # 3 ids
        embeddings = np.random.rand(5, 768).astype(np.float32)  # 5 embeddings

        with pytest.raises(ValueError, match="[Mm]ismatch"):
            builder.build_index_from_arrays(index_path, ids, embeddings)


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Skip model tests in CI to avoid MPS memory issues"
)
def test_build_from_arrays_without_chunks_creates_placeholders():
    """Call build_index_from_arrays without prior add_text; verify placeholder entries created."""
    from leann.api import LeannBuilder, LeannSearcher, compute_embeddings

    texts_for_embedding = [
        "Placeholder document alpha",
        "Placeholder document beta",
        "Placeholder document gamma",
    ]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        index_path = str(Path(temp_dir) / "no_chunks.hnsw")

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
        )

        embeddings = compute_embeddings(
            texts_for_embedding,
            model_name="facebook/contriever",
            mode="sentence-transformers",
            use_server=False,
            is_build=True,
        )
        ids = ["doc-a", "doc-b", "doc-c"]

        # No add_text calls — builder has no chunks
        builder.build_index_from_arrays(index_path, ids, embeddings)

        # Check passages file has placeholder entries
        import json

        passages_file = Path(index_path).parent / f"{Path(index_path).name}.passages.jsonl"
        assert passages_file.exists()
        passage_texts = []
        with open(passages_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    passage_texts.append(json.loads(line)["text"])

        assert len(passage_texts) == len(ids)
        # All entries should be placeholders ("Document <id>")
        for text in passage_texts:
            assert text.startswith("Document ")

        # Index should still be searchable
        with LeannSearcher(index_path) as searcher:
            results = searcher.search("document placeholder", top_k=2)
            assert len(results) > 0


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Skip model tests in CI to avoid MPS memory issues"
)
def test_pickle_method_delegates_to_arrays():
    """Verify build_index_from_embeddings still works after refactor (regression test)."""
    from leann.api import LeannBuilder, LeannSearcher, compute_embeddings

    texts = [
        "Regression test document one about science",
        "Regression test document two about history",
        "Regression test document three about art",
    ]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        index_path = str(Path(temp_dir) / "regression.hnsw")
        pickle_path = str(Path(temp_dir) / "embeddings.pkl")

        embeddings = compute_embeddings(
            texts,
            model_name="facebook/contriever",
            mode="sentence-transformers",
            use_server=False,
            is_build=True,
        )
        ids = list(range(len(texts)))

        with open(pickle_path, "wb") as f:
            pickle.dump((ids, embeddings), f)

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
        )
        for text in texts:
            builder.add_text(text)

        # This should still work exactly as before
        builder.build_index_from_embeddings(index_path, pickle_path)

        with LeannSearcher(index_path) as searcher:
            results = searcher.search("science history", top_k=2)
            assert len(results) > 0
            assert isinstance(results[0].text, str)
