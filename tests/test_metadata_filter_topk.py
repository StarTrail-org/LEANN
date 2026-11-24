"""Metadata filters must not shrink the result set below top_k when matches exist."""

from unittest.mock import Mock

import numpy as np
import pytest

N_DOCS = 20


@pytest.fixture
def searcher(tmp_path, monkeypatch):
    """A searcher over a mocked backend that ranks passage 0 first, then 1, 2, ..."""
    from leann.api import BACKEND_REGISTRY, LeannBuilder, LeannSearcher

    backend = Mock()
    monkeypatch.setitem(BACKEND_REGISTRY, "topk-test", backend)
    index_path = str(tmp_path / "topk.leann")
    ids = [f"doc-{i}" for i in range(N_DOCS)]
    embeddings = np.zeros((N_DOCS, 2), dtype=np.float32)

    builder = LeannBuilder(backend_name="topk-test", dimensions=2)
    for i in range(N_DOCS):
        # Only the lowest-ranked passages are "keep"; the top of the ranking is "skip".
        builder.add_text(
            f"passage {i}",
            metadata={
                "group": "keep" if i >= 15 else "skip",
                "band": "mid" if 3 <= i < 12 else "x",
            },
        )
    builder.build_index_from_arrays(index_path, ids, embeddings)

    def fake_search(query_embedding, top_k, **kwargs):
        top_k = min(top_k, N_DOCS)
        return {
            "labels": [ids[:top_k]],
            "distances": [[1.0 - i / N_DOCS for i in range(top_k)]],
        }

    backend.searcher.return_value.compute_query_embedding.return_value = embeddings[0]
    backend.searcher.return_value.search.side_effect = fake_search
    with LeannSearcher(index_path, enable_warmup=False, recompute_embeddings=False) as s:
        yield s


def test_filtered_search_fills_top_k_from_beyond_the_first_top_k(searcher):
    results = searcher.search("q", top_k=3, metadata_filters={"group": {"==": "keep"}})

    assert [r.id for r in results] == ["doc-15", "doc-16", "doc-17"]


def test_filtered_search_returns_all_matches_when_fewer_than_top_k(searcher):
    results = searcher.search("q", top_k=10, metadata_filters={"group": {"==": "keep"}})

    assert [r.id for r in results] == [f"doc-{i}" for i in range(15, 20)]


def test_filter_matching_nothing_returns_empty(searcher):
    results = searcher.search("q", top_k=3, metadata_filters={"group": {"==": "nope"}})

    assert results == []


def test_filter_that_matches_the_top_hits_does_not_widen_the_search(searcher):
    results = searcher.search("q", top_k=3, metadata_filters={"group": {"==": "skip"}})

    assert [r.id for r in results] == ["doc-0", "doc-1", "doc-2"]
    assert searcher.backend_impl.search.call_count == 1


def _candidate_sizes(searcher):
    return [call.args[1] for call in searcher.backend_impl.search.call_args_list]


def test_widening_keeps_the_requested_count_fixed(searcher):
    """Wanting 3 with matches at ranks 3..11 needs one widening (3 -> 12), not more."""
    results = searcher.search("q", top_k=3, metadata_filters={"band": {"==": "mid"}})

    assert [r.id for r in results] == ["doc-3", "doc-4", "doc-5"]
    assert _candidate_sizes(searcher) == [3, 12]


def test_widening_is_capped_at_the_passage_count(searcher):
    searcher.search("q", top_k=10, metadata_filters={"group": {"==": "keep"}})

    assert _candidate_sizes(searcher) == [10, N_DOCS]
