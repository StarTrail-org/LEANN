import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from leann.embedding_compute import compute_embeddings
from leann.searcher_base import BaseSearcher

MODEL = "gemini-embedding-001"


class StubSearcher(BaseSearcher):
    def search(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture
def embed_content(monkeypatch):
    client = Mock()
    client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.5, 0.25])]
    )
    google = ModuleType("google")
    genai = ModuleType("google.genai")
    genai.Client = Mock(return_value=client)
    genai.types = SimpleNamespace(EmbedContentConfig=SimpleNamespace)
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("leann.embedding_compute._model_cache", {})
    return client.models.embed_content


@pytest.mark.parametrize("is_build", [False, True])
def test_gemini_documents_use_document_task(embed_content, is_build):
    compute_embeddings(["A document"], MODEL, mode="gemini", is_build=is_build)

    assert embed_content.call_args.kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_gemini_search_uses_query_task(embed_content, tmp_path):
    index_path = tmp_path / "documents.leann"
    (tmp_path / "documents.leann.meta.json").write_text(
        json.dumps({"embedding_mode": "gemini", "embedding_model": MODEL, "dimensions": 2}),
        encoding="utf-8",
    )
    searcher = StubSearcher(str(index_path), "unused")

    searcher.compute_query_embedding("Find documents", use_server_if_available=False)

    assert embed_content.call_args.kwargs["config"].task_type == "RETRIEVAL_QUERY"
