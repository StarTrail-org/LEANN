import json
from types import SimpleNamespace

import pytest
from leann.api import LeannSearcher


@pytest.fixture
def searcher(tmp_path):
    shard = tmp_path / "passages.jsonl"
    passages = [
        {"id": "regex", "text": "release v1x2"},
        {"id": "metadata", "text": "Release schedule", "metadata": {"filename": "release v1.2"}},
        {"id": "once", "text": "Release V1.2"},
        {"id": "twice", "text": "release v1.2 replaces RELEASE V1.2"},
        {"id": "escaped", "text": 'flag "--café"'},
    ]
    shard.write_text("".join(json.dumps(passage) + "\n" for passage in passages), encoding="utf-8")
    searcher = LeannSearcher.__new__(LeannSearcher)
    searcher.passage_manager = SimpleNamespace(passage_files={"test": str(shard)})
    return searcher


def test_grep_search_matches_only_literal_passage_text(searcher):
    results = searcher.search("release v1.2", use_grep=True)

    assert [(result.id, result.score) for result in results] == [("twice", 2.0), ("once", 1.0)]


def test_grep_search_matches_decoded_json_text(searcher):
    results = searcher.search('"--café"', use_grep=True)

    assert [(result.id, result.score) for result in results] == [("escaped", 1.0)]
