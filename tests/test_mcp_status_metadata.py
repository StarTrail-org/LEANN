"""`leann_status` must describe the index, not the last passage it counted."""

import json
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest

module_path = Path(__file__).resolve().parents[1] / "packages/leann-core/src/leann/mcp.py"
mcp = ModuleType("leann_mcp_status_metadata_test")
SourceFileLoader(mcp.__name__, str(module_path)).exec_module(mcp)

INDEX_META = {
    "backend_name": "ivf",
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_mode": "local",
    "dimensions": 384,
}
PASSAGES = [
    {"text": "first passage", "metadata": {"file_path": "docs/a.md"}},
    {"text": "second passage", "metadata": {"source": "docs/b.md"}},
]


def _make_index(project: Path, passages: list[dict], extra_raw_lines: tuple[str, ...] = ()) -> Path:
    index_dir = project / ".leann" / "indexes" / "docs"
    index_dir.mkdir(parents=True)
    (index_dir / "documents.leann.meta.json").write_text(json.dumps(INDEX_META), encoding="utf-8")
    with open(index_dir / "documents.leann.passages.jsonl", "w", encoding="utf-8") as f:
        for passage in passages:
            f.write(json.dumps(passage) + "\n")
        for raw in extra_raw_lines:
            f.write(raw + "\n")
    return index_dir


@pytest.fixture
def status_text(tmp_path, monkeypatch):
    def _status(passages, extra_raw_lines=()):
        project = tmp_path / "project"
        _make_index(project, passages, extra_raw_lines)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(mcp, "_base_dir", str(project))
        result = mcp.handle_status(1, {"index_name": "docs"})
        return result["result"]["content"][0]["text"]

    return _status


def test_index_metadata_survives_the_passage_count(status_text):
    text = status_text(PASSAGES)

    assert "Backend: ivf" in text
    assert "Embedding: all-MiniLM-L6-v2 (local)" in text
    assert "Dimensions: 384" in text


def test_passage_counting_still_reports_chunks_and_files(status_text):
    text = status_text(PASSAGES)

    assert "Chunks: 2" in text
    assert "Files indexed: 2" in text


def test_a_malformed_final_passage_line_does_not_change_the_report(status_text):
    text = status_text(PASSAGES, ('{"text": "cut off mid-record',))

    assert "Backend: ivf" in text
    assert "Embedding: all-MiniLM-L6-v2 (local)" in text
    assert "Dimensions: 384" in text
    assert "Files indexed: 2" in text
