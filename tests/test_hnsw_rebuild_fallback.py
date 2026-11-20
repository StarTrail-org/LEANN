"""Regression test for #386: HNSW full-rebuild fallback wiped the corpus.

When HNSW can't apply an incremental update (any modified/removed file), `leann
build` falls back to a full rebuild. The fallback must reload the *entire*
corpus via `load_documents(docs_paths, ...)` — not reuse the partial
`all_texts` from the incremental path, which only contains the changed files.
Reusing the partial list silently drops every untouched file from the index.
"""

import asyncio
import json
import pickle
from pathlib import Path

from leann.cli import LeannCLI


def test_hnsw_modify_triggers_full_rebuild_that_reloads_untouched_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    keep_file = docs_root / "keep.md"
    keep_file.write_text("keep me around", encoding="utf-8")
    change_file = docs_root / "change.md"
    change_file.write_text("original content", encoding="utf-8")

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    meta_path = index_dir / "documents.leann.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "backend_name": "hnsw",
                "embedding_model": "facebook/contriever",
                "embedding_mode": "sentence-transformers",
                "backend_kwargs": {"is_compact": False, "is_recompute": True},
            }
        ),
        encoding="utf-8",
    )
    passages_file = index_dir / "documents.leann.passages.jsonl"
    passages_file.write_text(
        json.dumps({"id": "keep", "text": "keep me around", "metadata": {}}) + "\n",
        encoding="utf-8",
    )
    with open(index_dir / "documents.leann.passages.idx", "wb") as f:
        pickle.dump({"keep": 0}, f)

    # Simulate the modification that forces HNSW off the add-only path.
    change_file.write_text("modified content", encoding="utf-8")

    class FakeSynchronizer:
        def detect_changes(self):
            return set(), set(), {str(change_file.resolve())}

        def create_snapshot(self):
            pass

    load_documents_calls: list[list[str]] = []

    def fake_load_documents(paths, _file_types, include_hidden=False, args=None):
        load_documents_calls.append(list(paths))
        if list(paths) == [str(change_file.resolve())]:
            # Delta-only load taken on the incremental path before the
            # fallback decision is made.
            return [
                {"text": "modified content", "metadata": {"file_path": str(change_file.resolve())}}
            ]
        # Full-corpus reload, expected once HNSW falls back to a full rebuild.
        return [
            {"text": "keep me around", "metadata": {"file_path": str(keep_file.resolve())}},
            {"text": "modified content", "metadata": {"file_path": str(change_file.resolve())}},
        ]

    added_texts: list[str] = []

    class RecordingBuilder:
        def __init__(self, **_kwargs):
            pass

        def add_text(self, text, metadata=None):
            added_texts.append(text)

        def build_index(self, index_path):
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cli, "_build_synchronizers", lambda *_a, **_k: [FakeSynchronizer()])
    monkeypatch.setattr(cli, "load_documents", fake_load_documents)
    monkeypatch.setattr(cli, "register_project_dir", lambda: None)
    monkeypatch.setattr("leann.cli.LeannBuilder", RecordingBuilder)

    args = cli.create_parser().parse_args(
        [
            "build",
            "sample",
            "--docs",
            str(docs_root),
            "--backend-name",
            "hnsw",
        ]
    )

    asyncio.run(cli.build_index(args))

    # The fallback must have reloaded the full corpus, not just the delta.
    assert [str(docs_root)] in load_documents_calls
    assert "keep me around" in added_texts, (
        "full-rebuild fallback dropped an untouched file from the corpus (issue #386)"
    )
    assert "modified content" in added_texts


def test_hnsw_remove_only_triggers_full_rebuild(tmp_path, monkeypatch):
    """A deleted file leaves nothing to load from the changed paths.

    That must not stop the full rebuild, or the deleted file stays searchable.
    """
    monkeypatch.chdir(tmp_path)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    keep_file = docs_root / "keep.md"
    keep_file.write_text("keep me around", encoding="utf-8")
    gone_path = str((docs_root / "gone.md").resolve())  # indexed earlier, now deleted

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    (index_dir / "documents.leann.meta.json").write_text(
        json.dumps(
            {
                "backend_name": "hnsw",
                "embedding_model": "facebook/contriever",
                "embedding_mode": "sentence-transformers",
                "backend_kwargs": {"is_compact": False, "is_recompute": True},
            }
        ),
        encoding="utf-8",
    )

    class FakeSynchronizer:
        def detect_changes(self):
            return set(), {gone_path}, set()

        def create_snapshot(self):
            pass

    def fake_load_documents(paths, _file_types, include_hidden=False, args=None):
        if list(paths) == [str(docs_root)]:
            return [{"text": "keep me around", "metadata": {"file_path": str(keep_file)}}]
        return []  # nothing left to load from the deleted path

    added_texts: list[str] = []

    class RecordingBuilder:
        def __init__(self, **_kwargs):
            pass

        def add_text(self, text, metadata=None):
            added_texts.append(text)

        def build_index(self, index_path):
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cli, "_build_synchronizers", lambda *_a, **_k: [FakeSynchronizer()])
    monkeypatch.setattr(cli, "load_documents", fake_load_documents)
    monkeypatch.setattr(cli, "register_project_dir", lambda: None)
    monkeypatch.setattr("leann.cli.LeannBuilder", RecordingBuilder)

    args = cli.create_parser().parse_args(
        ["build", "sample", "--docs", str(docs_root), "--backend-name", "hnsw"]
    )
    asyncio.run(cli.build_index(args))

    assert added_texts == ["keep me around"]
