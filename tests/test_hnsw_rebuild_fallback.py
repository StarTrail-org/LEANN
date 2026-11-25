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
import sys
from pathlib import Path

import pytest
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


def _write_existing_index_artifacts(index_dir: Path) -> None:
    """Create the meta/passages/idx files that make an index look live."""
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
    (index_dir / "documents.leann.passages.jsonl").write_text(
        json.dumps({"id": "gone", "text": "deleted passage", "metadata": {}}) + "\n",
        encoding="utf-8",
    )
    with open(index_dir / "documents.leann.passages.idx", "wb") as f:
        pickle.dump({"gone": 0}, f)


def test_hnsw_delete_all_files_removes_the_index(tmp_path, monkeypatch, capsys):
    """When every tracked file was deleted, the stale index must be dropped.

    Otherwise the full-rebuild lane printed "No documents found", left the old
    artifacts in place, and `leann search` kept returning the deleted passages.
    The recorded scope in sync_roots.json matches this build's flags, so the
    deletion is one the index actually observed.
    """
    monkeypatch.chdir(tmp_path)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    gone_path = str((docs_root / "gone.md").resolve())  # indexed earlier, deleted since

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    _write_existing_index_artifacts(index_dir)

    commits: list[bool] = []

    class EmptyCorpusSynchronizer:
        # A previous snapshot was loaded, so this build compares against it and can
        # observe that every tracked file is gone.
        tree = object()

        def detect_changes(self):
            return set(), {gone_path}, set()

        def tracked_paths(self):
            return []  # the synchronizer now tracks zero files

        def commit(self):
            commits.append(True)

        def create_snapshot(self):
            pass

    def fake_load_documents(paths, _file_types, include_hidden=False, args=None):
        return []  # nothing left to load from the deleted corpus

    monkeypatch.setattr(cli, "_build_synchronizers", lambda *_a, **_k: [EmptyCorpusSynchronizer()])
    monkeypatch.setattr(cli, "load_documents", fake_load_documents)
    monkeypatch.setattr(cli, "register_project_dir", lambda: None)

    args = cli.create_parser().parse_args(
        ["build", "sample", "--docs", str(docs_root), "--backend-name", "hnsw"]
    )
    # Record the scope the index was built from (default file types) the way a real
    # first build writes it, so the guard reads a real sync_roots.json.
    cli._write_sync_config_for_docs(index_dir, [str(docs_root)], args, None)
    asyncio.run(cli.build_index(args))

    assert not (index_dir / "documents.leann.meta.json").exists()
    assert not (index_dir / "documents.leann.passages.jsonl").exists()
    assert not (index_dir / "documents.leann.passages.idx").exists()
    assert commits == [True], "the synchronizers must record the now-empty corpus"
    out = capsys.readouterr().out
    assert "Removed index 'sample'" in out
    assert "`leann remove sample`" in out


def test_unloadable_corpus_keeps_existing_index(tmp_path, monkeypatch, capsys):
    """Files still exist but yield no text: the index is still valid and must survive."""
    monkeypatch.chdir(tmp_path)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    keep_file = docs_root / "keep.md"
    keep_file.write_text("still here", encoding="utf-8")
    gone_path = str((docs_root / "gone.md").resolve())  # indexed earlier, deleted since

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    _write_existing_index_artifacts(index_dir)

    class PartiallyTrackedSynchronizer:
        # A previous snapshot was loaded, so the index survives only because a tracked
        # file still exists (tracked_paths is non-empty), not for lack of one.
        tree = object()

        def detect_changes(self):
            return set(), {gone_path}, set()

        def tracked_paths(self):
            return [str(keep_file.resolve())]  # a tracked file still exists

    def fake_load_documents(paths, _file_types, include_hidden=False, args=None):
        return []  # the remaining file loads to nothing (filtered out / unreadable)

    monkeypatch.setattr(
        cli, "_build_synchronizers", lambda *_a, **_k: [PartiallyTrackedSynchronizer()]
    )
    monkeypatch.setattr(cli, "load_documents", fake_load_documents)
    monkeypatch.setattr(cli, "register_project_dir", lambda: None)

    args = cli.create_parser().parse_args(
        ["build", "sample", "--docs", str(docs_root), "--backend-name", "hnsw"]
    )
    asyncio.run(cli.build_index(args))

    assert (index_dir / "documents.leann.meta.json").exists()
    assert (index_dir / "documents.leann.passages.jsonl").exists()
    assert (index_dir / "documents.leann.passages.idx").exists()
    assert "No documents found" in capsys.readouterr().out


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="chmod does not remove read permission on Windows",
)
def test_unreadable_file_keeps_existing_index(tmp_path, monkeypatch, capsys):
    """An intact file that cannot be read is still tracked, never "deleted".

    generate_file_hashes() skips files it cannot read (it logs and continues), so
    an empty hash dict proves nothing about presence: an intact-but-unreadable
    corpus used to look exactly like a deleted one. The guard asks tracked_paths()
    instead, which applies the same membership rules as the snapshot without
    reading file contents.
    """
    monkeypatch.chdir(tmp_path)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    note_file = docs_root / "note.md"
    note_file.write_text("The chough nests on sea cliffs.\n", encoding="utf-8")

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    _write_existing_index_artifacts(index_dir)

    args = cli.create_parser().parse_args(
        ["build", "sample", "--docs", str(docs_root), "--backend-name", "hnsw"]
    )

    # Stand in for the first build: the real synchronizer records the readable
    # corpus in its snapshot, and the sync config records the corpus definition.
    synchronizers = cli._build_synchronizers([str(docs_root)], index_dir)
    assert len(synchronizers) == 1
    synchronizers[0].create_snapshot()
    cli._write_sync_config_for_docs(index_dir, [str(docs_root)], args, None)

    def fake_load_documents(paths, _file_types, include_hidden=False, args=None):
        return []  # no text can be loaded from a file that cannot be read

    monkeypatch.setattr(cli, "load_documents", fake_load_documents)
    monkeypatch.setattr(cli, "register_project_dir", lambda: None)

    try:
        note_file.chmod(0o000)
        asyncio.run(cli.build_index(args))
    finally:
        note_file.chmod(0o644)  # let the tmp dir be cleaned up

    assert (index_dir / "documents.leann.meta.json").exists()
    assert (index_dir / "documents.leann.passages.jsonl").exists()
    assert (index_dir / "documents.leann.passages.idx").exists()
    out = capsys.readouterr().out
    assert "No documents found" in out
    assert "Removed index" not in out


def test_missing_docs_path_keeps_existing_index(tmp_path, monkeypatch, capsys):
    """A --docs path that does not exist must never delete a healthy index.

    Such a build yields no synchronizer at all (the real _build_synchronizers
    resolves the path to neither a directory nor a file, and a directory that
    vanishes before construction only prints a warning). An empty synchronizer
    list must not read as "every tracked file was deleted".
    """
    monkeypatch.chdir(tmp_path)
    missing_docs = tmp_path / "nope"  # never created: a mistyped path

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    _write_existing_index_artifacts(index_dir)

    def fake_load_documents(paths, _file_types, include_hidden=False, args=None):
        return []  # nothing loads from a path that does not exist

    monkeypatch.setattr(cli, "load_documents", fake_load_documents)
    monkeypatch.setattr(cli, "register_project_dir", lambda: None)

    args = cli.create_parser().parse_args(
        ["build", "sample", "--docs", str(missing_docs), "--backend-name", "hnsw"]
    )
    asyncio.run(cli.build_index(args))

    assert (index_dir / "documents.leann.meta.json").exists()
    assert (index_dir / "documents.leann.passages.jsonl").exists()
    assert (index_dir / "documents.leann.passages.idx").exists()
    out = capsys.readouterr().out
    assert "No documents found" in out
    assert "Removed index" not in out


def test_new_empty_root_keeps_existing_index(tmp_path, monkeypatch, capsys):
    """A synchronizer with no snapshot that tracks zero files must not drop the index.

    Absence of a snapshot (tree is None) means this build has no previous state to
    compare against: an empty tracked set is not evidence that the indexed corpus
    was deleted, so the index must survive.
    """
    monkeypatch.chdir(tmp_path)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    gone_path = str((docs_root / "gone.md").resolve())  # indexed earlier, deleted since

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    _write_existing_index_artifacts(index_dir)

    commits: list[bool] = []

    class SnapshotlessSynchronizer:
        tree = None  # brand-new root: load_snapshot() found nothing to load

        def detect_changes(self):
            # Reports the same removal an all-deleted corpus would, so the flow
            # reaches the empty guard instead of exiting at "Index up to date.".
            return set(), {gone_path}, set()

        def tracked_paths(self):
            return []  # tracks zero files, like an empty directory

        def commit(self):
            commits.append(True)

    def fake_load_documents(paths, _file_types, include_hidden=False, args=None):
        return []

    monkeypatch.setattr(cli, "_build_synchronizers", lambda *_a, **_k: [SnapshotlessSynchronizer()])
    monkeypatch.setattr(cli, "load_documents", fake_load_documents)
    monkeypatch.setattr(cli, "register_project_dir", lambda: None)

    args = cli.create_parser().parse_args(
        ["build", "sample", "--docs", str(docs_root), "--backend-name", "hnsw"]
    )
    asyncio.run(cli.build_index(args))

    assert (index_dir / "documents.leann.meta.json").exists()
    assert (index_dir / "documents.leann.passages.jsonl").exists()
    assert (index_dir / "documents.leann.passages.idx").exists()
    assert commits == [], "nothing may be committed when the corpus was never observed"
    out = capsys.readouterr().out
    assert "No documents found" in out
    assert "Removed index" not in out


def test_changed_file_types_keeps_existing_index(tmp_path, monkeypatch, capsys):
    """A --file-types change that matches nothing must not drop the index.

    The new filter empties the synchronizer's tracked set while every indexed
    file is still on disk, so the deletion the guard infers from an empty set
    only counts when this build runs with the scope recorded in sync_roots.json.
    """
    monkeypatch.chdir(tmp_path)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "note.md").write_text("still here", encoding="utf-8")
    gone_path = str((docs_root / "gone.md").resolve())  # indexed earlier, deleted since

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    _write_existing_index_artifacts(index_dir)

    commits: list[bool] = []

    class FilterMatchedNothingSynchronizer:
        # Same root as the recorded scope, so the previous snapshot loads — but the
        # build now filters for .rst, which matches nothing in this .md corpus.
        tree = object()

        def detect_changes(self):
            return set(), {gone_path}, set()

        def tracked_paths(self):
            return []  # zero files match the new filter

        def commit(self):
            commits.append(True)

        def create_snapshot(self):
            pass

    def fake_load_documents(paths, _file_types, include_hidden=False, args=None):
        return []  # nothing to load under the new filter

    monkeypatch.setattr(
        cli, "_build_synchronizers", lambda *_a, **_k: [FilterMatchedNothingSynchronizer()]
    )
    monkeypatch.setattr(cli, "load_documents", fake_load_documents)
    monkeypatch.setattr(cli, "register_project_dir", lambda: None)

    # The index was built with the default file types; seed the real sync_roots.json
    # that first build wrote, exactly as _write_sync_config_for_docs records it.
    built_args = cli.create_parser().parse_args(
        ["build", "sample", "--docs", str(docs_root), "--backend-name", "hnsw"]
    )
    cli._write_sync_config_for_docs(index_dir, [str(docs_root)], built_args, None)

    args = cli.create_parser().parse_args(
        [
            "build",
            "sample",
            "--docs",
            str(docs_root),
            "--backend-name",
            "hnsw",
            "--file-types",
            ".rst",
        ]
    )
    asyncio.run(cli.build_index(args))

    assert (index_dir / "documents.leann.meta.json").exists()
    assert (index_dir / "documents.leann.passages.jsonl").exists()
    assert (index_dir / "documents.leann.passages.idx").exists()
    assert commits == [], "nothing may be committed when the index is kept"
    out = capsys.readouterr().out
    assert "No documents found" in out
    assert "Removed index" not in out


def test_include_hidden_toggle_keeps_existing_index(tmp_path, monkeypatch, capsys):
    """A --include-hidden change that matches nothing must not drop the index.

    Twin of the file-types case: the index tracked a hidden-only corpus (built with
    --include-hidden), and rebuilding without the flag empties the tracked set while
    the file is still on disk, so the recorded scope must be consulted first.
    """
    monkeypatch.chdir(tmp_path)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / ".secret.md").write_text("still here", encoding="utf-8")
    gone_path = str((docs_root / ".secret.md").resolve())  # indexed earlier, untracked now

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    _write_existing_index_artifacts(index_dir)

    commits: list[bool] = []

    class HiddenOnlyCorpusSynchronizer:
        # The previous snapshot was built with --include-hidden and tracked the hidden
        # file; this build excludes hidden files, so nothing matches anymore.
        tree = object()

        def detect_changes(self):
            return set(), {gone_path}, set()

        def tracked_paths(self):
            return []  # hidden files are excluded now

        def commit(self):
            commits.append(True)

        def create_snapshot(self):
            pass

    def fake_load_documents(paths, _file_types, include_hidden=False, args=None):
        return []  # nothing to load without hidden files

    monkeypatch.setattr(
        cli, "_build_synchronizers", lambda *_a, **_k: [HiddenOnlyCorpusSynchronizer()]
    )
    monkeypatch.setattr(cli, "load_documents", fake_load_documents)
    monkeypatch.setattr(cli, "register_project_dir", lambda: None)

    # Seed the real sync_roots.json the --include-hidden build wrote.
    built_args = cli.create_parser().parse_args(
        ["build", "sample", "--docs", str(docs_root), "--backend-name", "hnsw", "--include-hidden"]
    )
    cli._write_sync_config_for_docs(index_dir, [str(docs_root)], built_args, None)

    args = cli.create_parser().parse_args(
        ["build", "sample", "--docs", str(docs_root), "--backend-name", "hnsw"]
    )
    asyncio.run(cli.build_index(args))

    assert (index_dir / "documents.leann.meta.json").exists()
    assert (index_dir / "documents.leann.passages.jsonl").exists()
    assert (index_dir / "documents.leann.passages.idx").exists()
    assert commits == [], "nothing may be committed when the index is kept"
    out = capsys.readouterr().out
    assert "No documents found" in out
    assert "Removed index" not in out
