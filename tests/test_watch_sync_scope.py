"""Regression tests for leann watch sync scope (#345)."""

from pathlib import Path

from leann.cli import LeannCLI
from leann.sync import FileSynchronizer


def test_resolve_sync_scope_keeps_loose_files_separate(tmp_path):
    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    readme = repo / "README.md"
    readme.write_text("# hello", encoding="utf-8")

    cli = LeannCLI()
    directories, files = cli._resolve_sync_scope([str(src), str(readme)])

    assert directories == [str(src.resolve())]
    assert files == [str(readme.resolve())]


def test_watch_scope_does_not_scan_sibling_media(tmp_path):
    repo = tmp_path / "repo"
    src = repo / "src"
    assets = repo / "assets"
    src.mkdir(parents=True)
    assets.mkdir()
    readme = repo / "README.md"
    readme.write_text("# hello", encoding="utf-8")
    (src / "main.py").write_text("print('ok')", encoding="utf-8")
    (assets / "icon.png").write_bytes(b"png")

    synchronizers = [
        FileSynchronizer(
            root_dir=str(src),
            include_extensions=[".py", ".md"],
            snapshot_path=str(tmp_path / "sync_src.pickle"),
            auto_load=False,
        ),
        FileSynchronizer(
            explicit_files=[str(readme.resolve())],
            include_extensions=[".py", ".md"],
            snapshot_path=str(tmp_path / "sync_readme.pickle"),
            auto_load=False,
        ),
    ]

    hashed_paths: set[str] = set()
    for fs in synchronizers:
        hashed_paths.update(fs.generate_file_hashes().keys())

    assert str((src / "main.py").resolve()) in hashed_paths
    assert str(readme.resolve()) in hashed_paths
    assert str((assets / "icon.png").resolve()) not in hashed_paths


def test_mixed_txt_and_bin_directory_skips_bin_without_crash(tmp_path):
    """Same dir with .txt and .bin: hash only text, ignore binary (review #377)."""
    docs = tmp_path / "docs"
    docs.mkdir()
    txt = docs / "notes.txt"
    bin_file = docs / "payload.bin"
    txt.write_text("hello", encoding="utf-8")
    bin_file.write_bytes(bytes(range(256)))

    fs = FileSynchronizer(
        root_dir=str(docs),
        include_extensions=[".txt"],
        snapshot_path=str(tmp_path / "sync.pickle"),
        auto_load=False,
    )

    hashes = fs.generate_file_hashes()
    assert set(hashes.keys()) == {str(txt.resolve())}
    assert str(bin_file.resolve()) not in hashes

    fs.create_snapshot()
    fs2 = FileSynchronizer(
        root_dir=str(docs),
        include_extensions=[".txt"],
        snapshot_path=str(tmp_path / "sync.pickle"),
    )
    added, removed, modified = fs2.detect_changes()
    assert not added and not removed and not modified


def _docs_with_gitignore(tmp_path):
    docs = tmp_path / "docs"
    (docs / "build").mkdir(parents=True)
    (docs / ".gitignore").write_text("secret.txt\nbuild/\n", encoding="utf-8")
    (docs / "notes.md").write_text("Project notes about glaciers.", encoding="utf-8")
    (docs / "secret.txt").write_text("API token list, do not index.", encoding="utf-8")
    (docs / "build" / "out.md").write_text("Generated build output.", encoding="utf-8")
    return docs


def test_build_sync_tracks_what_the_directory_scan_loads(tmp_path):
    """Files that .gitignore excludes from a full build must not enter the snapshot either.

    Otherwise a new or edited ignored file is reported as a change and loaded by name, and
    files named on --docs skip .gitignore, so the incremental update indexes it.
    """
    docs = _docs_with_gitignore(tmp_path)
    cli = LeannCLI()

    scanned = {chunk["metadata"]["file_path"] for chunk in cli.load_documents([str(docs)])}
    [fs] = cli._build_synchronizers([str(docs)], tmp_path / "index")

    assert set(fs.generate_file_hashes()) == {str(Path(p).resolve()) for p in scanned}
    assert set(fs.generate_file_hashes()) == {str((docs / "notes.md").resolve())}


def test_gitignore_edits_show_up_as_added_and_removed_files(tmp_path):
    docs = _docs_with_gitignore(tmp_path)
    (tmp_path / "index").mkdir()
    cli = LeannCLI()
    [fs] = cli._build_synchronizers([str(docs)], tmp_path / "index")
    fs.create_snapshot()

    (docs / ".gitignore").write_text("build/\nnotes.md\n", encoding="utf-8")
    [fs] = cli._build_synchronizers([str(docs)], tmp_path / "index")
    added, removed, modified = fs.detect_changes()

    assert added == [str((docs / "secret.txt").resolve())]
    assert removed == [str((docs / "notes.md").resolve())]
    assert modified == []
