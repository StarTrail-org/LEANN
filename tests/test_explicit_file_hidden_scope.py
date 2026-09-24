"""A file named explicitly on --docs is judged by its own name, not by its ancestors.

Pointing `--docs` at a hidden directory already indexes what is inside it (#52/#56).
Naming one file inside that directory used to behave differently in two ways: the
build skipped it whenever the path spelled out the hidden ancestor, and the sync
snapshot never tracked it at all, because it resolves every path to absolute first.
"""

from leann.cli import LeannCLI
from leann.sync import FileSynchronizer


def _hidden_dir_with_note(tmp_path):
    hidden = tmp_path / ".vault"
    hidden.mkdir()
    note = hidden / "notes.md"
    note.write_text("# hello", encoding="utf-8")
    return hidden, note


def _cli_load(path, include_hidden=False):
    return LeannCLI().load_documents([str(path.resolve())], include_hidden=include_hidden)


def test_explicit_file_under_hidden_dir_is_synced(tmp_path):
    _, note = _hidden_dir_with_note(tmp_path)

    fs = FileSynchronizer(
        explicit_files=[str(note.resolve())],
        include_extensions=[".md"],
        snapshot_path=str(tmp_path / "sync.pickle"),
        auto_load=False,
    )

    assert str(note.resolve()) in fs.generate_file_hashes()


def test_explicit_dotfile_still_needs_include_hidden(tmp_path):
    dotfile = tmp_path / ".secrets.md"
    dotfile.write_text("shh", encoding="utf-8")

    def sync(include_hidden):
        return FileSynchronizer(
            explicit_files=[str(dotfile.resolve())],
            include_extensions=[".md"],
            include_hidden=include_hidden,
            snapshot_path=str(tmp_path / f"sync_{include_hidden}.pickle"),
            auto_load=False,
        ).generate_file_hashes()

    assert sync(include_hidden=False) == {}
    assert str(dotfile.resolve()) in sync(include_hidden=True)


def test_build_loads_explicit_file_under_hidden_dir(tmp_path, monkeypatch):
    hidden, note = _hidden_dir_with_note(tmp_path)
    cli = LeannCLI()

    from_absolute = cli.load_documents([str(note.resolve())])

    monkeypatch.chdir(hidden)
    from_relative = cli.load_documents(["notes.md"])

    assert len(from_absolute) == len(from_relative) == 1


def test_build_skips_explicit_dotfile(tmp_path):
    dotfile = tmp_path / ".secrets.md"
    dotfile.write_text("shh", encoding="utf-8")

    assert _cli_load(dotfile) == []
    assert len(_cli_load(dotfile, include_hidden=True)) == 1
