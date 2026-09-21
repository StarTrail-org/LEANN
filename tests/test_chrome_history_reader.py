import importlib.util
import os
import sqlite3
import sys
import types
from pathlib import Path

try:
    from llama_index.core import Document as _Document
except ModuleNotFoundError:
    llama_index = types.ModuleType("llama_index")
    llama_index_core = types.ModuleType("llama_index.core")
    llama_index_readers = types.ModuleType("llama_index.core.readers")
    llama_index_base = types.ModuleType("llama_index.core.readers.base")

    class _Document:
        def __init__(self, text, metadata):
            self.text = text
            self.metadata = metadata

    class _BaseReader:
        pass

    llama_index_core.Document = _Document
    llama_index_base.BaseReader = _BaseReader
    sys.modules["llama_index"] = llama_index
    sys.modules["llama_index.core"] = llama_index_core
    sys.modules["llama_index.core.readers"] = llama_index_readers
    sys.modules["llama_index.core.readers.base"] = llama_index_base

readers_path = Path(__file__).parents[1] / "packages/leann-core/src/leann/readers.py"
readers_spec = importlib.util.spec_from_file_location("leann_readers", readers_path)
assert readers_spec is not None and readers_spec.loader is not None
readers = importlib.util.module_from_spec(readers_spec)
readers_spec.loader.exec_module(readers)
ChromeHistoryReader = readers.ChromeHistoryReader


def test_chrome_history_reader_includes_rows_still_in_wal(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    history_path = profile_dir / "History"

    source = sqlite3.connect(history_path)
    assert source.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    source.execute(
        """
        CREATE TABLE urls (
            last_visit_time INTEGER,
            url TEXT,
            title TEXT,
            visit_count INTEGER,
            typed_count INTEGER,
            hidden INTEGER
        )
        """
    )
    source.commit()
    source.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    source.execute(
        "INSERT INTO urls VALUES (?, ?, ?, ?, ?, ?)",
        (13300000000000000, "https://example.com/recent", "Recent page", 1, 0, 0),
    )
    source.commit()

    isolated_copy = tmp_path / "main-database-only"
    legacy_temp_path = "/tmp/leann_history_index_copy"
    real_connect = sqlite3.connect
    real_copy2 = readers.shutil.copy2
    real_exists = os.path.exists

    def redirected_connect(database, *args, **kwargs):
        if os.fspath(database) == legacy_temp_path:
            database = isolated_copy
        return real_connect(database, *args, **kwargs)

    def redirected_copy(source_path, _destination, *args, **kwargs):
        return real_copy2(source_path, isolated_copy, *args, **kwargs)

    def redirected_exists(path):
        if os.fspath(path) == legacy_temp_path:
            return False
        return real_exists(path)

    monkeypatch.setattr(readers.sqlite3, "connect", redirected_connect)
    monkeypatch.setattr(readers.shutil, "copy2", redirected_copy)
    monkeypatch.setattr(readers.os.path, "exists", redirected_exists)

    documents = ChromeHistoryReader().load_data(str(profile_dir))
    source.close()

    assert [document.metadata["url"] for document in documents] == ["https://example.com/recent"]
