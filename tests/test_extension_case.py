"""File extensions match regardless of case when `leann build` scans a directory.

`leann watch` and incremental builds already treated NOTES.TXT like notes.txt, but the
directory scan behind a full build compared suffixes case-sensitively and silently
skipped such files, so a `--force` rebuild dropped what an incremental one had added.
"""

from pathlib import Path

import pytest
from leann.cli import LeannCLI


def _load(docs_dir, custom_file_types=None):
    chunks = LeannCLI().load_documents([str(docs_dir)], custom_file_types=custom_file_types)
    return sorted({Path(chunk["metadata"]["source"]).name for chunk in chunks})


def _write(docs_dir, *names):
    for name in names:
        (docs_dir / name).write_text(f"contents of {name}", encoding="utf-8")


def test_default_scan_loads_uppercase_and_mixed_case_extensions(tmp_path):
    _write(tmp_path, "lower.txt", "NOTES.TXT", "Readme.Md", "SCRIPT.PY")

    assert _load(tmp_path) == ["NOTES.TXT", "Readme.Md", "SCRIPT.PY", "lower.txt"]


def test_file_types_option_matches_extensions_case_insensitively(tmp_path):
    _write(tmp_path, "a.txt", "B.TXT", "c.md")

    assert _load(tmp_path, ".txt") == ["B.TXT", "a.txt"]
    assert _load(tmp_path, ".TXT") == ["B.TXT", "a.txt"]
    assert _load(tmp_path, "TXT, .MD") == ["B.TXT", "a.txt", "c.md"]


def test_uppercase_pdf_extension_is_read(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    for name in ("lower.pdf", "REPORT.PDF"):
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), f"text of {name}")
        doc.save(tmp_path / name)

    assert _load(tmp_path) == ["REPORT.PDF", "lower.pdf"]
    assert _load(tmp_path, "PDF") == ["REPORT.PDF", "lower.pdf"]
