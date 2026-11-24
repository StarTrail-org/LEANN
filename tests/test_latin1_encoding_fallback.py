"""Latin-1 / Windows-1252 text files keep their accented characters (issue #483).

`leann build` used to load .txt/.md/code files through llama-index's
SimpleDirectoryReader standard read, which decodes with errors="ignore":
bytes that are not valid UTF-8 were silently dropped ("crème brûlée"
became "crme brle"). Plain-text extensions are now read through
FallbackTextReader, which falls back to cp1252 with a one-line warning.

Offline: no network, no model downloads.
"""

from leann.cli import TEXT_FILE_EXTRACTOR
from leann.readers import FallbackTextReader
from llama_index.core import SimpleDirectoryReader

TEXT = "Recette: crème brûlée à la façon de Grand-mère.\n"


def _load(tmp_path, name, raw, capsys):
    path = tmp_path / name
    path.write_bytes(raw)
    docs = SimpleDirectoryReader(
        str(tmp_path),
        encoding="utf-8",
        required_exts=[path.suffix],
        file_extractor=TEXT_FILE_EXTRACTOR,
        filename_as_id=True,
    ).load_data()
    assert len(docs) == 1
    return docs[0], capsys.readouterr().out


def test_latin1_txt_preserves_accents_and_warns(tmp_path, capsys):
    doc, out = _load(tmp_path, "recette.txt", TEXT.encode("latin-1"), capsys)
    assert doc.text == TEXT
    assert "crème brûlée" in doc.text
    assert "not valid UTF-8" in out
    assert "recette.txt" in out
    assert doc.metadata.get("file_path", "").endswith("recette.txt")


def test_utf8_file_loads_byte_identically_without_warning(tmp_path, capsys):
    doc, out = _load(tmp_path, "recette.txt", TEXT.encode("utf-8"), capsys)
    assert doc.text == TEXT
    assert "not valid UTF-8" not in out


def test_latin1_markdown_preserves_accents(tmp_path, capsys):
    doc, out = _load(tmp_path, "recette.md", TEXT.encode("latin-1"), capsys)
    assert doc.text == TEXT
    assert "crème brûlée" in doc.text
    assert "recette.md" in out


def test_fallback_reader_never_raises_on_undecodable_bytes(tmp_path, capsys):
    # 0x81 is undefined in cp1252 and invalid on its own in UTF-8.
    path = tmp_path / "odd.bin.txt"
    path.write_bytes(b"a\x81b")
    docs = FallbackTextReader().load_data(path)
    assert len(docs) == 1
    assert docs[0].text == "a\ufffdb"
    assert "not valid UTF-8" in capsys.readouterr().out


def test_fallback_reader_decodes_cp1252_punctuation(tmp_path):
    path = tmp_path / "quotes.txt"
    path.write_bytes(b"\x93hi\x94")  # cp1252 smart quotes, invalid UTF-8
    docs = FallbackTextReader().load_data(path)
    assert docs[0].text == "\u201chi\u201d"


def test_text_file_extractor_covers_plain_text_extensions():
    for ext in (".txt", ".md", ".py", ".json", ".yaml", ".html", ".sh"):
        assert ext in TEXT_FILE_EXTRACTOR
        assert isinstance(TEXT_FILE_EXTRACTOR[ext], FallbackTextReader)
