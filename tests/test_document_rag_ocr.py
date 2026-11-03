import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


class _LlamaDocument:
    def __init__(self, text, metadata=None):
        self.text = text
        self.metadata = metadata or {}


def _load_document_rag(monkeypatch):
    base_rag = types.SimpleNamespace(BaseRAGExample=type("BaseRAGExample", (), {}))
    chunking = types.SimpleNamespace(create_text_chunks=lambda *args, **kwargs: [])
    llama_index_core = types.SimpleNamespace(
        Document=_LlamaDocument,
        SimpleDirectoryReader=type("SimpleDirectoryReader", (), {}),
    )
    llama_index = types.SimpleNamespace(core=llama_index_core)
    readers_base = types.SimpleNamespace(BaseReader=type("BaseReader", (), {}))

    monkeypatch.setitem(sys.modules, "base_rag_example", base_rag)
    monkeypatch.setitem(sys.modules, "chunking", chunking)
    monkeypatch.setitem(sys.modules, "llama_index", llama_index)
    monkeypatch.setitem(sys.modules, "llama_index.core", llama_index_core)
    monkeypatch.setitem(sys.modules, "llama_index.core.readers", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "llama_index.core.readers.base", readers_base)

    path = REPO_ROOT / "apps" / "document_rag.py"
    spec = importlib.util.spec_from_file_location("document_rag_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class _Page:
    def __init__(self, text):
        self._text = text

    def get_text(self):
        return self._text


class _Document:
    def __init__(self, text):
        self._page = _Page(text)

    def __iter__(self):
        return iter([self._page])

    def close(self):
        pass


def test_use_ocr_skips_text_extractors(monkeypatch):
    module = _load_document_rag(monkeypatch)
    monkeypatch.setattr(module, "OCR_AVAILABLE", True)

    fitz_calls = []
    fitz = types.SimpleNamespace()

    def open_pdf(path):
        fitz_calls.append(path)
        return _Document("text layer " * 20)

    fitz.open = open_pdf
    monkeypatch.setitem(sys.modules, "fitz", fitz)

    parse_calls = []
    mineru = types.SimpleNamespace()

    class ParseResult:
        def markdown(self):
            return "ocr markdown"

    def parse(path, **kwargs):
        parse_calls.append((path, kwargs))
        return ParseResult()

    mineru.parse = parse
    monkeypatch.setitem(sys.modules, "mineru", mineru)

    assert module.extract_pdf_with_ocr_fallback("paper.pdf", use_ocr=True) == "ocr markdown"
    assert fitz_calls == []
    assert parse_calls == [("paper.pdf", {"ocr_mode": "ocr", "tier": "standard", "page_range": ""})]


def test_ocr_output_drops_embedded_images_and_keeps_text(monkeypatch):
    module = _load_document_rag(monkeypatch)
    monkeypatch.setattr(module, "OCR_AVAILABLE", True)
    markdown = (
        "Figure 1: index layout.\n"
        "![](data:image/jpeg;base64,/9j/4AAQ+/=)\n"
        '<td><img src="data:image/png;base64,iVBORw0KGgo="></td><td>cell</td>\n'
        "See ![diagram](https://example.com/a.png)."
    )
    result = types.SimpleNamespace(markdown=lambda: markdown)
    mineru = types.SimpleNamespace(parse=lambda path, **kwargs: result)
    monkeypatch.setitem(sys.modules, "mineru", mineru)

    assert module.extract_pdf_with_ocr_fallback("scan.pdf", use_ocr=True) == (
        "Figure 1: index layout.\n"
        "\n"
        "<td></td><td>cell</td>\n"
        "See ![diagram](https://example.com/a.png)."
    )


def test_text_extraction_remains_the_default(monkeypatch):
    module = _load_document_rag(monkeypatch)
    monkeypatch.setattr(module, "OCR_AVAILABLE", True)

    text = "text layer " * 20
    fitz = types.SimpleNamespace(open=lambda path: _Document(text))
    monkeypatch.setitem(sys.modules, "fitz", fitz)

    mineru = types.SimpleNamespace()

    def unexpected_parse(*args, **kwargs):
        raise AssertionError("MinerU should not run when use_ocr is false")

    mineru.parse = unexpected_parse
    monkeypatch.setitem(sys.modules, "mineru", mineru)

    assert module.extract_pdf_with_ocr_fallback("paper.pdf") == text


def test_ocr_reader_satisfies_simple_directory_reader_contract(monkeypatch):
    module = _load_document_rag(monkeypatch)
    monkeypatch.setattr(module, "OCR_AVAILABLE", True)

    fitz_calls = []
    fitz = types.SimpleNamespace()

    def open_pdf(path):
        fitz_calls.append(path)
        return _Document("text layer " * 20)

    fitz.open = open_pdf
    monkeypatch.setitem(sys.modules, "fitz", fitz)

    mineru = types.SimpleNamespace()

    class ParseResult:
        def markdown(self):
            return "ocr markdown"

    mineru.parse = lambda path, **kwargs: ParseResult()
    monkeypatch.setitem(sys.modules, "mineru", mineru)

    reader = module.OcrPDFReader(use_ocr=True)
    documents = reader.load_data(Path("paper.pdf"), extra_info={"source": "test"})

    assert len(documents) == 1
    assert documents[0].text == "ocr markdown"
    assert documents[0].metadata == {"file_name": "paper.pdf", "source": "test"}
    assert fitz_calls == []


def test_forced_ocr_without_mineru_raises_import_error(monkeypatch, capsys):
    module = _load_document_rag(monkeypatch)
    monkeypatch.setattr(module, "OCR_AVAILABLE", False)

    reader = module.OcrPDFReader(use_ocr=True)

    # SimpleDirectoryReader logs and skips a file on any other exception type,
    # so only ImportError reaches the user instead of "No documents found".
    with pytest.raises(ImportError, match=r"pip install \"mineru"):
        reader.load_data(Path("scan.pdf"))
    assert "downloads" not in capsys.readouterr().out


def _stub_extractor(module, monkeypatch, text_layer, ocr_text="ocr text"):
    """Replace the extractor and return the OCR calls it receives."""
    ocr_calls = []

    def extract(path, use_ocr=False, tier="standard", page_range=""):
        if not use_ocr:
            return text_layer
        ocr_calls.append({"tier": tier, "page_range": page_range})
        return ocr_text

    monkeypatch.setattr(module, "extract_pdf_with_ocr_fallback", extract)
    return ocr_calls


def test_reader_passes_tier_to_ocr(monkeypatch):
    module = _load_document_rag(monkeypatch)
    ocr_calls = _stub_extractor(module, monkeypatch, text_layer="")

    module.OcrPDFReader(use_ocr=True, tier="basic").load_data(Path("scan.pdf"))

    assert ocr_calls == [{"tier": "basic", "page_range": ""}]


def test_auto_detect_ocrs_only_scanned_pages_and_keeps_text_layer(monkeypatch):
    module = _load_document_rag(monkeypatch)
    text_layer = "text layer " * 20
    ocr_calls = _stub_extractor(module, monkeypatch, text_layer=text_layer)
    monkeypatch.setattr(module, "find_scanned_pages", lambda path: [2, 5])

    documents = module.OcrPDFReader(auto_detect_scanned=True).load_data(Path("mixed.pdf"))

    assert ocr_calls == [{"tier": "standard", "page_range": "2,5"}]
    assert documents[0].text == f"{text_layer}\n\nocr text"


def test_auto_detect_keeps_text_layer_when_page_ocr_fails(monkeypatch):
    module = _load_document_rag(monkeypatch)
    text_layer = "text layer " * 20
    _stub_extractor(module, monkeypatch, text_layer=text_layer, ocr_text="")
    monkeypatch.setattr(module, "find_scanned_pages", lambda path: [2])

    documents = module.OcrPDFReader(auto_detect_scanned=True).load_data(Path("mixed.pdf"))

    assert documents[0].text == text_layer


def test_auto_detect_sends_images_straight_to_ocr(monkeypatch):
    module = _load_document_rag(monkeypatch)
    ocr_calls = _stub_extractor(module, monkeypatch, text_layer="unused text layer " * 20)

    documents = module.OcrPDFReader(auto_detect_scanned=True).load_data(Path("photo.PNG"))

    assert ocr_calls == [{"tier": "standard", "page_range": ""}]
    assert documents[0].text == "ocr text"


def test_auto_detect_skips_ocr_for_text_only_pdf(monkeypatch):
    module = _load_document_rag(monkeypatch)
    ocr_calls = _stub_extractor(module, monkeypatch, text_layer="text layer " * 20)
    monkeypatch.setattr(module, "find_scanned_pages", lambda path: [])

    module.OcrPDFReader(auto_detect_scanned=True).load_data(Path("paper.pdf"))

    assert ocr_calls == []


def test_ocr_warns_about_model_download_once(monkeypatch, capsys):
    module = _load_document_rag(monkeypatch)
    monkeypatch.setattr(module, "OCR_AVAILABLE", True)
    _stub_extractor(module, monkeypatch, text_layer="")
    reader = module.OcrPDFReader(use_ocr=True)

    reader.load_data(Path("a.pdf"))
    reader.load_data(Path("b.pdf"))

    assert capsys.readouterr().out.count("downloads") == 1


def test_find_scanned_pages_flags_only_image_pages_without_text(monkeypatch, tmp_path):
    fitz = pytest.importorskip("fitz")
    module = _load_document_rag(monkeypatch)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50))
    pixmap.clear_with(200)

    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "A page with a normal text layer. " * 3)
    scanned = pdf.new_page()
    scanned.insert_image(scanned.rect, pixmap=pixmap)
    figure_only = pdf.new_page()
    figure_only.insert_image(fitz.Rect(0, 0, 50, 50), pixmap=pixmap)
    pdf.new_page()
    path = tmp_path / "mixed.pdf"
    pdf.save(path)
    pdf.close()

    assert module.find_scanned_pages(str(path)) == [2]


def test_find_scanned_pages_returns_nothing_for_unreadable_pdf(monkeypatch, tmp_path):
    pytest.importorskip("fitz")
    module = _load_document_rag(monkeypatch)

    assert module.find_scanned_pages(str(tmp_path / "missing.pdf")) == []


def test_file_extractor_maps_pdfs_and_images_when_ocr_is_on(monkeypatch):
    module = _load_document_rag(monkeypatch)

    forced = module.build_file_extractor(use_ocr=True, auto_detect_scanned=False, tier="basic")
    auto = module.build_file_extractor(use_ocr=False, auto_detect_scanned=True, tier="basic")

    assert set(forced) == set(auto) == {".pdf", ".png", ".jpg", ".jpeg"}
    assert forced[".png"].use_ocr is True
    assert forced[".png"].tier == "basic"
    assert auto[".png"].use_ocr is False
    assert module.build_file_extractor(False, False, "basic") == {}


def test_drop_empty_documents_reports_files_with_no_text(monkeypatch, capsys):
    module = _load_document_rag(monkeypatch)
    monkeypatch.setattr(module, "OCR_AVAILABLE", False)
    documents = [
        _LlamaDocument("", {"file_name": "scan.pdf"}),
        _LlamaDocument("  \n", {"file_name": "paper.pdf"}),
        _LlamaDocument("real text", {"file_name": "paper.pdf"}),
    ]

    kept = module.drop_empty_documents(documents)

    assert [document.text for document in kept] == ["real text"]
    out = capsys.readouterr().out
    assert "scan.pdf" in out
    assert "paper.pdf" not in out
    assert "mineru" in out


def test_drop_empty_documents_skips_the_ocr_hint_for_text_files(monkeypatch, capsys):
    module = _load_document_rag(monkeypatch)
    monkeypatch.setattr(module, "OCR_AVAILABLE", False)

    module.drop_empty_documents([_LlamaDocument("", {"file_name": "notes.txt"})])

    out = capsys.readouterr().out
    assert "notes.txt" in out
    assert "mineru" not in out
