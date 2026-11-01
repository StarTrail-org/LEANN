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
    assert parse_calls == [("paper.pdf", {"ocr_mode": "ocr"})]


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


def test_forced_ocr_without_mineru_raises_import_error(monkeypatch):
    module = _load_document_rag(monkeypatch)
    monkeypatch.setattr(module, "OCR_AVAILABLE", False)

    reader = module.OcrPDFReader(use_ocr=True)

    # SimpleDirectoryReader logs and skips a file on any other exception type,
    # so only ImportError reaches the user instead of "No documents found".
    with pytest.raises(ImportError, match=r"pip install \"mineru"):
        reader.load_data(Path("scan.pdf"))
