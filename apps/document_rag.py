"""
Document RAG example using the unified interface.
Supports PDF, TXT, MD, and other document formats.
"""

import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import importlib.util

from base_rag_example import BaseRAGExample
from chunking import create_text_chunks
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader

# MinerU powers OCR for scanned PDFs. Probe for it rather than importing it:
# the real import happens lazily in extract_pdf_with_ocr_fallback, so a run
# that never touches a scanned PDF does not pay for it.
OCR_AVAILABLE = importlib.util.find_spec("mineru") is not None


def extract_pdf_with_ocr_fallback(pdf_path: str, use_ocr: bool = False) -> str:
    """
    Extract text from PDF with OCR fallback.
    Used as a custom file extractor for SimpleDirectoryReader.

    Args:
        pdf_path: Path to PDF file
        use_ocr: Whether to force MinerU OCR instead of trying text extraction

    Returns:
        Extracted text string

    Raises:
        ImportError: If use_ocr is set and MinerU is not installed
    """
    if use_ocr:
        if not OCR_AVAILABLE:
            raise ImportError('OCR requires MinerU. Install it with: pip install "mineru>=4.0,<5"')
        try:
            from mineru import parse

            return parse(pdf_path, ocr_mode="ocr").markdown()
        except Exception as e:
            print(f"  OCR failed for {pdf_path}: {e}")
            return ""

    # Try PyMuPDF first
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        if text and len(text.strip()) > 100:
            return text
    except Exception:
        pass

    # Try pdfplumber
    try:
        import pdfplumber

        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

        if text and len(text.strip()) > 100:
            return text
    except Exception:
        pass

    return ""  # Return empty if all fail


class OcrPDFReader(BaseReader):
    def __init__(self, *, use_ocr: bool = False, auto_detect_scanned: bool = False):
        self.use_ocr = use_ocr
        self.auto_detect_scanned = auto_detect_scanned

    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        file_path = str(file)
        used_ocr = self.use_ocr
        text = extract_pdf_with_ocr_fallback(file_path, use_ocr=used_ocr)

        if not used_ocr and self.auto_detect_scanned:
            if len(text.strip()) < 100:
                print(f"Detected scanned PDF: {Path(file_path).name}")
                text = extract_pdf_with_ocr_fallback(file_path, use_ocr=True)
                used_ocr = True

        if used_ocr and text:
            print(f"OCR complete: {Path(file_path).name}")
        if not text.strip():
            return []

        metadata = {"file_name": Path(file_path).name}
        if extra_info:
            metadata.update(extra_info)
        return [Document(text=text, metadata=metadata)]


class DocumentRAG(BaseRAGExample):
    """RAG example for document processing (PDF, TXT, MD, etc.)."""

    def __init__(self):
        super().__init__(
            name="Document",
            description="Process and query documents (PDF, TXT, MD, etc.) with LEANN",
            default_index_name="test_doc_files",
        )

    def _add_specific_arguments(self, parser):
        """Add document-specific arguments."""
        doc_group = parser.add_argument_group("Document Parameters")
        doc_group.add_argument(
            "--data-dir",
            type=str,
            default="data",
            help="Directory containing documents to index (default: data)",
        )
        doc_group.add_argument(
            "--file-types",
            nargs="+",
            default=None,
            help="Filter by file types (e.g., .pdf .txt .md). If not specified, all supported types are processed",
        )
        doc_group.add_argument(
            "--chunk-size", type=int, default=256, help="Text chunk size (default: 256)"
        )
        doc_group.add_argument(
            "--chunk-overlap", type=int, default=128, help="Text chunk overlap (default: 128)"
        )
        doc_group.add_argument(
            "--enable-code-chunking",
            action="store_true",
            help="Enable AST-aware chunking for code files in the data directory",
        )

        # OCR parameters
        ocr_group = parser.add_argument_group("OCR Parameters (for scanned PDFs)")
        ocr_group.add_argument(
            "--use-ocr",
            action="store_true",
            help="Force OCR processing for all PDFs (even if they contain text)",
        )
        ocr_group.add_argument(
            "--auto-detect-scanned",
            action="store_true",
            default=True,
            help="Automatically detect and OCR scanned PDFs (default: True)",
        )
        ocr_group.add_argument(
            "--no-auto-detect-scanned",
            dest="auto_detect_scanned",
            action="store_false",
            help="Disable automatic detection of scanned PDFs",
        )

    async def load_data(self, args) -> list[dict[str, Any]]:
        """Load documents and convert to text chunks."""
        print(f"Loading documents from: {args.data_dir}")
        if args.file_types:
            print(f"Filtering by file types: {args.file_types}")
        else:
            print("Processing all supported file types")

        # Check if data directory exists
        data_path = Path(args.data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory not found: {args.data_dir}")

        use_ocr_for_all = args.use_ocr
        auto_detect_scanned = args.auto_detect_scanned and OCR_AVAILABLE

        # Load documents with custom PDF extractor
        reader_kwargs = {
            "recursive": True,
            "encoding": "utf-8",
        }
        if args.file_types:
            reader_kwargs["required_exts"] = args.file_types

        # Add custom PDF extractor if we need OCR
        if use_ocr_for_all or auto_detect_scanned:
            reader_kwargs["file_extractor"] = {
                ".pdf": OcrPDFReader(
                    use_ocr=use_ocr_for_all,
                    auto_detect_scanned=auto_detect_scanned,
                )
            }

        documents = SimpleDirectoryReader(args.data_dir, **reader_kwargs).load_data(
            show_progress=True
        )

        if not documents:
            print(f"No documents found in {args.data_dir} with extensions {args.file_types}")
            return []

        print(f"Loaded {len(documents)} documents")

        # Determine chunking strategy
        use_ast = args.enable_code_chunking or getattr(args, "use_ast_chunking", False)

        if use_ast:
            print("Using AST-aware chunking for code files")

        # Convert to text chunks with optional AST support
        all_texts = create_text_chunks(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            use_ast_chunking=use_ast,
            ast_chunk_size=getattr(args, "ast_chunk_size", 512),
            ast_chunk_overlap=getattr(args, "ast_chunk_overlap", 64),
            code_file_extensions=getattr(args, "code_file_extensions", None),
            ast_fallback_traditional=getattr(args, "ast_fallback_traditional", True),
        )

        # Apply max_items limit if specified
        if args.max_items > 0 and len(all_texts) > args.max_items:
            print(f"Limiting to {args.max_items} chunks (from {len(all_texts)})")
            all_texts = all_texts[: args.max_items]

        return all_texts


if __name__ == "__main__":
    import asyncio

    # Example queries for document RAG
    print("\nDocument RAG Example")
    print("=" * 50)
    print("\nExample queries you can try:")
    print("- 'What are the main techniques LEANN uses?'")
    print("- 'What is the technique DLPM?'")
    print("- 'Who does Elizabeth Bennet marry?'")
    print("- 'What challenges did Huawei face while developing the Pangu model?'")
    print("\nNEW: Code-aware chunking available!")
    print("- Use --enable-code-chunking to enable AST-aware chunking for code files")
    print("- Supports Python, Java, C#, TypeScript files")
    print("- Better semantic understanding of code structure")
    # Keep this file ASCII-only: tests/test_ci_minimal.py runs it as a subprocess,
    # and Windows stdout defaults to cp1252, which cannot encode emoji. This banner
    # prints before argparse, so one unencodable character breaks even --help.
    if OCR_AVAILABLE:
        print("\nOCR Support: Scanned PDF processing available!")
        print("- Use --use-ocr to force OCR for all PDFs")
        print("- Use --auto-detect-scanned (default) to automatically detect scanned PDFs")
    else:
        print("\nOCR Support: Install mineru for scanned PDF processing:")
        print('  pip install "mineru>=4.0,<5"')
    print("\nOr run without --query for interactive mode\n")

    rag = DocumentRAG()
    asyncio.run(rag.run())
