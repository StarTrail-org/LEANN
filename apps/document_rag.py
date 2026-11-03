"""
Document RAG example using the unified interface.
Supports PDF, TXT, MD, and other document formats.
"""

import re
import sys
from pathlib import Path
from typing import Any, Literal, get_args

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


OCR_INSTALL_HINT = 'pip install "mineru>=4.0,<5"'
OcrTier = Literal["flash", "basic", "standard", "advanced"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
OCR_EXTENSIONS = (".pdf", *IMAGE_EXTENSIONS)

# A page counts as scanned when it has almost no text and images cover most of it.
# The coverage check keeps blank pages and pages with one small figure out of OCR.
MAX_SCANNED_PAGE_TEXT_CHARS = 20
MIN_SCANNED_PAGE_IMAGE_COVERAGE = 0.5

# MinerU's Markdown embeds each figure as a base64 data URI, in Markdown or HTML form.
# Captions and recognized text are separate blocks, so removing the images loses no text.
EMBEDDED_IMAGE = re.compile(
    r"!\[[^\]]*\]\(data:[^)]*\)|<img\b[^>]*\bsrc=[\"']data:[^>]*>", re.IGNORECASE
)


def strip_embedded_images(markdown: str) -> str:
    """Remove base64 images, which would otherwise fill the index with noise."""
    return EMBEDDED_IMAGE.sub("", markdown)


def extract_pdf_with_ocr_fallback(
    pdf_path: str, use_ocr: bool = False, tier: OcrTier = "standard", page_range: str = ""
) -> str:
    """
    Extract text from PDF with OCR fallback.
    Used as a custom file extractor for SimpleDirectoryReader.

    Args:
        pdf_path: Path to PDF file
        use_ocr: Whether to force MinerU OCR instead of trying text extraction
        tier: MinerU quality tier for OCR. "flash" and "basic" are faster on CPU but less accurate
        page_range: Pages to OCR, such as "2,5". Blank means every page

    Returns:
        Extracted text string

    Raises:
        ImportError: If use_ocr is set and MinerU is not installed
    """
    if use_ocr:
        if not OCR_AVAILABLE:
            raise ImportError(f"OCR requires MinerU. Install it with: {OCR_INSTALL_HINT}")
        try:
            from mineru import parse

            result = parse(pdf_path, ocr_mode="ocr", tier=tier, page_range=page_range)
            return strip_embedded_images(result.markdown())
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


def find_scanned_pages(pdf_path: str) -> list[int]:
    """Return the 1-based numbers of pages that are images with no usable text layer."""
    try:
        import fitz  # PyMuPDF

        with fitz.open(pdf_path) as doc:
            scanned = []
            for number, page in enumerate(doc, start=1):
                if len(page.get_text().strip()) > MAX_SCANNED_PAGE_TEXT_CHARS:
                    continue
                image_area = sum(
                    abs(fitz.Rect(image["bbox"]) & page.rect) for image in page.get_image_info()
                )
                if image_area / abs(page.rect) >= MIN_SCANNED_PAGE_IMAGE_COVERAGE:
                    scanned.append(number)
            return scanned
    except Exception:
        return []


class OcrPDFReader(BaseReader):
    def __init__(
        self,
        *,
        use_ocr: bool = False,
        auto_detect_scanned: bool = False,
        tier: OcrTier = "standard",
    ):
        self.use_ocr = use_ocr
        self.auto_detect_scanned = auto_detect_scanned
        self.tier = tier
        self._download_notice_shown = False

    def _ocr(self, file_path: str, page_range: str = "") -> str:
        if OCR_AVAILABLE and not self._download_notice_shown:
            print(
                f"Running MinerU OCR (tier={self.tier}). "
                "The first run downloads its models, which take over 1 GB."
            )
            self._download_notice_shown = True
        text = extract_pdf_with_ocr_fallback(
            file_path, use_ocr=True, tier=self.tier, page_range=page_range
        )
        if text:
            print(f"OCR complete: {Path(file_path).name}")
        return text

    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        file_path = str(file)
        name = Path(file_path).name
        # An image has no text layer, so it always needs OCR.
        if self.use_ocr or Path(file_path).suffix.lower() in IMAGE_EXTENSIONS:
            text = self._ocr(file_path)
        else:
            text = extract_pdf_with_ocr_fallback(file_path)
            if self.auto_detect_scanned:
                if len(text.strip()) < 100:
                    print(f"Detected scanned PDF: {name}")
                    text = self._ocr(file_path)
                elif scanned_pages := find_scanned_pages(file_path):
                    print(f"Detected {len(scanned_pages)} scanned page(s) in {name}")
                    page_range = ",".join(str(page) for page in scanned_pages)
                    if page_text := self._ocr(file_path, page_range):
                        text = f"{text}\n\n{page_text}"

        if not text.strip():
            return []

        metadata = {"file_name": name}
        if extra_info:
            metadata.update(extra_info)
        return [Document(text=text, metadata=metadata)]


def build_file_extractor(
    use_ocr: bool, auto_detect_scanned: bool, tier: OcrTier
) -> dict[str, OcrPDFReader]:
    """Map PDFs and images to an OCR reader, or nothing when OCR is off."""
    if not use_ocr and not auto_detect_scanned:
        return {}
    reader = OcrPDFReader(use_ocr=use_ocr, auto_detect_scanned=auto_detect_scanned, tier=tier)
    return dict.fromkeys(OCR_EXTENSIONS, reader)


def drop_empty_documents(documents: list[Document]) -> list[Document]:
    """Remove documents with no text and name the files that produced nothing at all."""
    kept = [document for document in documents if document.text.strip()]
    kept_files = {document.metadata.get("file_name") for document in kept}
    all_files = {document.metadata.get("file_name") for document in documents}
    skipped = sorted(str(name) for name in all_files - kept_files)
    if skipped:
        print(f"Skipping {len(skipped)} file(s) with no text: {', '.join(skipped)}")
        if not OCR_AVAILABLE and any(
            Path(name).suffix.lower() in OCR_EXTENSIONS for name in skipped
        ):
            print(f"Scanned PDFs and images need OCR. Install MinerU with: {OCR_INSTALL_HINT}")
    return kept


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
            help="Force OCR for all PDFs, even those with a text layer",
        )
        ocr_group.add_argument(
            "--ocr-tier",
            choices=get_args(OcrTier),
            default="standard",
            help="MinerU quality tier, fastest first. Faster tiers are less accurate (default: standard)",
        )
        ocr_group.add_argument(
            "--auto-detect-scanned",
            action="store_true",
            default=True,
            help="OCR images, scanned PDFs, and scanned pages inside PDFs (default: True)",
        )
        ocr_group.add_argument(
            "--no-auto-detect-scanned",
            dest="auto_detect_scanned",
            action="store_false",
            help="Disable automatic OCR of images and scanned PDFs",
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

        # Add custom extractors if we need OCR
        file_extractor = build_file_extractor(use_ocr_for_all, auto_detect_scanned, args.ocr_tier)
        if file_extractor:
            reader_kwargs["file_extractor"] = file_extractor

        documents = drop_empty_documents(
            SimpleDirectoryReader(args.data_dir, **reader_kwargs).load_data(show_progress=True)
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
        print(f"  {OCR_INSTALL_HINT}")
    print("\nOr run without --query for interactive mode\n")

    rag = DocumentRAG()
    asyncio.run(rag.run())
