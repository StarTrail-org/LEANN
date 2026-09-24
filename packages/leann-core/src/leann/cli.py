import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import os
import pickle
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Union

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from tqdm import tqdm

from .api import Fts5BM25Index, LeannBuilder, LeannChat, LeannSearcher
from .embedding_server_manager import EmbeddingServerManager
from .interactive_utils import create_cli_session
from .registry import DEFAULT_INDEX_SCAN_DEPTH, iter_index_meta_files, register_project_directory
from .settings import (
    resolve_anthropic_base_url,
    resolve_atlascloud_api_key,
    resolve_atlascloud_base_url,
    resolve_litellm_api_key,
    resolve_litellm_base_url,
    resolve_minimax_api_key,
    resolve_minimax_base_url,
    resolve_ollama_host,
    resolve_openai_api_key,
    resolve_openai_base_url,
)
from .sync import DEFAULT_INDEX_EXTENSIONS, FileSynchronizer, parse_include_extensions


def _non_negative_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed_value


def _default_embedding_model() -> str:
    """Pick a sensible default embedding model based on platform.

    | Platform   | Default model                                  |
    |------------|------------------------------------------------|
    | NVIDIA GPU | BAAI/bge-base-en-v1.5                          |
    | macOS      | sentence-transformers/all-MiniLM-L6-v2         |
    | Other/CPU  | sentence-transformers/all-MiniLM-L6-v2         |
    """

    try:
        import torch

        if torch.cuda.is_available():
            return "BAAI/bge-base-en-v1.5"
    except ImportError:
        pass

    # macOS (MPS or CPU) and all other platforms: lightweight model
    return "sentence-transformers/all-MiniLM-L6-v2"


def _normalize_path(path: str) -> str:
    """Return absolute path string for consistent keys."""
    if not path:
        return path
    return str(Path(path).resolve())


def _cleanup_path(path: Path) -> None:
    if path.is_dir():
        import shutil

        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _existing_index_artifacts(index_dir: Path) -> bool:
    return (
        (index_dir / "documents.leann.meta.json").exists()
        and (index_dir / "documents.leann.passages.jsonl").exists()
        and (index_dir / "documents.leann.passages.idx").exists()
    )


def _publish_rebuilt_index(staging_dir: Path, index_dir: Path) -> None:
    """Publish a staged full rebuild, restoring the previous directory if publish fails."""
    backup_dir = index_dir.with_name(f".{index_dir.name}.backup-{uuid.uuid4().hex}")
    live_moved = False
    published = False
    try:
        if index_dir.exists():
            os.replace(index_dir, backup_dir)
            live_moved = True
        os.replace(staging_dir, index_dir)
        published = True
    except Exception:
        if published:
            _cleanup_path(index_dir)
        if live_moved and backup_dir.exists():
            os.replace(backup_dir, index_dir)
        raise
    else:
        if backup_dir.exists():
            _cleanup_path(backup_dir)


@contextlib.contextmanager
def suppress_cpp_output(suppress: bool = True):
    """Context manager to suppress C++ stdout/stderr output from FAISS/HNSW
    while preserving Python print() output.

    C++ native code writes directly to OS file descriptors (fd 1 / fd 2).
    Python print() goes through sys.stdout / sys.stderr, which are Python
    file objects.  We redirect the OS fds to /dev/null (silencing C++) but
    point sys.stdout / sys.stderr at copies of the *original* fds so that
    Python output still reaches the terminal.
    """
    if not suppress:
        yield
        return

    # 1. Duplicate the original OS file descriptors.
    #    May fail in no-console environments (e.g. pythonw, Windows GUI apps)
    #    where fd 1/2 are not valid — in that case, skip suppression.
    try:
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
    except OSError:
        yield
        return

    # 2. Build Python file objects that write to the saved (real) fds.
    #    closefd=False so closing these wrappers won't close the duped fds.
    py_stdout = io.TextIOWrapper(
        io.FileIO(saved_stdout_fd, mode="w", closefd=False), encoding=sys.stdout.encoding or "utf-8"
    )
    py_stderr = io.TextIOWrapper(
        io.FileIO(saved_stderr_fd, mode="w", closefd=False), encoding=sys.stderr.encoding or "utf-8"
    )

    old_sys_stdout = sys.stdout
    old_sys_stderr = sys.stderr

    try:
        # 3. Redirect OS-level fds to /dev/null → silences C++ output
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)

        # 4. Point Python's sys.stdout/stderr at the real terminal
        sys.stdout = py_stdout
        sys.stderr = py_stderr

        yield
    finally:
        # 5. Restore everything
        #    Flush wrappers first (they still need the saved fds to be open)
        py_stdout.flush()
        py_stderr.flush()

        sys.stdout = old_sys_stdout
        sys.stderr = old_sys_stderr

        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def extract_pdf_text_with_pymupdf(file_path: str) -> str | None:
    """Extract text from PDF using PyMuPDF for better quality."""
    try:
        if os.path.getsize(file_path) == 0:
            # Empty file: nothing to extract, skip instead of letting
            # fitz.EmptyFileError abort the whole build.
            return ""

        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        # Fallback to default reader
        return None
    except Exception:
        # Skip corrupted PDFs instead of aborting the whole build
        return ""


def extract_pdf_text_with_pdfplumber(file_path: str) -> str | None:
    """Extract text from PDF using pdfplumber for better quality."""
    try:
        if os.path.getsize(file_path) == 0:
            # Empty file: nothing to extract, skip instead of aborting the build.
            return ""

        import pdfplumber

        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except ImportError:
        # Fallback to default reader
        return None
    except Exception:
        # Skip corrupted PDFs instead of aborting the whole build
        return ""


class LeannCLI:
    def __init__(self):
        # Always use project-local .leann directory (like .git)
        self.indexes_dir = Path.cwd() / ".leann" / "indexes"
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

        # Default parser for documents
        self.node_parser = SentenceSplitter(
            chunk_size=256, chunk_overlap=128, separator=" ", paragraph_separator="\n\n"
        )

        # Code-optimized parser
        self.code_parser = SentenceSplitter(
            chunk_size=512,  # Larger chunks for code context
            chunk_overlap=50,  # Less overlap to preserve function boundaries
            separator="\n",  # Split by lines for code
            paragraph_separator="\n\n",  # Preserve logical code blocks
        )

    def get_index_path(self, index_name: str) -> str:
        index_dir = self.indexes_dir / index_name
        return str(index_dir / "documents.leann")

    def index_exists(self, index_name: str) -> bool:
        index_dir = self.indexes_dir / index_name
        meta_file = index_dir / "documents.leann.meta.json"
        return meta_file.exists()

    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="leann",
            description="The smallest vector index in the world. RAG Everything with LEANN!",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  leann build my-docs --docs ./documents                                  # Build index from directory
  leann build my-code --docs ./src ./tests ./config                      # Build index from multiple directories
  leann build my-files --docs ./file1.py ./file2.txt ./docs/             # Build index from files and directories
  leann build my-mixed --docs ./readme.md ./src/ ./config.json           # Build index from mixed files/dirs
  leann build my-ppts --docs ./ --file-types .pptx,.pdf                  # Index only PowerPoint and PDF files
  leann search my-docs "query"                                           # Search in my-docs index
  leann ask my-docs "question"                                           # Ask my-docs index
  leann react my-docs "complex question"                                 # Use ReAct agent for multiturn retrieval
  leann index-browser chrome                                             # Index Chrome browser history
  leann index-email                                                      # Index Apple Mail
  leann index-imessage                                                   # Index iMessage conversations
  leann index-chatgpt --export-path ~/chatgpt-export.zip                 # Index ChatGPT export
  leann list                                                             # List all stored indexes
  leann remove my-docs                                                   # Remove an index (local first, then global)
            """,
        )

        # Global verbosity options
        verbosity_group = parser.add_mutually_exclusive_group()
        verbosity_group.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Show detailed output including C++ backend logs from FAISS/HNSW",
        )
        verbosity_group.add_argument(
            "-q",
            "--quiet",
            action="store_true",
            help="Suppress all non-essential output (default behavior)",
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        def add_embedding_args(target_parser: argparse.ArgumentParser) -> None:
            """Add common embedding arguments to a parser."""
            target_parser.add_argument(
                "--embedding-model",
                type=str,
                default="facebook/contriever",
                help="Embedding model (default: facebook/contriever)",
            )
            target_parser.add_argument(
                "--embedding-mode",
                type=str,
                default="sentence-transformers",
                choices=["sentence-transformers", "openai", "mlx", "ollama"],
                help="Embedding backend mode (default: sentence-transformers)",
            )
            target_parser.add_argument(
                "--embedding-host",
                type=str,
                default=None,
                help="Override Ollama-compatible embedding host",
            )
            target_parser.add_argument(
                "--embedding-api-base",
                type=str,
                default=None,
                help="Base URL for OpenAI-compatible embedding services",
            )
            target_parser.add_argument(
                "--embedding-api-key",
                type=str,
                default=None,
                help="API key for embedding service (defaults to OPENAI_API_KEY)",
            )

        # Build command
        build_parser = subparsers.add_parser("build", help="Build document index")
        build_parser.add_argument(
            "index_name", nargs="?", help="Index name (default: current directory name)"
        )
        build_parser.add_argument(
            "--docs",
            type=str,
            nargs="+",
            default=["."],
            help="Documents directories and/or files (default: current directory)",
        )
        build_parser.add_argument(
            "--backend-name",
            type=str,
            default="hnsw",
            choices=["hnsw", "diskann", "ivf"],
            help="Backend to use (default: hnsw)",
        )
        _default_model = _default_embedding_model()
        build_parser.add_argument(
            "--embedding-model",
            type=str,
            default=_default_model,
            help=f"Embedding model (default: {_default_model})",
        )
        build_parser.add_argument(
            "--embedding-mode",
            type=str,
            default="sentence-transformers",
            choices=["sentence-transformers", "openai", "mlx", "ollama"],
            help="Embedding backend mode (default: sentence-transformers)",
        )
        build_parser.add_argument(
            "--embedding-host",
            type=str,
            default=None,
            help="Override Ollama-compatible embedding host",
        )
        build_parser.add_argument(
            "--embedding-api-base",
            type=str,
            default=None,
            help="Base URL for OpenAI-compatible embedding services",
        )
        build_parser.add_argument(
            "--embedding-api-key",
            type=str,
            default=None,
            help="API key for embedding service (defaults to OPENAI_API_KEY)",
        )
        build_parser.add_argument(
            "--embedding-prompt-template",
            type=str,
            default=None,
            help="Prompt template to prepend to all texts for embedding (e.g., 'query: ' for search)",
        )
        build_parser.add_argument(
            "--embedding-batch-size",
            type=int,
            default=None,
            help="Embedding batch size for sentence-transformers (overrides device defaults; also set LEANN_CUDA_BATCH_SIZE env)",
        )
        build_parser.add_argument(
            "--query-prompt-template",
            type=str,
            default=None,
            help="Prompt template for queries (different from build template for task-specific models)",
        )
        build_parser.add_argument(
            "--force",
            "-f",
            action="store_true",
            help="Force full rebuild of existing index (without this, build does incremental update: add new files only)",
        )
        build_parser.add_argument(
            "--graph-degree", type=int, default=32, help="Graph degree (default: 32)"
        )
        build_parser.add_argument(
            "--complexity", type=int, default=64, help="Build complexity (default: 64)"
        )
        build_parser.add_argument("--num-threads", type=int, default=1)
        build_parser.add_argument(
            "--compact",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Use compact (CSR) graph storage. Compact indices are read-only and cannot be updated incrementally. Default: false (allows incremental updates while still pruning embeddings for 97%% compression).",
        )
        build_parser.add_argument(
            "--recompute",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable recomputation (default: true)",
        )
        build_parser.add_argument(
            "--file-types",
            type=str,
            help="Comma-separated list of file extensions to include (e.g., '.txt,.pdf,.pptx'). If not specified, uses default supported types.",
        )
        build_parser.add_argument(
            "--include-hidden",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Include hidden files and directories (paths starting with '.') during indexing (default: false)",
        )
        build_parser.add_argument(
            "--doc-chunk-size",
            type=int,
            default=256,
            help="Document chunk size in TOKENS (default: 256). Final chunks may be larger due to overlap. For 512 token models: recommended 350 tokens (350 + 128 overlap = 478 max)",
        )
        build_parser.add_argument(
            "--doc-chunk-overlap",
            type=int,
            default=128,
            help="Document chunk overlap in TOKENS (default: 128). Added to chunk size, not included in it",
        )
        build_parser.add_argument(
            "--code-chunk-size",
            type=int,
            default=512,
            help="Code chunk size in TOKENS (default: 512). Final chunks may be larger due to overlap. For 512 token models: recommended 400 tokens (400 + 50 overlap = 450 max)",
        )
        build_parser.add_argument(
            "--code-chunk-overlap",
            type=int,
            default=50,
            help="Code chunk overlap in TOKENS (default: 50). Added to chunk size, not included in it",
        )
        build_parser.add_argument(
            "--use-ast-chunking",
            action="store_true",
            help="Enable AST-aware chunking for code files (requires astchunk)",
        )
        build_parser.add_argument(
            "--ast-chunk-size",
            type=int,
            default=300,
            help="AST chunk size in CHARACTERS (non-whitespace) (default: 300). Final chunks may be larger due to overlap and expansion. For 512 token models: recommended 300 chars (300 + 64 overlap ~= 480 tokens)",
        )
        build_parser.add_argument(
            "--ast-chunk-overlap",
            type=int,
            default=64,
            help="AST chunk overlap in CHARACTERS (default: 64). Added to chunk size, not included in it. ~1.2 tokens per character for code",
        )
        build_parser.add_argument(
            "--ast-fallback-traditional",
            action="store_true",
            default=True,
            help="Fall back to traditional chunking if AST chunking fails (default: True)",
        )
        build_parser.add_argument(
            "--id-scheme",
            choices=["sequential", "content-hash"],
            default="sequential",
            help=(
                "How passage IDs are assigned. 'sequential' (default) keys by insertion "
                "order; 'content-hash' uses sha256(text)[:16], stable across file moves "
                "and reorderings. See #329."
            ),
        )

        # Slack indexing command
        slack_parser = subparsers.add_parser("index-slack", help="Index Slack workspace via MCP")
        slack_parser.add_argument(
            "index_name", nargs="?", default="slack", help="Index name (default: slack)"
        )
        slack_parser.add_argument(
            "--mcp-server",
            type=str,
            required=True,
            help="MCP server command (e.g., 'slack-mcp-server')",
        )
        slack_parser.add_argument("--workspace-name", type=str, help="Slack workspace name")
        slack_parser.add_argument(
            "--channels",
            type=str,
            nargs="+",
            default=[],
            help="Specific channels to index (optional)",
        )
        add_embedding_args(slack_parser)

        # Watch command
        watch_parser = subparsers.add_parser(
            "watch",
            help="Monitor source files and auto-rebuild index when changes are detected",
        )
        watch_parser.add_argument("index_name", help="Index name")
        watch_parser.add_argument(
            "--interval",
            type=int,
            default=30,
            help="Poll interval in seconds (default: 30)",
        )
        watch_parser.add_argument(
            "--once",
            action="store_true",
            help="Check once for changes and exit (do not loop)",
        )
        watch_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report changes without rebuilding (original watch behavior)",
        )

        migrate_parser = subparsers.add_parser(
            "migrate-ids",
            help=(
                "Rewrite an existing index's passage IDs to content-hash form. "
                "Irreversible; back up the index first."
            ),
        )
        migrate_parser.add_argument("index_name", help="Index name")
        migrate_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would change without writing anything.",
        )
        migrate_parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Skip the interactive confirmation prompt.",
        )

        rebuild_parser = subparsers.add_parser(
            "rebuild",
            help="Rebuild an existing index using its stored config (delta by default; --force for full rebuild)",
        )
        rebuild_parser.add_argument("index_name", help="Index name")
        rebuild_parser.add_argument(
            "-f",
            "--force",
            action="store_true",
            help="Full rebuild from scratch instead of incremental delta",
        )

        # Search command
        search_parser = subparsers.add_parser("search", help="Search documents")
        search_parser.add_argument("index_name", help="Index name")
        search_parser.add_argument("query", help="Search query")
        search_parser.add_argument(
            "--top-k", type=int, default=5, help="Number of results (default: 5)"
        )
        search_parser.add_argument(
            "--complexity", type=int, default=64, help="Search complexity (default: 64)"
        )
        search_parser.add_argument("--beam-width", type=int, default=1)
        search_parser.add_argument("--prune-ratio", type=float, default=0.0)
        search_parser.add_argument(
            "--recompute",
            dest="recompute_embeddings",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable/disable embedding recomputation (default: enabled). Should not do a `no-recompute` search in a `recompute` build.",
        )
        search_parser.add_argument(
            "--pruning-strategy",
            choices=["global", "local", "proportional"],
            default="global",
            help="Pruning strategy (default: global)",
        )
        search_parser.add_argument(
            "--json",
            action="store_true",
            help="Output results as JSON array (machine-readable)",
        )
        search_parser.add_argument(
            "--non-interactive",
            action="store_true",
            help="Non-interactive mode: automatically select index without prompting",
        )
        search_parser.add_argument(
            "--show-metadata",
            action="store_true",
            help="Display file paths and metadata in search results",
        )
        search_parser.add_argument(
            "--embedding-prompt-template",
            type=str,
            default=None,
            help="Prompt template to prepend to query for embedding (e.g., 'query: ' for search)",
        )
        search_parser.add_argument(
            "--daemon",
            dest="use_daemon",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable/disable cross-process daemon reuse for embedding server (default: enabled)",
        )
        search_parser.add_argument(
            "--daemon-ttl",
            type=int,
            default=900,
            help="Daemon idle TTL in seconds (default: 900, 0 = never expire)",
        )
        search_parser.add_argument(
            "--warmup",
            dest="enable_warmup",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable/disable warmup when starting embedding server (default: enabled)",
        )
        search_parser.add_argument(
            "--metadata-filters",
            type=str,
            default=None,
            help=(
                "Filter results by metadata fields (JSON string). "
                'Format: \'{"field": {"operator": value}}\'. '
                "Operators: ==, !=, <, <=, >, >=, in, not_in, contains, starts_with, ends_with. "
                'Example: \'{"chapter": {"<=": 5}, "genre": {"==": "fiction"}}\''
            ),
        )

        # Warmup command
        warmup_parser = subparsers.add_parser("warmup", help="Warm up an index embedding server")
        warmup_parser.add_argument("index_name", help="Index name")
        warmup_parser.add_argument(
            "--daemon",
            dest="use_daemon",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable/disable daemon mode for warmup (default: enabled)",
        )
        warmup_parser.add_argument(
            "--daemon-ttl",
            type=int,
            default=900,
            help="Daemon idle TTL in seconds (default: 900, 0 = never expire)",
        )
        warmup_parser.add_argument(
            "--warmup",
            dest="enable_warmup",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable/disable warmup request itself (default: enabled)",
        )

        # Daemon command
        daemon_parser = subparsers.add_parser("daemon", help="Manage embedding daemons")
        daemon_subparsers = daemon_parser.add_subparsers(dest="daemon_command")

        daemon_start = daemon_subparsers.add_parser("start", help="Start daemon for an index")
        daemon_start.add_argument("index_name", help="Index name")
        daemon_start.add_argument(
            "--daemon-ttl",
            type=int,
            default=900,
            help="Daemon idle TTL in seconds (default: 900, 0 = never expire)",
        )
        daemon_start.add_argument(
            "--warmup",
            dest="enable_warmup",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable/disable startup warmup (default: enabled)",
        )

        daemon_stop = daemon_subparsers.add_parser("stop", help="Stop daemon(s)")
        daemon_stop.add_argument("index_name", nargs="?", help="Index name to stop")
        daemon_stop.add_argument(
            "--all",
            action="store_true",
            help="Stop all LEANN embedding daemons",
        )

        daemon_status = daemon_subparsers.add_parser("status", help="Show daemon status")
        daemon_status.add_argument("index_name", nargs="?", help="Optional index name filter")

        # Ask command
        ask_parser = subparsers.add_parser("ask", help="Ask questions")
        ask_parser.add_argument("index_name", help="Index name")
        ask_parser.add_argument(
            "query",
            nargs="?",
            help="Question to ask (omit for prompt or when using --interactive)",
        )
        ask_parser.add_argument(
            "--llm",
            type=str,
            default="ollama",
            choices=[
                "simulated",
                "ollama",
                "hf",
                "openai",
                "anthropic",
                "litellm",
                "minimax",
                "novita",
                "atlascloud",
                "atlas-cloud",
                "atlas",
            ],
            help="LLM provider (default: ollama)",
        )
        ask_parser.add_argument(
            "--model", type=str, default="qwen3:8b", help="Model name (default: qwen3:8b)"
        )
        ask_parser.add_argument(
            "--host",
            type=str,
            default=None,
            help="Override Ollama-compatible host (defaults to LEANN_OLLAMA_HOST/OLLAMA_HOST)",
        )
        ask_parser.add_argument(
            "--interactive", "-i", action="store_true", help="Interactive chat mode"
        )
        ask_parser.add_argument(
            "--top-k", type=int, default=20, help="Retrieval count (default: 20)"
        )
        ask_parser.add_argument("--complexity", type=int, default=32)
        ask_parser.add_argument("--beam-width", type=int, default=1)
        ask_parser.add_argument("--prune-ratio", type=float, default=0.0)
        ask_parser.add_argument(
            "--recompute",
            dest="recompute_embeddings",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable/disable embedding recomputation during ask (default: enabled)",
        )
        ask_parser.add_argument(
            "--pruning-strategy",
            choices=["global", "local", "proportional"],
            default="global",
        )
        ask_parser.add_argument(
            "--thinking-budget",
            type=str,
            choices=["low", "medium", "high"],
            default=None,
            help="Thinking budget for reasoning models (low/medium/high). Supported by GPT-Oss:20b and other reasoning models.",
        )
        ask_parser.add_argument(
            "--api-base",
            type=str,
            default=None,
            help="Base URL for OpenAI-compatible APIs (e.g., http://localhost:10000/v1)",
        )
        ask_parser.add_argument(
            "--api-key",
            type=str,
            default=None,
            help="API key for cloud LLM providers",
        )
        ask_parser.add_argument(
            "--metadata-filters",
            type=str,
            default=None,
            help=(
                "Filter retrieved chunks by metadata fields before answering (JSON string). "
                'Format: \'{"field": {"operator": value}}\'. '
                "Operators: ==, !=, <, <=, >, >=, in, not_in, contains, starts_with, ends_with. "
                'Example: \'{"chapter": {"<=": 5}, "genre": {"==": "fiction"}}\''
            ),
        )

        # React command (multiturn retrieval agent)
        react_parser = subparsers.add_parser(
            "react", help="Use ReAct agent for multiturn retrieval and reasoning"
        )
        react_parser.add_argument("index_name", help="Index name")
        react_parser.add_argument("query", help="Question to research")
        react_parser.add_argument(
            "--llm",
            type=str,
            default="ollama",
            choices=[
                "simulated",
                "ollama",
                "hf",
                "openai",
                "anthropic",
                "litellm",
                "minimax",
                "novita",
                "atlascloud",
                "atlas-cloud",
                "atlas",
            ],
            help="LLM provider (default: ollama)",
        )
        react_parser.add_argument(
            "--model", type=str, default="qwen3:8b", help="Model name (default: qwen3:8b)"
        )
        react_parser.add_argument(
            "--host",
            type=str,
            default=None,
            help="Override Ollama-compatible host (defaults to LEANN_OLLAMA_HOST/OLLAMA_HOST)",
        )
        react_parser.add_argument(
            "--top-k", type=int, default=5, help="Number of results per search (default: 5)"
        )
        react_parser.add_argument(
            "--max-iterations",
            type=int,
            default=5,
            help="Maximum number of search iterations (default: 5)",
        )
        react_parser.add_argument(
            "--api-base",
            type=str,
            default=None,
            help="Base URL for OpenAI-compatible APIs (e.g., http://localhost:10000/v1)",
        )
        react_parser.add_argument(
            "--api-key",
            type=str,
            default=None,
            help="API key for cloud LLM providers",
        )
        react_parser.add_argument(
            "--serper-api-key",
            type=str,
            default=None,
            help="Serper API key for web search (or set SERPER_API_KEY env var)",
        )
        react_parser.add_argument(
            "--jina-api-key",
            type=str,
            default=None,
            help="Jina API key for page content fetching (or set JINA_API_KEY env var)",
        )

        # ── index-* commands: data source indexing ──────────────────────

        def _add_index_args(p, default_name):
            """Add common embedding and index args to an index-* subparser."""
            p.add_argument(
                "--index-name",
                type=str,
                default=default_name,
                help=f"Index name (default: {default_name})",
            )
            p.add_argument(
                "--embedding-model", type=str, default="facebook/contriever", help="Embedding model"
            )
            p.add_argument(
                "--embedding-mode",
                type=str,
                default="sentence-transformers",
                choices=["sentence-transformers", "openai", "mlx", "ollama"],
                help="Embedding backend",
            )
            p.add_argument("--embedding-host", type=str, default=None, help="Ollama embedding host")
            p.add_argument(
                "--embedding-api-base",
                type=str,
                default=None,
                help="OpenAI-compatible embedding base URL",
            )
            p.add_argument("--embedding-api-key", type=str, default=None, help="Embedding API key")
            p.add_argument(
                "--embedding-batch-size",
                type=int,
                default=None,
                help="Embedding batch size for sentence-transformers",
            )
            p.add_argument(
                "--max-count", type=int, default=1000, help="Max items to index (default: 1000)"
            )
            p.add_argument(
                "--no-recompute",
                action="store_true",
                help="Disable embedding recomputation (stores full embeddings)",
            )

        idx_browser = subparsers.add_parser(
            "index-browser", help="Index browser history (Chrome/Brave)"
        )
        _add_index_args(idx_browser, "browser_history")
        idx_browser.add_argument(
            "browser",
            nargs="?",
            default="chrome",
            choices=["chrome", "brave"],
            help="Browser to index (default: chrome)",
        )

        idx_email = subparsers.add_parser("index-email", help="Index Apple Mail")
        _add_index_args(idx_email, "email")

        idx_calendar = subparsers.add_parser("index-calendar", help="Index Apple Calendar events")
        _add_index_args(idx_calendar, "calendar")

        idx_imessage = subparsers.add_parser("index-imessage", help="Index iMessage conversations")
        _add_index_args(idx_imessage, "imessage")

        idx_wechat = subparsers.add_parser("index-wechat", help="Index WeChat chat history")
        _add_index_args(idx_wechat, "wechat")
        idx_wechat.add_argument(
            "--export-dir", type=str, required=True, help="Path to WeChat JSON export directory"
        )

        idx_chatgpt = subparsers.add_parser("index-chatgpt", help="Index ChatGPT export")
        _add_index_args(idx_chatgpt, "chatgpt")
        idx_chatgpt.add_argument(
            "--export-path",
            type=str,
            required=True,
            help="Path to ChatGPT export (chat.html or .zip)",
        )

        idx_claude = subparsers.add_parser("index-claude", help="Index Claude export")
        _add_index_args(idx_claude, "claude")
        idx_claude.add_argument(
            "--export-path", type=str, required=True, help="Path to Claude export (.json or .zip)"
        )

        # List command
        list_parser = subparsers.add_parser("list", help="List all indexes")
        list_parser.add_argument(
            "--max-depth",
            type=_non_negative_int,
            default=DEFAULT_INDEX_SCAN_DEPTH,
            help=(
                "Maximum directory depth for app index discovery "
                f"(default: {DEFAULT_INDEX_SCAN_DEPTH})"
            ),
        )

        # Remove command
        remove_parser = subparsers.add_parser("remove", help="Remove an index")
        remove_parser.add_argument("index_name", help="Index name to remove")
        remove_parser.add_argument(
            "--force", "-f", action="store_true", help="Force removal without confirmation"
        )

        # Serve command (HTTP API server)
        serve_parser = subparsers.add_parser(
            "serve", help="Start HTTP API server for LEANN vector DB"
        )
        serve_parser.add_argument(
            "--host", type=str, default=None, help="Host to bind to (default: 127.0.0.1)"
        )
        serve_parser.add_argument(
            "--port", type=int, default=None, help="Port to bind to (default: 8000)"
        )

        return parser

    def register_project_dir(self):
        """Register current project directory in global registry"""
        register_project_directory()

    def _build_gitignore_parser(self, docs_dir: str):
        """Build gitignore parser using gitignore-parser library."""
        from gitignore_parser import parse_gitignore

        # Try to parse the root .gitignore
        gitignore_path = Path(docs_dir) / ".gitignore"

        if gitignore_path.exists():
            try:
                # gitignore-parser automatically handles all subdirectory .gitignore files!
                matches = parse_gitignore(str(gitignore_path))
                print(f"📋 Loaded .gitignore from {docs_dir} (includes all subdirectories)")
                return matches
            except Exception as e:
                print(f"Warning: Could not parse .gitignore: {e}")
        else:
            print("📋 No .gitignore found")

        # Fallback: basic pattern matching for essential files
        essential_patterns = {".git", ".DS_Store", "__pycache__", "node_modules", ".venv", "venv"}

        def basic_matches(file_path):
            path_parts = Path(file_path).parts
            return any(part in essential_patterns for part in path_parts)

        return basic_matches

    def _should_exclude_file(self, file_path: Path, gitignore_matches) -> bool:
        """Check if a file should be excluded using gitignore parser.

        Always match against absolute, posix-style paths for consistency with
        gitignore_parser expectations.
        """
        try:
            absolute_path = file_path.resolve()
        except Exception:
            absolute_path = Path(str(file_path))
        return gitignore_matches(absolute_path.as_posix())

    def _is_git_submodule(self, path: Path) -> bool:
        """Check if a path is a git submodule."""
        try:
            # Find the git repo root
            current_dir = Path.cwd()
            while current_dir != current_dir.parent:
                if (current_dir / ".git").exists():
                    gitmodules_path = current_dir / ".gitmodules"
                    if gitmodules_path.exists():
                        # Read .gitmodules to check if this path is a submodule
                        gitmodules_content = gitmodules_path.read_text()
                        # Convert path to relative to git root
                        try:
                            relative_path = path.resolve().relative_to(current_dir)
                            # Check if this path appears in .gitmodules
                            return f"path = {relative_path}" in gitmodules_content
                        except ValueError:
                            # Path is not under git root
                            return False
                    break
                current_dir = current_dir.parent
            return False
        except Exception:
            # If anything goes wrong, assume it's not a submodule
            return False

    @staticmethod
    def _project_has_discoverable_indexes(project_path: Path, max_depth: int) -> bool:
        """Return whether a project contains an index within the scan boundary."""
        cli_indexes_dir = project_path / ".leann" / "indexes"
        return cli_indexes_dir.exists() or any(
            iter_index_meta_files(project_path, max_depth=max_depth)
        )

    @staticmethod
    def _nested_project_dirs(project_path: Path, projects: list[Path]) -> list[Path]:
        """Return registered project roots nested below a project."""
        return [
            candidate
            for candidate in projects
            if candidate != project_path and candidate.is_relative_to(project_path)
        ]

    def list_indexes(self, max_depth: int = DEFAULT_INDEX_SCAN_DEPTH):
        # Get all project directories with .leann
        global_registry = Path.home() / ".leann" / "projects.json"
        all_projects = []

        if global_registry.exists():
            try:
                import json

                with open(global_registry) as f:
                    all_projects = json.load(f)
            except Exception:
                pass

        # Filter to existing projects with a discoverable CLI- or App-format index.
        valid_projects = []
        for project_dir in all_projects:
            project_path = Path(project_dir)
            if project_path.exists() and self._project_has_discoverable_indexes(
                project_path, max_depth
            ):
                valid_projects.append(project_path)

        # Add current project if it has a discoverable index but is not in registry.
        current_path = Path.cwd()
        if (
            self._project_has_discoverable_indexes(current_path, max_depth)
            and current_path not in valid_projects
        ):
            valid_projects.append(current_path)

        # Separate current and other projects
        other_projects = []

        for project_path in valid_projects:
            if project_path != current_path:
                other_projects.append(project_path)

        # Exclude only registered projects nested below the current project.
        # Registered ancestors must not hide indexes owned by the current directory.
        current_exclude_dirs = self._nested_project_dirs(current_path, valid_projects)

        print("📚 LEANN Indexes")
        print("=" * 50)

        total_indexes = 0
        current_indexes_count = 0

        # Show current project first (most important)
        print("\n🏠 Current Project")
        print(f"   {current_path}")
        print("   " + "─" * 45)

        current_indexes = self._discover_indexes_in_project(
            current_path, exclude_dirs=current_exclude_dirs, max_depth=max_depth
        )
        if current_indexes:
            for idx in current_indexes:
                total_indexes += 1
                current_indexes_count += 1
                type_icon = "📁" if idx["type"] == "cli" else "📄"
                print(f"   {current_indexes_count}. {type_icon} {idx['name']} {idx['status']}")
                if idx["size_mb"] > 0:
                    print(f"      📦 Size: {idx['size_mb']:.1f} MB")
        else:
            print("   📭 No indexes in current project")

        # Show other projects (reference information)
        if other_projects:
            print("\n\n🗂️  Other Projects")
            print("   " + "─" * 45)

            for project_path in other_projects:
                nested_projects = self._nested_project_dirs(project_path, valid_projects)
                project_indexes = self._discover_indexes_in_project(
                    project_path, exclude_dirs=nested_projects, max_depth=max_depth
                )
                if not project_indexes:
                    continue

                print(f"\n   📂 {project_path.name}")
                print(f"      {project_path}")

                for idx in project_indexes:
                    total_indexes += 1
                    type_icon = "📁" if idx["type"] == "cli" else "📄"
                    print(f"      • {type_icon} {idx['name']} {idx['status']}")
                    if idx["size_mb"] > 0:
                        print(f"        📦 {idx['size_mb']:.1f} MB")

        # Summary and usage info
        print("\n" + "=" * 50)
        if total_indexes == 0:
            print("💡 Get started:")
            print("   leann build my-docs --docs ./documents")
        else:
            # Count only projects that have at least one discoverable index
            projects_count = 0
            for p in valid_projects:
                if p == current_path:
                    discovered = self._discover_indexes_in_project(
                        p, exclude_dirs=current_exclude_dirs, max_depth=max_depth
                    )
                else:
                    nested_projects = self._nested_project_dirs(p, valid_projects)
                    discovered = self._discover_indexes_in_project(
                        p, exclude_dirs=nested_projects, max_depth=max_depth
                    )
                if len(discovered) > 0:
                    projects_count += 1
            print(f"📊 Total: {total_indexes} indexes across {projects_count} projects")

            if current_indexes_count > 0:
                print("\n💫 Quick start (current project):")
                # Get first index from current project for example
                current_indexes_dir = current_path / ".leann" / "indexes"
                if current_indexes_dir.exists():
                    current_index_dirs = [d for d in current_indexes_dir.iterdir() if d.is_dir()]
                    if current_index_dirs:
                        example_name = current_index_dirs[0].name
                        print(f'   leann search {example_name} "your query"')
                        print(f"   leann ask {example_name} --interactive")
            else:
                print("\n💡 Create your first index:")
                print("   leann build my-docs --docs ./documents")

    def _discover_indexes_in_project(
        self,
        project_path: Path,
        exclude_dirs: Optional[list[Path]] = None,
        max_depth: Optional[int] = None,
    ):
        """Discover all indexes in a project directory (both CLI and apps formats)

        exclude_dirs: when provided, skip any APP-format index files that are
        located under these directories. This prevents duplicates when the
        current project is a parent directory of other registered projects.
        max_depth: maximum number of directories to descend for APP-format indexes.
            None preserves full-depth discovery for non-CLI callers.
        """
        indexes = []
        exclude_dirs = exclude_dirs or []
        # normalize to resolved paths once for comparison
        try:
            exclude_dirs_resolved = [p.resolve() for p in exclude_dirs]
        except Exception:
            exclude_dirs_resolved = exclude_dirs

        # 1. CLI format: .leann/indexes/index_name/
        cli_indexes_dir = project_path / ".leann" / "indexes"
        if cli_indexes_dir.exists():
            for index_dir in cli_indexes_dir.iterdir():
                if index_dir.is_dir():
                    meta_file = index_dir / "documents.leann.meta.json"
                    status = "✅" if meta_file.exists() else "❌"

                    size_mb = 0
                    if meta_file.exists():
                        try:
                            size_mb = sum(
                                f.stat().st_size for f in index_dir.iterdir() if f.is_file()
                            ) / (1024 * 1024)
                        except (OSError, PermissionError):
                            pass

                    indexes.append(
                        {
                            "name": index_dir.name,
                            "type": "cli",
                            "status": status,
                            "size_mb": size_mb,
                            "path": index_dir,
                        }
                    )

        # 2. Apps format: *.leann.meta.json files within the configured scan depth
        cli_indexes_dir = project_path / ".leann" / "indexes"
        for meta_file in iter_index_meta_files(project_path, max_depth=max_depth):
            if meta_file.is_file():
                # Skip CLI-built indexes (which store meta under .leann/indexes/<name>/)
                try:
                    if cli_indexes_dir.exists() and cli_indexes_dir in meta_file.parents:
                        continue
                except Exception:
                    pass
                # Skip meta files that live under excluded directories
                try:
                    meta_parent_resolved = meta_file.parent.resolve()
                    if any(
                        meta_parent_resolved.is_relative_to(ex_dir)
                        for ex_dir in exclude_dirs_resolved
                    ):
                        continue
                except Exception:
                    # best effort; if resolve or comparison fails, do not exclude
                    pass
                # Use the parent directory name as the app index display name
                display_name = meta_file.parent.name
                # Extract file base used to store files
                file_base = meta_file.name.replace(".leann.meta.json", "")

                # Apps indexes are considered complete if the .leann.meta.json file exists
                status = "✅"

                # Calculate total size of all related files (use file base)
                size_mb = 0
                try:
                    index_dir = meta_file.parent
                    for related_file in index_dir.glob(f"{file_base}.leann*"):
                        size_mb += related_file.stat().st_size / (1024 * 1024)
                except (OSError, PermissionError):
                    pass

                indexes.append(
                    {
                        "name": display_name,
                        "type": "app",
                        "status": status,
                        "size_mb": size_mb,
                        "path": meta_file,
                    }
                )

        return indexes

    def remove_index(self, index_name: str, force: bool = False):
        """Safely remove an index - always show all matches for transparency"""

        # Always do a comprehensive search for safety
        print(f"🔍 Searching for all indexes named '{index_name}'...")
        all_matches = self._find_all_matching_indexes(index_name)

        if not all_matches:
            print(f"❌ Index '{index_name}' not found in any project.")
            return False

        if len(all_matches) == 1:
            return self._remove_single_match(all_matches[0], index_name, force)
        else:
            return self._remove_from_multiple_matches(all_matches, index_name, force)

    def _find_all_matching_indexes(self, index_name: str):
        """Find all indexes with the given name across all projects"""
        matches = []

        # Get all registered projects
        global_registry = Path.home() / ".leann" / "projects.json"
        all_projects = []

        if global_registry.exists():
            try:
                import json

                with open(global_registry) as f:
                    all_projects = json.load(f)
            except Exception:
                pass

        # Always include current project
        current_path = Path.cwd()
        if str(current_path) not in all_projects:
            all_projects.append(str(current_path))

        # Search across all projects
        for project_dir in all_projects:
            project_path = Path(project_dir)
            if not project_path.exists():
                continue

            # 1) CLI-format index under .leann/indexes/<name>
            index_dir = project_path / ".leann" / "indexes" / index_name
            if index_dir.exists():
                is_current = project_path == current_path
                matches.append(
                    {
                        "project_path": project_path,
                        "index_dir": index_dir,
                        "is_current": is_current,
                        "kind": "cli",
                    }
                )

            # 2) App-format indexes
            # We support two ways of addressing apps:
            #   a) by the file base (e.g., `pdf_documents`)
            #   b) by the parent directory name (e.g., `new_txt`)
            seen_app_meta = set()

            # 2a) by file base
            for meta_file in project_path.rglob(f"{index_name}.leann.meta.json"):
                if meta_file.is_file():
                    # Skip CLI-built indexes' meta under .leann/indexes
                    try:
                        cli_indexes_dir = project_path / ".leann" / "indexes"
                        if cli_indexes_dir.exists() and cli_indexes_dir in meta_file.parents:
                            continue
                    except Exception:
                        pass
                    is_current = project_path == current_path
                    key = (str(project_path), str(meta_file))
                    if key in seen_app_meta:
                        continue
                    seen_app_meta.add(key)
                    matches.append(
                        {
                            "project_path": project_path,
                            "files_dir": meta_file.parent,
                            "meta_file": meta_file,
                            "is_current": is_current,
                            "kind": "app",
                            "display_name": meta_file.parent.name,
                            "file_base": meta_file.name.replace(".leann.meta.json", ""),
                        }
                    )

            # 2b) by parent directory name
            for meta_file in project_path.rglob("*.leann.meta.json"):
                if meta_file.is_file() and meta_file.parent.name == index_name:
                    # Skip CLI-built indexes' meta under .leann/indexes
                    try:
                        cli_indexes_dir = project_path / ".leann" / "indexes"
                        if cli_indexes_dir.exists() and cli_indexes_dir in meta_file.parents:
                            continue
                    except Exception:
                        pass
                    is_current = project_path == current_path
                    key = (str(project_path), str(meta_file))
                    if key in seen_app_meta:
                        continue
                    seen_app_meta.add(key)
                    matches.append(
                        {
                            "project_path": project_path,
                            "files_dir": meta_file.parent,
                            "meta_file": meta_file,
                            "is_current": is_current,
                            "kind": "app",
                            "display_name": meta_file.parent.name,
                            "file_base": meta_file.name.replace(".leann.meta.json", ""),
                        }
                    )

        # Sort: current project first, then by project name
        matches.sort(key=lambda x: (not x["is_current"], x["project_path"].name))
        return matches

    def _remove_single_match(self, match, index_name: str, force: bool):
        """Handle removal when only one match is found"""
        project_path = match["project_path"]
        is_current = match["is_current"]
        kind = match.get("kind", "cli")

        if is_current:
            location_info = "current project"
            emoji = "🏠"
        else:
            location_info = f"other project '{project_path.name}'"
            emoji = "📂"

        print(f"✅ Found 1 index named '{index_name}':")
        print(f"   {emoji} Location: {location_info}")
        if kind == "cli":
            print(f"   📍 Path: {project_path / '.leann' / 'indexes' / index_name}")
        else:
            print(f"   📍 Meta: {match['meta_file']}")

        if not force:
            if not is_current:
                print("\n⚠️  CROSS-PROJECT REMOVAL!")
                print("   This will delete the index from another project.")

            response = input(f"   ❓ Confirm removal from {location_info}? (y/N): ").strip().lower()
            if response not in ["y", "yes"]:
                print("   ❌ Removal cancelled.")
                return False

        if kind == "cli":
            return self._delete_index_directory(
                match["index_dir"],
                index_name,
                project_path if not is_current else None,
                is_app=False,
            )
        else:
            return self._delete_index_directory(
                match["files_dir"],
                match.get("display_name", index_name),
                project_path if not is_current else None,
                is_app=True,
                meta_file=match.get("meta_file"),
                app_file_base=match.get("file_base"),
            )

    def _remove_from_multiple_matches(self, matches, index_name: str, force: bool):
        """Handle removal when multiple matches are found"""

        print(f"⚠️  Found {len(matches)} indexes named '{index_name}':")
        print("   " + "─" * 50)

        for i, match in enumerate(matches, 1):
            project_path = match["project_path"]
            is_current = match["is_current"]
            kind = match.get("kind", "cli")

            if is_current:
                print(f"   {i}. 🏠 Current project ({'CLI' if kind == 'cli' else 'APP'})")
            else:
                print(f"   {i}. 📂 {project_path.name} ({'CLI' if kind == 'cli' else 'APP'})")

            # Show path details
            if kind == "cli":
                print(f"      📍 {project_path / '.leann' / 'indexes' / index_name}")
            else:
                print(f"      📍 {match['meta_file']}")

            # Show size info
            try:
                if kind == "cli":
                    size_mb = sum(
                        f.stat().st_size for f in match["index_dir"].iterdir() if f.is_file()
                    ) / (1024 * 1024)
                else:
                    file_base = match.get("file_base")
                    size_mb = 0.0
                    if file_base:
                        size_mb = sum(
                            f.stat().st_size
                            for f in match["files_dir"].glob(f"{file_base}.leann*")
                            if f.is_file()
                        ) / (1024 * 1024)
                print(f"      📦 Size: {size_mb:.1f} MB")
            except (OSError, PermissionError):
                pass

        print("   " + "─" * 50)

        if force:
            print("   ❌ Multiple matches found, but --force specified.")
            print("   Please run without --force to choose which one to remove.")
            return False

        try:
            choice = input(
                f"   ❓ Which one to remove? (1-{len(matches)}, or 'c' to cancel): "
            ).strip()
            if choice.lower() == "c":
                print("   ❌ Removal cancelled.")
                return False

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(matches):
                selected_match = matches[choice_idx]
                project_path = selected_match["project_path"]
                is_current = selected_match["is_current"]
                kind = selected_match.get("kind", "cli")

                location = "current project" if is_current else f"'{project_path.name}' project"
                print(f"   🎯 Selected: Remove from {location}")

                # Final confirmation for safety
                confirm = input(
                    f"   ❓ FINAL CONFIRMATION - Type '{index_name}' to proceed: "
                ).strip()
                if confirm != index_name:
                    print("   ❌ Confirmation failed. Removal cancelled.")
                    return False

                if kind == "cli":
                    return self._delete_index_directory(
                        selected_match["index_dir"],
                        index_name,
                        project_path if not is_current else None,
                        is_app=False,
                    )
                else:
                    return self._delete_index_directory(
                        selected_match["files_dir"],
                        selected_match.get("display_name", index_name),
                        project_path if not is_current else None,
                        is_app=True,
                        meta_file=selected_match.get("meta_file"),
                        app_file_base=selected_match.get("file_base"),
                    )
            else:
                print("   ❌ Invalid choice. Removal cancelled.")
                return False

        except (ValueError, KeyboardInterrupt):
            print("\n   ❌ Invalid input. Removal cancelled.")
            return False

    def _delete_index_directory(
        self,
        index_dir: Path,
        index_display_name: str,
        project_path: Optional[Path] = None,
        is_app: bool = False,
        meta_file: Optional[Path] = None,
        app_file_base: Optional[str] = None,
    ):
        """Delete a CLI index directory or APP index files safely."""
        try:
            if is_app:
                removed = 0
                errors = 0
                # Delete only files that belong to this app index (based on file base)
                pattern_base = app_file_base or ""
                for f in index_dir.glob(f"{pattern_base}.leann*"):
                    try:
                        f.unlink()
                        removed += 1
                    except Exception:
                        errors += 1
                # Best-effort: also remove the meta file if specified and still exists
                if meta_file and meta_file.exists():
                    try:
                        meta_file.unlink()
                        removed += 1
                    except Exception:
                        errors += 1

                if removed > 0 and errors == 0:
                    if project_path:
                        print(
                            f"✅ App index '{index_display_name}' removed from {project_path.name}"
                        )
                    else:
                        print(f"✅ App index '{index_display_name}' removed successfully")
                    return True
                elif removed > 0 and errors > 0:
                    print(
                        f"⚠️  App index '{index_display_name}' partially removed (some files couldn't be deleted)"
                    )
                    return True
                else:
                    print(
                        f"❌ No files found to remove for app index '{index_display_name}' in {index_dir}"
                    )
                    return False
            else:
                import shutil

                shutil.rmtree(index_dir)

                if project_path:
                    print(f"✅ Index '{index_display_name}' removed from {project_path.name}")
                else:
                    print(f"✅ Index '{index_display_name}' removed successfully")
                return True
        except Exception as e:
            print(f"❌ Error removing index '{index_display_name}': {e}")
            return False

    def load_documents(
        self,
        docs_paths: Union[str, list],
        custom_file_types: Union[str, None] = None,
        include_hidden: bool = False,
        args: Optional[dict[str, Any]] = None,
    ):
        # Handle both single path (string) and multiple paths (list) for backward compatibility
        if isinstance(docs_paths, str):
            docs_paths = [docs_paths]

        # Separate files and directories
        files = []
        directories = []
        for path in docs_paths:
            path_obj = Path(path)
            if path_obj.is_file():
                files.append(str(path_obj))
            elif path_obj.is_dir():
                # Check if this is a git submodule - if so, skip it
                if self._is_git_submodule(path_obj):
                    print(f"⚠️  Skipping git submodule: {path}")
                    continue
                directories.append(str(path_obj))
            else:
                print(f"⚠️  Warning: Path '{path}' does not exist, skipping...")
                continue

        # Print summary of what we're processing
        total_items = len(files) + len(directories)
        items_desc = []
        if files:
            items_desc.append(f"{len(files)} file{'s' if len(files) > 1 else ''}")
        if directories:
            items_desc.append(
                f"{len(directories)} director{'ies' if len(directories) > 1 else 'y'}"
            )

        print(f"Loading documents from {' and '.join(items_desc)} ({total_items} total):")
        if files:
            print(f"  📄 Files: {', '.join([Path(f).name for f in files])}")
        if directories:
            print(f"  📁 Directories: {', '.join(directories)}")

        if custom_file_types:
            print(f"Using custom file types: {custom_file_types}")

        all_documents = []

        # Helper to detect hidden path components
        def _path_has_hidden_segment(p: Path) -> bool:
            return any(part.startswith(".") and part not in [".", ".."] for part in p.parts)

        # First, process individual files if any
        if files:
            print(f"\n🔄 Processing {len(files)} individual file{'s' if len(files) > 1 else ''}...")

            # Load individual files using SimpleDirectoryReader with input_files
            # Note: We skip gitignore filtering for explicitly specified files
            try:
                # Group files by their parent directory for efficient loading
                from collections import defaultdict

                files_by_dir = defaultdict(list)
                for file_path in files:
                    file_path_obj = Path(file_path)
                    if not include_hidden and _path_has_hidden_segment(file_path_obj):
                        print(f"  ⚠️  Skipping hidden file: {file_path}")
                        continue
                    parent_dir = str(file_path_obj.parent)
                    files_by_dir[parent_dir].append(str(file_path_obj))

                # Load files from each parent directory
                for parent_dir, file_list in files_by_dir.items():
                    print(
                        f"  Loading {len(file_list)} file{'s' if len(file_list) > 1 else ''} from {parent_dir}"
                    )
                    try:
                        file_docs = SimpleDirectoryReader(
                            parent_dir,
                            input_files=file_list,
                            # exclude_hidden only affects directory scans; input_files are explicit
                            filename_as_id=True,
                        ).load_data()
                        for doc in file_docs:
                            if not doc.metadata.get("source"):
                                doc.metadata["source"] = doc.metadata.get("file_path", "")
                        all_documents.extend(file_docs)
                        print(
                            f"    ✅ Loaded {len(file_docs)} document{'s' if len(file_docs) > 1 else ''}"
                        )
                    except Exception as e:
                        print(f"    ❌ Warning: Could not load files from {parent_dir}: {e}")

            except Exception as e:
                print(f"❌ Error processing individual files: {e}")

        # Define file extensions to process
        if custom_file_types:
            # Parse custom file types from comma-separated string
            code_extensions = [ext.strip() for ext in custom_file_types.split(",") if ext.strip()]
            # Ensure extensions start with a dot
            code_extensions = [ext if ext.startswith(".") else f".{ext}" for ext in code_extensions]
        else:
            code_extensions = list(DEFAULT_INDEX_EXTENSIONS)

        # Process each directory
        if directories:
            print(
                f"\n🔄 Processing {len(directories)} director{'ies' if len(directories) > 1 else 'y'}..."
            )

        for docs_dir in directories:
            print(f"Processing directory: {docs_dir}")
            # Build gitignore parser for each directory
            gitignore_matches = self._build_gitignore_parser(docs_dir)

            # Try to use better PDF parsers first, but only if PDFs are requested
            documents = []
            # Use resolved absolute paths to avoid mismatches (symlinks, relative vs absolute)
            docs_path = Path(docs_dir).resolve()

            # Check if we should process PDFs
            should_process_pdfs = custom_file_types is None or ".pdf" in custom_file_types

            if should_process_pdfs:
                for file_path in docs_path.rglob("*.pdf"):
                    # Check if file matches any exclude pattern
                    try:
                        # Ensure both paths are resolved before computing relativity
                        file_path_resolved = file_path.resolve()
                        # Determine directory scope using the non-resolved path to avoid
                        # misclassifying symlinked entries as outside the docs directory
                        relative_path = file_path.relative_to(docs_path)
                        if not include_hidden and _path_has_hidden_segment(relative_path):
                            continue
                        # Use absolute path for gitignore matching
                        if self._should_exclude_file(file_path_resolved, gitignore_matches):
                            continue
                    except ValueError:
                        # Skip files that can't be made relative to docs_path
                        print(f"⚠️  Skipping file outside directory scope: {file_path}")
                        continue

                    print(f"Processing PDF: {file_path}")

                    # Try PyMuPDF first (best quality)
                    text = extract_pdf_text_with_pymupdf(str(file_path))
                    if text is None:
                        # Try pdfplumber
                        text = extract_pdf_text_with_pdfplumber(str(file_path))

                    if text:
                        # Create a simple document structure
                        from llama_index.core import Document

                        doc = Document(text=text, metadata={"source": str(file_path)})
                        documents.append(doc)
                    else:
                        # Fallback to default reader
                        print(f"Using default reader for {file_path}")
                        try:
                            default_docs = SimpleDirectoryReader(
                                str(file_path.parent),
                                exclude_hidden=not include_hidden,
                                filename_as_id=True,
                                required_exts=[file_path.suffix],
                            ).load_data()
                            documents.extend(default_docs)
                        except Exception as e:
                            print(f"Warning: Could not process {file_path}: {e}")

            # Load other file types with default reader
            # Exclude PDFs from code_extensions if they were already processed separately
            other_file_extensions = code_extensions
            if should_process_pdfs and ".pdf" in code_extensions:
                other_file_extensions = [ext for ext in code_extensions if ext != ".pdf"]

            try:
                # Create a custom file filter function using our PathSpec
                def file_filter(
                    file_path: str, docs_dir=docs_dir, gitignore_matches=gitignore_matches
                ) -> bool:
                    """Return True if file should be included (not excluded)"""
                    try:
                        docs_path_obj = Path(docs_dir).resolve()
                        file_path_obj = Path(file_path).resolve()
                        # Use absolute path for gitignore matching
                        _ = file_path_obj.relative_to(docs_path_obj)  # validate scope
                        return not self._should_exclude_file(file_path_obj, gitignore_matches)
                    except (ValueError, OSError):
                        return True  # Include files that can't be processed

                # Only load other file types if there are extensions to process
                if other_file_extensions:
                    other_docs = SimpleDirectoryReader(
                        docs_dir,
                        recursive=True,
                        encoding="utf-8",
                        required_exts=other_file_extensions,
                        file_extractor={},  # Use default extractors
                        exclude_hidden=not include_hidden,
                        filename_as_id=True,
                    ).load_data(show_progress=True)
                else:
                    other_docs = []

                # Filter documents after loading based on gitignore rules
                filtered_docs = []
                for doc in other_docs:
                    file_path = doc.metadata.get("file_path", "")
                    if file_filter(file_path):
                        doc.metadata["source"] = file_path
                        filtered_docs.append(doc)

                documents.extend(filtered_docs)
            except ValueError as e:
                if "No files found" in str(e):
                    print(f"No additional files found for other supported types in {docs_dir}.")
                else:
                    raise e

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} documents from {docs_dir}")

        documents = all_documents

        all_texts = []

        # Define code file extensions for intelligent chunking
        code_file_exts = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".cs",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".r",
            ".sql",
            ".sh",
            ".bash",
            ".zsh",
            ".fish",
            ".ps1",
            ".bat",
            ".json",
            ".yaml",
            ".yml",
            ".xml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            ".html",
            ".css",
            ".scss",
            ".less",
            ".vue",
            ".svelte",
            ".ipynb",
            ".R",
            ".jl",
        }

        print("start chunking documents")

        # Check if AST chunking is requested
        use_ast = getattr(args, "use_ast_chunking", False)

        if use_ast:
            print("🧠 Using AST-aware chunking for code files")
            try:
                # Import enhanced chunking utilities from packaged module
                from .chunking_utils import create_text_chunks

                # Use enhanced chunking with AST support
                chunk_texts = create_text_chunks(
                    documents,
                    chunk_size=self.node_parser.chunk_size,
                    chunk_overlap=self.node_parser.chunk_overlap,
                    use_ast_chunking=True,
                    ast_chunk_size=getattr(args, "ast_chunk_size", 768),
                    ast_chunk_overlap=getattr(args, "ast_chunk_overlap", 96),
                    code_file_extensions=None,  # Use defaults
                    ast_fallback_traditional=getattr(args, "ast_fallback_traditional", True),
                )

                # create_text_chunks now returns list[dict] with metadata preserved
                all_texts.extend(chunk_texts)

            except ImportError as e:
                print(
                    f"⚠️  AST chunking utilities not available in package ({e}), falling back to traditional chunking"
                )
                use_ast = False

        if not use_ast:
            # Use traditional chunking logic
            for doc in tqdm(documents, desc="Chunking documents", unit="doc"):
                # Check if this is a code file based on source path
                source_path = doc.metadata.get("source", "")
                file_path = doc.metadata.get("file_path", "")
                is_code_file = any(
                    (source_path or file_path).endswith(ext) for ext in code_file_exts
                )

                # For code files, prepend line numbers so chunks carry them
                if is_code_file:
                    from llama_index.core.schema import MediaResource

                    original_text = doc.get_content()
                    lines = original_text.split("\n")
                    width = len(str(len(lines)))
                    numbered = "\n".join(f"{i + 1:>{width}}|{line}" for i, line in enumerate(lines))
                    doc.text_resource = MediaResource(text=numbered)

                # Extract metadata to preserve with chunks
                chunk_metadata = {
                    "file_path": file_path or source_path,
                    "file_name": doc.metadata.get("file_name", ""),
                    "source": source_path,
                }

                # Add optional metadata if available
                if "creation_date" in doc.metadata:
                    chunk_metadata["creation_date"] = doc.metadata["creation_date"]
                if "last_modified_date" in doc.metadata:
                    chunk_metadata["last_modified_date"] = doc.metadata["last_modified_date"]

                # Use appropriate parser based on file type
                parser = self.code_parser if is_code_file else self.node_parser
                nodes = parser.get_nodes_from_documents([doc])

                for node in nodes:
                    text = node.get_content()
                    # For code chunks, trim a partial first line left by overlap
                    # (a valid line starts with digits followed by '|')
                    if is_code_file and text and not text[0].isdigit():
                        first_nl = text.find("\n")
                        if first_nl != -1:
                            text = text[first_nl + 1 :]
                    all_texts.append({"text": text, "metadata": chunk_metadata.copy()})

        print(f"Loaded {len(documents)} documents, {len(all_texts)} chunks")
        return all_texts

    def _parse_file_types(self, custom_file_types: Optional[str]) -> list[str]:
        return parse_include_extensions(custom_file_types)

    def _sync_ignore_patterns(self, include_hidden: bool) -> Optional[list[str]]:
        if include_hidden:
            return None
        return ["**/.*"]

    def _build_embedding_options(self, args) -> dict[str, Any]:
        """Build embedding provider options dict from CLI args."""
        opts: dict[str, Any] = {}
        if args.embedding_mode == "ollama":
            opts["host"] = resolve_ollama_host(args.embedding_host)
        elif args.embedding_mode == "openai":
            opts["base_url"] = resolve_openai_base_url(args.embedding_api_base)
            resolved_key = resolve_openai_api_key(args.embedding_api_key)
            if resolved_key:
                opts["api_key"] = resolved_key
        if args.query_prompt_template:
            if args.embedding_prompt_template:
                opts["build_prompt_template"] = args.embedding_prompt_template
            opts["query_prompt_template"] = args.query_prompt_template
        elif args.embedding_prompt_template:
            opts["prompt_template"] = args.embedding_prompt_template
        batch_size = getattr(args, "embedding_batch_size", None)
        if batch_size is not None:
            opts["batch_size"] = batch_size
        return opts

    def _build_config_from_args(
        self,
        args,
        docs_paths: list[str],
        *,
        doc_chunk_size: Optional[int] = None,
        doc_chunk_overlap: Optional[int] = None,
        code_chunk_size: Optional[int] = None,
        code_chunk_overlap: Optional[int] = None,
    ) -> dict[str, Any]:
        """Capture the CLI build settings needed to replay `leann build`."""
        embedding_options = self._build_embedding_options(args)
        config: dict[str, Any] = {
            "docs": [str(Path(path).resolve()) for path in docs_paths],
            "file_types": args.file_types,
            "include_hidden": bool(args.include_hidden),
            "doc_chunk_size": doc_chunk_size
            if doc_chunk_size is not None
            else int(args.doc_chunk_size),
            "doc_chunk_overlap": doc_chunk_overlap
            if doc_chunk_overlap is not None
            else int(args.doc_chunk_overlap),
            "code_chunk_size": code_chunk_size
            if code_chunk_size is not None
            else int(args.code_chunk_size),
            "code_chunk_overlap": code_chunk_overlap
            if code_chunk_overlap is not None
            else int(args.code_chunk_overlap),
            "use_ast_chunking": bool(args.use_ast_chunking),
            "ast_chunk_size": int(args.ast_chunk_size),
            "ast_chunk_overlap": int(args.ast_chunk_overlap),
            "ast_fallback_traditional": bool(args.ast_fallback_traditional),
            "graph_degree": int(args.graph_degree),
            "complexity": int(args.complexity),
            "num_threads": int(args.num_threads),
            "compact": bool(args.compact),
            "recompute": bool(args.recompute),
        }
        if "host" in embedding_options:
            config["embedding_host"] = embedding_options["host"]
        if "base_url" in embedding_options:
            config["embedding_api_base"] = embedding_options["base_url"]
        if args.embedding_prompt_template:
            config["embedding_prompt_template"] = args.embedding_prompt_template
        if args.query_prompt_template:
            config["query_prompt_template"] = args.query_prompt_template
        return config

    def _resolve_sync_scope(self, docs_paths: list[str]) -> tuple[list[str], list[str]]:
        directories: list[str] = []
        files: list[str] = []
        for path in docs_paths:
            path_obj = Path(path).resolve()
            if path_obj.is_dir():
                directories.append(str(path_obj))
            elif path_obj.is_file():
                files.append(str(path_obj))
        return sorted(directories), sorted(files)

    def _resolve_sync_roots(self, docs_paths: list[str]) -> list[str]:
        """Directory roots only (legacy). Prefer _resolve_sync_scope."""
        directories, _files = self._resolve_sync_scope(docs_paths)
        return directories

    def _create_synchronizers(
        self,
        index_dir: Path,
        directories: list[str],
        explicit_files: list[str],
        include_extensions: list[str],
        include_hidden: bool = False,
    ) -> list[FileSynchronizer]:
        """Create FileSynchronizers with snapshots stored in the index dir. Shared by build and watch."""
        synchronizers: list[FileSynchronizer] = []
        for root in directories:
            tag = hashlib.sha256(root.encode()).hexdigest()[:12]
            snapshot_path = str(index_dir / f"sync_{tag}.pickle")
            try:
                fs = FileSynchronizer(
                    root_dir=root,
                    include_extensions=include_extensions,
                    include_hidden=include_hidden,
                    snapshot_path=snapshot_path,
                )
                synchronizers.append(fs)
            except Exception as exc:
                print(f"Warning: Failed to init synchronizer for {root}: {exc}")
        if explicit_files:
            tag = hashlib.sha256("|".join(explicit_files).encode()).hexdigest()[:12]
            snapshot_path = str(index_dir / f"sync_files_{tag}.pickle")
            try:
                fs = FileSynchronizer(
                    explicit_files=explicit_files,
                    include_extensions=include_extensions,
                    include_hidden=include_hidden,
                    snapshot_path=snapshot_path,
                )
                synchronizers.append(fs)
            except Exception as exc:
                print(f"Warning: Failed to init synchronizer for explicit files: {exc}")
        return synchronizers

    def _build_synchronizers(
        self,
        docs_paths: list[str],
        index_dir: Path,
        file_types: Optional[str] = None,
        include_hidden: bool = False,
    ) -> list[FileSynchronizer]:
        """Create FileSynchronizers for build from docs_paths."""
        directories, files = self._resolve_sync_scope(docs_paths)
        include_extensions = self._parse_file_types(file_types)
        return self._create_synchronizers(
            index_dir, directories, files, include_extensions, include_hidden
        )

    def _detect_build_changes(
        self,
        synchronizers: list[FileSynchronizer],
    ) -> tuple[set[str], set[str], set[str]]:
        """Detect added/removed/modified files across all source roots using content hashes."""
        all_added: set[str] = set()
        all_removed: set[str] = set()
        all_modified: set[str] = set()
        for fs in synchronizers:
            added, removed, modified = fs.detect_changes()
            all_added.update(added)
            all_removed.update(removed)
            all_modified.update(modified)
        return all_added, all_removed, all_modified

    def _commit_synchronizers(self, synchronizers: list[FileSynchronizer]) -> None:
        """Persist all synchronizer snapshots after a successful build."""
        for fs in synchronizers:
            fs.commit()

    @staticmethod
    def _assign_chunk_ids(chunks: list[dict]) -> None:
        """Assign stable IDs to chunks based on their file path and position."""
        from collections import defaultdict

        by_path: dict[str, list] = defaultdict(list)
        for c in chunks:
            p = c.get("metadata", {}).get("file_path") or c.get("metadata", {}).get("source") or ""
            by_path[_normalize_path(p)].append(c)
        for path_key, path_chunks in by_path.items():
            for idx, c in enumerate(path_chunks):
                sid = hashlib.sha256(f"{path_key}:{idx}".encode()).hexdigest()[:16]
                c.setdefault("metadata", {})["id"] = sid
                c["id"] = sid

    @staticmethod
    def _assign_unique_chunk_ids(chunks: list[dict]) -> None:
        """Assign unique IDs for incremental (avoids collision when path lookup misses some old ids)."""
        for c in chunks:
            sid = uuid.uuid4().hex[:16]
            c.setdefault("metadata", {})["id"] = sid
            c["id"] = sid

    def _chunks_for_paths(self, all_texts: list[dict], paths: set[str]) -> list[dict]:
        """Filter chunks belonging to the given file paths."""
        return [
            c
            for c in all_texts
            if _normalize_path(
                c.get("metadata", {}).get("file_path") or c.get("metadata", {}).get("source") or ""
            )
            in paths
        ]

    def _existing_index_id_scheme(self, index_path: str) -> Optional[str]:
        """Return the passage_id_scheme recorded in an existing index's meta.json.

        Returns None when the index doesn't exist yet. Older indexes pre-#330
        have a meta file without passage_id_scheme and must be treated as
        sequential so incremental updates never mix ID schemes.
        """
        meta_path = Path(index_path).with_suffix(".leann.meta.json")
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f).get("passage_id_scheme", "sequential")
        except Exception:
            return None

    def _make_incremental_builder(self, args) -> "LeannBuilder":
        # For incremental updates, the existing index's scheme wins. Otherwise
        # IDs would mix schemes within one index, which breaks lookups.
        existing_scheme = self._existing_index_id_scheme(self.get_index_path(args.index_name))
        scheme = existing_scheme or getattr(args, "id_scheme", "sequential")
        if existing_scheme and getattr(args, "id_scheme", existing_scheme) != existing_scheme:
            print(
                f"Note: --id-scheme={args.id_scheme} ignored — index '{args.index_name}' "
                f"was built with passage_id_scheme={existing_scheme!r}, keeping that."
            )
        return LeannBuilder(
            backend_name=args.backend_name,
            embedding_model=args.embedding_model,
            embedding_mode=args.embedding_mode,
            embedding_options=self._build_embedding_options(args) or None,
            graph_degree=args.graph_degree,
            complexity=args.complexity,
            is_compact=args.compact,
            is_recompute=args.recompute,
            num_threads=args.num_threads,
            passage_id_scheme=scheme,
        )

    def _incremental_add_only(
        self,
        index_path: str,
        all_texts: list[dict],
        args,
        new_paths: set[str],
    ) -> bool:
        """Add-only incremental update (works for HNSW and IVF)."""
        new_chunks = self._chunks_for_paths(all_texts, new_paths)
        if not new_chunks:
            return False
        self._assign_chunk_ids(new_chunks)
        builder = self._make_incremental_builder(args)
        for chunk in new_chunks:
            builder.add_text(chunk["text"], metadata=chunk["metadata"])
        print(
            f"Incremental update: adding {len(new_chunks)} chunks from {len(new_paths)} new file(s)..."
        )
        builder.update_index(index_path)
        print(f"Index updated at {index_path}")
        return True

    def _incremental_ivf_remove_only(
        self, index_path: str, index_dir: Path, removed_paths: set[str], args
    ) -> bool:
        """IVF remove-only fast path: remove chunk IDs without loading or chunking documents."""
        passages_file = index_dir / "documents.leann.passages.jsonl"
        if not passages_file.exists():
            return False
        offset_file = index_dir / "documents.leann.passages.idx"
        live_ids: set[str] | None = None
        if offset_file.exists():
            with open(offset_file, "rb") as f:
                live_ids = set(pickle.load(f).keys())
        chunk_ids_by_file = self._load_chunk_ids_by_file(passages_file, live_ids=live_ids)
        roots = self._load_sync_roots(index_dir)
        ids_to_remove: list[str] = []
        seen_ids: set[str] = set()
        for p in removed_paths:
            for key in self._path_lookup_keys(p, roots):
                for pid in chunk_ids_by_file.get(key, []):
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        ids_to_remove.append(pid)
        if not ids_to_remove:
            return False
        print(
            f"Incremental IVF update (-{len(removed_paths)} removed): removing {len(ids_to_remove)} old chunks..."
        )
        builder = self._make_incremental_builder(args)
        builder.update_index(index_path, remove_passage_ids=ids_to_remove)
        print(f"Index updated at {index_path}")
        return True

    def _path_lookup_keys(self, path: str, roots: list[str]) -> list[str]:
        """Return path variations for lookup (sync paths may be relative to roots)."""
        keys = [path, _normalize_path(path)]
        for root in roots:
            candidate = str((Path(root) / path).resolve())
            if candidate not in keys:
                keys.append(candidate)
        return keys

    def _incremental_ivf_update(
        self,
        index_path: str,
        index_dir: Path,
        all_texts: list[dict],
        args,
        new_paths: set[str],
        removed_paths: set[str],
        modified_paths: set[str],
        sync_roots: list[str],
    ) -> bool:
        """IVF incremental update: remove old chunks for modified/removed files, add new chunks."""
        passages_file = index_dir / "documents.leann.passages.jsonl"
        offset_file = index_dir / "documents.leann.passages.idx"
        live_ids: set[str] | None = None
        if offset_file.exists():
            with open(offset_file, "rb") as f:
                live_ids = set(pickle.load(f).keys())
        chunk_ids_by_file = (
            self._load_chunk_ids_by_file(passages_file, live_ids=live_ids)
            if passages_file.exists()
            else {}
        )

        # Collect old chunk IDs to remove (modified + removed files)
        # Try all path variations: same file can have different path formats in passages
        ids_to_remove: list[str] = []
        seen_ids: set[str] = set()
        for p in modified_paths | removed_paths:
            for key in self._path_lookup_keys(p, sync_roots):
                for pid in chunk_ids_by_file.get(key, []):
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        ids_to_remove.append(pid)

        # Collect new chunks to add (modified + new files)
        # Build path set for matching: chunks may have paths in different formats
        changed_paths = new_paths | modified_paths
        path_set: set[str] = set()
        for p in changed_paths:
            path_set.update(self._path_lookup_keys(p, sync_roots))
        new_chunks = self._chunks_for_paths(all_texts, path_set)
        # Use unique IDs: passages can have mixed path formats so we may miss some ids_to_remove
        self._assign_unique_chunk_ids(new_chunks)

        if not ids_to_remove and not new_chunks:
            return False

        builder = self._make_incremental_builder(args)
        for chunk in new_chunks:
            builder.add_text(chunk["text"], metadata=chunk["metadata"])

        parts = []
        if ids_to_remove:
            parts.append(f"removing {len(ids_to_remove)} old chunks")
        if new_chunks:
            parts.append(f"adding {len(new_chunks)} new chunks")
        file_parts = []
        if new_paths:
            file_parts.append(f"+{len(new_paths)} added")
        if modified_paths:
            file_parts.append(f"~{len(modified_paths)} modified")
        if removed_paths:
            file_parts.append(f"-{len(removed_paths)} removed")
        print(f"Incremental IVF update ({', '.join(file_parts)}): {', '.join(parts)}...")

        builder.update_index(
            index_path, remove_passage_ids=ids_to_remove if ids_to_remove else None
        )
        print(f"Index updated at {index_path}")
        return True

    @staticmethod
    def _log_rebuild_reason(
        meta: dict, args, new_paths: set, removed_paths: set, modified_paths: set
    ) -> None:
        """Print a human-readable explanation of why incremental update is not possible."""
        if removed_paths or modified_paths:
            reasons = []
            if removed_paths:
                reasons.append(f"{len(removed_paths)} file(s) removed")
            if modified_paths:
                reasons.append(f"{len(modified_paths)} file(s) modified")
            print(
                f"Incremental update not possible ({', '.join(reasons)}); falling back to full rebuild."
            )
            return

        blockers = []
        if meta.get("backend_name") not in ("hnsw", "ivf"):
            blockers.append(
                f"backend '{meta.get('backend_name')}' does not support incremental updates"
            )
        if meta.get("is_compact", meta.get("backend_kwargs", {}).get("is_compact", True)):
            blockers.append("index is compact (read-only); rebuild with --no-compact to enable")
        if meta.get("embedding_model") != args.embedding_model:
            blockers.append(
                f"embedding model changed ('{meta.get('embedding_model')}' -> '{args.embedding_model}')"
            )
        if meta.get("embedding_mode") != args.embedding_mode:
            blockers.append(
                f"embedding mode changed ('{meta.get('embedding_mode')}' -> '{args.embedding_mode}')"
            )
        if blockers:
            print(
                f"Incremental update not possible: {'; '.join(blockers)}. Falling back to full rebuild."
            )
        else:
            changes = []
            if new_paths:
                changes.append(f"+{len(new_paths)} added")
            summary = ", ".join(changes) if changes else "unknown reason"
            print(f"Full rebuild starting ({summary})...")

    def _write_sync_config(
        self,
        index_dir: Path,
        directories: list[str],
        explicit_files: list[str],
        include_extensions: list[str],
        include_hidden: bool,
        build_config: Optional[dict[str, Any]] = None,
    ) -> None:
        sync_config_path = index_dir / "sync_roots.json"
        config = {
            "directories": directories,
            "files": explicit_files,
            "roots": directories,
            "include_extensions": include_extensions,
            "ignore_patterns": self._sync_ignore_patterns(include_hidden),
        }
        if build_config is not None:
            config["build_config"] = build_config
        with open(sync_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def _write_sync_config_for_docs(
        self,
        index_dir: Path,
        docs_paths: list[str],
        args,
        build_config: Optional[dict[str, Any]] = None,
    ) -> None:
        directories, files = self._resolve_sync_scope(docs_paths)
        self._write_sync_config(
            index_dir,
            directories,
            files,
            self._parse_file_types(args.file_types),
            args.include_hidden,
            build_config,
        )

    def _load_sync_scope(self, index_dir: Path) -> tuple[list[str], list[str], list[str], bool]:
        """Load sync scope from index dir (directories, files, extensions, include_hidden)."""
        sync_config_path = index_dir / "sync_roots.json"
        if not sync_config_path.exists():
            return [], [], list(DEFAULT_INDEX_EXTENSIONS), False
        try:
            with open(sync_config_path, encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            return [], [], list(DEFAULT_INDEX_EXTENSIONS), False
        directories = config.get("directories") or config.get("roots") or []
        files = config.get("files") or []
        include_extensions = config.get("include_extensions") or list(DEFAULT_INDEX_EXTENSIONS)
        include_hidden = config.get("ignore_patterns") is None
        return directories, files, include_extensions, include_hidden

    def _load_sync_roots(self, index_dir: Path) -> list[str]:
        """Load directory + explicit file paths for chunk ID lookup."""
        directories, files, _, _ = self._load_sync_scope(index_dir)
        return directories + files

    def _resolve_index_for_watch(self, index_name: str) -> Optional[dict[str, Path]]:
        if self.index_exists(index_name):
            index_dir = self.indexes_dir / index_name
            passages_file = index_dir / "documents.leann.passages.jsonl"
            return {"index_dir": index_dir, "passages_file": passages_file}

        all_matches = self._find_all_matching_indexes(index_name)
        if not all_matches:
            print(
                f"Index '{index_name}' not found. Use 'leann build {index_name} --docs <dir> [<dir2> ...]' to create it."
            )
            return None

        if len(all_matches) == 1:
            match = all_matches[0]
        else:
            current_matches = [m for m in all_matches if m.get("is_current")]
            match = current_matches[0] if current_matches else all_matches[0]
            location_desc = (
                "current project"
                if match.get("is_current")
                else f"project '{match['project_path'].name}'"
            )
            print(
                f"Found {len(all_matches)} indexes named '{index_name}', using index from {location_desc}"
            )

        if match.get("kind") == "cli":
            index_dir = match["index_dir"]
            passages_file = index_dir / "documents.leann.passages.jsonl"
        else:
            index_dir = match["meta_file"].parent
            file_base = match["file_base"]
            passages_file = index_dir / f"{file_base}.passages.jsonl"

        return {"index_dir": index_dir, "passages_file": passages_file}

    def _load_chunk_ids_by_file(
        self, passages_file: Path, live_ids: set[str] | None = None
    ) -> dict[str, list[str]]:
        """Load chunk IDs grouped by file path from passages.jsonl.

        If *live_ids* is provided, skip entries whose ID is not in the set
        (filters out stale entries left by prior incremental updates).
        """
        chunk_ids_by_file: dict[str, list[str]] = {}
        with open(passages_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metadata = data.get("metadata") or {}
                file_path = metadata.get("file_path") or metadata.get("source")
                if not file_path:
                    continue
                chunk_id = data.get("id")
                if chunk_id is None:
                    continue
                if live_ids is not None and str(chunk_id) not in live_ids:
                    continue
                normalized_path = str(Path(file_path).resolve())
                chunk_ids_by_file.setdefault(normalized_path, []).append(str(chunk_id))
                if file_path != normalized_path:
                    chunk_ids_by_file.setdefault(file_path, []).append(str(chunk_id))
        return chunk_ids_by_file

    async def index_slack(self, args):
        """Build an index from Slack workspace via MCP."""
        from .readers import SlackMCPReader

        index_name = args.index_name
        index_dir = self.indexes_dir / index_name
        index_path = str(index_dir / "documents.leann")

        print("📱 Indexing Slack workspace...")

        reader = SlackMCPReader(
            mcp_server_command=args.mcp_server,
            workspace_name=args.workspace_name,
        )
        documents = await reader.load_data(channels=args.channels)

        if not documents:
            print("❌ No Slack data found. Make sure MCP server is running.")
            return

        print(f"📚 Loaded {len(documents)} messages. Building index...")
        index_dir.mkdir(parents=True, exist_ok=True)

        embedding_options = {}
        if args.embedding_mode == "ollama":
            embedding_options["host"] = resolve_ollama_host(None)

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model=args.embedding_model,
            embedding_mode=args.embedding_mode,
            embedding_options=embedding_options or None,
            is_recompute=False,
            is_compact=False,
        )

        for doc in documents:
            builder.add_text(doc.text, metadata=doc.metadata)

        builder.build_index(index_path)
        print(f"✅ Slack index built at: {index_path}")
        print(f'   Usage: leann search {index_name} "query"')

    async def build_index(self, args):
        docs_paths = args.docs
        # Use current directory name if index_name not provided
        if args.index_name:
            index_name = args.index_name
        else:
            index_name = Path.cwd().name
            print(f"Using current directory name as index: '{index_name}'")

        index_dir = self.indexes_dir / index_name
        index_path = self.get_index_path(index_name)

        # Display all paths being indexed with file/directory distinction
        files = [p for p in docs_paths if Path(p).is_file()]
        directories = [p for p in docs_paths if Path(p).is_dir()]

        print(f"📂 Indexing {len(docs_paths)} path{'s' if len(docs_paths) > 1 else ''}:")
        if files:
            print(f"  📄 Files ({len(files)}):")
            for i, file_path in enumerate(files, 1):
                print(f"    {i}. {Path(file_path).resolve()}")
        if directories:
            print(f"  📁 Directories ({len(directories)}):")
            for i, dir_path in enumerate(directories, 1):
                print(f"    {i}. {Path(dir_path).resolve()}")

        # Configure chunking based on CLI args before loading documents
        # Guard against invalid configurations
        doc_chunk_size = max(1, int(args.doc_chunk_size))
        doc_chunk_overlap = max(0, int(args.doc_chunk_overlap))
        if doc_chunk_overlap >= doc_chunk_size:
            print(
                f"⚠️  Adjusting doc chunk overlap from {doc_chunk_overlap} to {doc_chunk_size - 1} (must be < chunk size)"
            )
            doc_chunk_overlap = doc_chunk_size - 1

        code_chunk_size = max(1, int(args.code_chunk_size))
        code_chunk_overlap = max(0, int(args.code_chunk_overlap))
        if code_chunk_overlap >= code_chunk_size:
            print(
                f"⚠️  Adjusting code chunk overlap from {code_chunk_overlap} to {code_chunk_size - 1} (must be < chunk size)"
            )
            code_chunk_overlap = code_chunk_size - 1

        self.node_parser = SentenceSplitter(
            chunk_size=doc_chunk_size,
            chunk_overlap=doc_chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n",
        )
        self.code_parser = SentenceSplitter(
            chunk_size=code_chunk_size,
            chunk_overlap=code_chunk_overlap,
            separator="\n",
            paragraph_separator="\n\n",
        )
        build_config = self._build_config_from_args(
            args,
            docs_paths,
            doc_chunk_size=doc_chunk_size,
            doc_chunk_overlap=doc_chunk_overlap,
            code_chunk_size=code_chunk_size,
            code_chunk_overlap=code_chunk_overlap,
        )

        # Detect changes first so we can skip load_documents for remove-only
        index_dir.mkdir(parents=True, exist_ok=True)
        synchronizers = self._build_synchronizers(
            docs_paths, index_dir, file_types=args.file_types, include_hidden=args.include_hidden
        )

        if index_dir.exists() and not args.force and synchronizers:
            meta_path = index_dir / "documents.leann.meta.json"
            new_paths, removed_paths, modified_paths = self._detect_build_changes(synchronizers)

            if not new_paths and not removed_paths and not modified_paths:
                print("Index up to date.")
                return

            # Show change summary (add / modify / delete) before build
            change_parts = []
            if new_paths:
                change_parts.append(f"+{len(new_paths)} added")
            if modified_paths:
                change_parts.append(f"~{len(modified_paths)} modified")
            if removed_paths:
                change_parts.append(f"-{len(removed_paths)} removed")
            if change_parts:
                print(f"Changes: {', '.join(change_parts)}")

            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)

                backend_name = meta.get("backend_name")
                is_compact = meta.get(
                    "is_compact", meta.get("backend_kwargs", {}).get("is_compact", True)
                )

                # Normalize model names for comparison — e.g. "all-MiniLM-L6-v2"
                # should match "sentence-transformers/all-MiniLM-L6-v2"
                def _normalize_model_name(name: str) -> str:
                    return name.rsplit("/", 1)[-1] if name else name

                same_embedding = (
                    _normalize_model_name(meta.get("embedding_model", ""))
                    == _normalize_model_name(args.embedding_model)
                    and meta.get("embedding_mode") == args.embedding_mode
                )

                # IVF supports remove+add, so it can handle modified and removed files incrementally
                can_ivf_update = backend_name == "ivf" and not is_compact and same_embedding
                # HNSW only supports add (no remove), so it needs add-only changes
                can_add_only = (
                    not removed_paths
                    and not modified_paths
                    and backend_name in ("hnsw", "ivf")
                    and not is_compact
                    and same_embedding
                )

                # Remove-only fast path: no load/chunk, just remove IDs from index
                if can_ivf_update and removed_paths and not new_paths and not modified_paths:
                    result = self._incremental_ivf_remove_only(
                        index_path, index_dir, removed_paths, args
                    )
                    if result:
                        self._commit_synchronizers(synchronizers)
                        self._write_sync_config_for_docs(index_dir, docs_paths, args, build_config)
                        self.register_project_dir()
                        return

                # Resolve paths relative to sync scope (directories + explicit files).
                sync_dirs, sync_files = self._resolve_sync_scope(docs_paths)
                scope_roots = sync_dirs + sync_files
                paths_to_load = new_paths | modified_paths
                resolved_paths: list[str] = []
                for p in paths_to_load:
                    path_obj = Path(p)
                    if path_obj.is_absolute() and path_obj.exists():
                        resolved_paths.append(p)
                    else:
                        for root in scope_roots:
                            candidate = Path(root) / p
                            if candidate.exists():
                                resolved_paths.append(str(candidate.resolve()))
                                break
                        else:
                            resolved_paths.append(p)  # fallback: pass as-is
                all_texts = self.load_documents(
                    resolved_paths,
                    args.file_types,
                    include_hidden=args.include_hidden,
                    args=args,
                )
                # Proceed even when all_texts is empty (e.g. file emptied): we still need to remove old chunks
                if not all_texts and not (can_ivf_update and (modified_paths or removed_paths)):
                    print("No documents found")
                    return

                if can_ivf_update and (new_paths or modified_paths or removed_paths):
                    result = self._incremental_ivf_update(
                        index_path,
                        index_dir,
                        all_texts,
                        args,
                        new_paths,
                        removed_paths,
                        modified_paths,
                        scope_roots,
                    )
                    if result:
                        self._commit_synchronizers(synchronizers)
                        self._write_sync_config_for_docs(index_dir, docs_paths, args, build_config)
                        self.register_project_dir()
                        return

                elif can_add_only and new_paths:
                    result = self._incremental_add_only(
                        index_path,
                        all_texts,
                        args,
                        new_paths,
                    )
                    if result:
                        self._commit_synchronizers(synchronizers)
                        self._write_sync_config_for_docs(index_dir, docs_paths, args, build_config)
                        self.register_project_dir()
                        return

                else:
                    self._log_rebuild_reason(meta, args, new_paths, removed_paths, modified_paths)
                    # Force full reload — the partial all_texts from incremental
                    # path only contains changed files, not the full corpus.
                    all_texts = self.load_documents(
                        docs_paths, args.file_types, include_hidden=args.include_hidden, args=args
                    )

        # Full rebuild: load documents if not already loaded (first build or force)
        try:
            _ = all_texts
        except NameError:
            all_texts = self.load_documents(
                docs_paths, args.file_types, include_hidden=args.include_hidden, args=args
            )
        if not all_texts:
            print("No documents found")
            return

        print(f"Building index '{index_name}' with {args.backend_name} backend...")
        publish_from_staging = _existing_index_artifacts(index_dir)
        target_index_dir = index_dir
        target_index_path = index_path
        if publish_from_staging:
            target_index_dir = index_dir.with_name(f".{index_dir.name}.rebuild-{uuid.uuid4().hex}")
            target_index_path = str(target_index_dir / "documents.leann")

        builder = LeannBuilder(
            backend_name=args.backend_name,
            embedding_model=args.embedding_model,
            embedding_mode=args.embedding_mode,
            embedding_options=self._build_embedding_options(args) or None,
            graph_degree=args.graph_degree,
            complexity=args.complexity,
            is_compact=args.compact,
            is_recompute=args.recompute,
            num_threads=args.num_threads,
            passage_id_scheme=getattr(args, "id_scheme", "sequential"),
        )

        for chunk in all_texts:
            builder.add_text(chunk["text"], metadata=chunk["metadata"])

        try:
            builder.build_index(target_index_path)
            target_synchronizers = (
                self._build_synchronizers(
                    docs_paths,
                    target_index_dir,
                    file_types=args.file_types,
                    include_hidden=args.include_hidden,
                )
                if publish_from_staging
                else synchronizers
            )
            for fs in target_synchronizers:
                fs.create_snapshot()
            self._write_sync_config_for_docs(target_index_dir, docs_paths, args, build_config)
            if publish_from_staging:
                _publish_rebuilt_index(target_index_dir, index_dir)
        except Exception:
            if publish_from_staging and target_index_dir.exists():
                import shutil

                shutil.rmtree(target_index_dir)
            raise
        print(f"Index built at {index_path}")
        self.register_project_dir()

    def _watch_check_changes(self, index_name: str) -> tuple[set[str], set[str], set[str]]:
        """Check for file changes using the same snapshots as build (index_dir)."""
        resolved = self._resolve_index_for_watch(index_name)
        if not resolved:
            return set(), set(), set()

        index_dir = resolved["index_dir"]
        sync_config_path = index_dir / "sync_roots.json"
        if not sync_config_path.exists():
            return set(), set(), set()

        directories, files, include_extensions, include_hidden = self._load_sync_scope(index_dir)
        if not directories and not files:
            return set(), set(), set()

        synchronizers = self._create_synchronizers(
            index_dir,
            directories,
            files,
            include_extensions,
            include_hidden,
        )
        return self._detect_build_changes(synchronizers)

    def _watch_report_changes(
        self,
        index_name: str,
        added: set[str],
        removed: set[str],
        modified: set[str],
    ) -> None:
        """Print a summary of detected file changes."""
        resolved = self._resolve_index_for_watch(index_name)
        passages_file = resolved["passages_file"] if resolved else None

        chunk_ids_by_file: dict[str, list[str]] = {}
        if passages_file and passages_file.exists():
            chunk_ids_by_file = self._load_chunk_ids_by_file(passages_file)

        print("\n=== Changes detected ===")
        for label, paths in (
            ("added", sorted(added)),
            ("removed", sorted(removed)),
            ("modified", sorted(modified)),
        ):
            if not paths:
                continue
            print(f"\n{label} ({len(paths)}):")
            for file_path in paths:
                normalized_path = str(Path(file_path).resolve())
                chunk_ids = chunk_ids_by_file.get(normalized_path) or chunk_ids_by_file.get(
                    file_path, []
                )
                chunk_display = ", ".join(chunk_ids) if chunk_ids else "(not in index)"
                print(f"  - {file_path}")
                print(f"    chunks: {chunk_display}")

    def _reconstruct_build_args(
        self, index_name: str, *, force: bool = False, verbose: bool = False
    ) -> Optional[list[str]]:
        """Reconstruct the `leann build` CLI args for an existing index from its
        stored config (.meta.json + sync_roots.json).

        Returns None when the index can't be rebuilt this way (no sync config,
        missing metadata, etc.). With verbose=True the reason gets printed (for
        the user-driven `leann rebuild` path); with verbose=False the failures
        are silent (for the watch loop, which polls and shouldn't spam logs).
        """
        resolved = self._resolve_index_for_watch(index_name)
        if not resolved:
            return None
        index_dir = resolved["index_dir"]
        sync_config_path = index_dir / "sync_roots.json"
        if not sync_config_path.exists():
            if verbose:
                print(
                    f"Cannot rebuild '{index_name}': no sync config at {sync_config_path}. "
                    f"This usually means the index was built via the Python API rather than "
                    f"`leann build --docs <dir>`, so the original docs roots aren't recorded."
                )
            return None
        with open(sync_config_path, encoding="utf-8") as f:
            config = json.load(f)
        roots = config.get("roots") or []
        if not roots:
            if verbose:
                print(f"Cannot rebuild '{index_name}': sync config has no document roots.")
            return None

        meta_path = index_dir / "documents.leann.meta.json"
        if not meta_path.exists():
            if verbose:
                print(f"Cannot rebuild '{index_name}': index metadata missing at {meta_path}.")
            else:
                print(f"Index metadata missing for '{index_name}', cannot rebuild.")
            return None
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        build_config = config.get("build_config") or meta.get("build_config") or {}
        docs = build_config.get("docs") or roots
        build_args_list = [
            "build",
            index_name,
            "--docs",
            *[str(doc) for doc in docs],
            "--backend-name",
            meta.get("backend_name", "hnsw"),
            "--embedding-model",
            meta.get("embedding_model", "all-MiniLM-L6-v2"),
            "--embedding-mode",
            meta.get("embedding_mode", "sentence-transformers"),
        ]
        bkw = meta.get("backend_kwargs", {})

        def add_option(flag: str, value: Any) -> None:
            if value is not None:
                build_args_list.extend([flag, str(value)])

        def add_bool(true_flag: str, false_flag: str, value: Any) -> None:
            if value is None:
                return
            build_args_list.append(true_flag if bool(value) else false_flag)

        def config_or_backend(config_key: str, backend_key: str) -> Any:
            if config_key in build_config:
                return build_config[config_key]
            return bkw.get(backend_key)

        add_option("--graph-degree", config_or_backend("graph_degree", "graph_degree"))
        add_option("--complexity", config_or_backend("complexity", "complexity"))
        add_option("--num-threads", config_or_backend("num_threads", "num_threads"))

        compact = (
            build_config["compact"]
            if "compact" in build_config
            else meta.get("is_compact", bkw.get("is_compact"))
        )
        recompute = (
            build_config["recompute"]
            if "recompute" in build_config
            else meta.get("is_recompute", bkw.get("is_recompute"))
        )
        add_bool("--compact", "--no-compact", compact)
        add_bool("--recompute", "--no-recompute", recompute)

        file_types = build_config.get("file_types")
        if file_types is None:
            include_extensions = config.get("include_extensions")
            if include_extensions:
                file_types = ",".join(include_extensions)
        add_option("--file-types", file_types)

        include_hidden = build_config.get("include_hidden")
        if include_hidden is None:
            include_hidden = config.get("ignore_patterns") is None
        if include_hidden:
            build_args_list.append("--include-hidden")

        add_option("--doc-chunk-size", build_config.get("doc_chunk_size"))
        add_option("--doc-chunk-overlap", build_config.get("doc_chunk_overlap"))
        add_option("--code-chunk-size", build_config.get("code_chunk_size"))
        add_option("--code-chunk-overlap", build_config.get("code_chunk_overlap"))
        if build_config.get("use_ast_chunking"):
            build_args_list.append("--use-ast-chunking")
        add_option("--ast-chunk-size", build_config.get("ast_chunk_size"))
        add_option("--ast-chunk-overlap", build_config.get("ast_chunk_overlap"))
        if build_config.get("ast_fallback_traditional"):
            build_args_list.append("--ast-fallback-traditional")

        embedding_options = meta.get("embedding_options") or {}
        embedding_host = build_config.get("embedding_host", embedding_options.get("host"))
        embedding_api_base = build_config.get(
            "embedding_api_base", embedding_options.get("base_url")
        )
        embedding_prompt_template = build_config.get("embedding_prompt_template")
        if embedding_prompt_template is None:
            embedding_prompt_template = embedding_options.get(
                "build_prompt_template", embedding_options.get("prompt_template")
            )
        query_prompt_template = build_config.get(
            "query_prompt_template", embedding_options.get("query_prompt_template")
        )
        add_option("--embedding-host", embedding_host)
        add_option("--embedding-api-base", embedding_api_base)
        add_option("--embedding-prompt-template", embedding_prompt_template)
        add_option("--query-prompt-template", query_prompt_template)
        if force:
            build_args_list.append("--force")
        return build_args_list

    async def rebuild_index(self, args) -> None:
        """Rebuild an existing index using its stored config.

        By default this is an incremental delta (leann build is idempotent and
        only re-indexes added/changed/removed files). Pass --force to do a full
        rebuild from scratch.
        """
        build_args_list = self._reconstruct_build_args(
            args.index_name, force=getattr(args, "force", False), verbose=True
        )
        if build_args_list is None:
            return
        parser = self.create_parser()
        build_args = parser.parse_args(build_args_list)
        await self.build_index(build_args)

    async def _watch_trigger_build(self, index_name: str) -> None:
        """Trigger an idempotent build for the given index, reusing its stored config."""
        build_args_list = self._reconstruct_build_args(index_name, verbose=False)
        if build_args_list is None:
            return
        parser = self.create_parser()
        build_args = parser.parse_args(build_args_list)
        await self.build_index(build_args)

    def migrate_ids(self, args) -> None:
        """Rewrite an existing index's passage IDs to content-hash form.

        Migration is purely a Python-side rewrite — the vector graph isn't
        touched, so FAISS labels stay valid. What gets rewritten:
          - <index>.passages.jsonl  : new IDs in each line's "id" field
          - <index>.passages.idx    : new offset map keyed by new IDs
          - <index>.ids.txt         : new label → ID mapping for FAISS
          - <index>.meta.json       : passage_id_scheme = "content-hash"
          - <index>.bm25.sqlite     : BM25 IDs, when an FTS5 artifact is configured

        Identical-text chunks collide on the same sha256 prefix; the later
        occurrence wins in the offset map (dedup). A `--preserve-duplicates`
        knob to suffix collisions can land separately.

        Irreversible. Prompts for confirmation unless --yes is passed.
        """
        import hashlib
        import shutil

        index_name = args.index_name
        index_path = self._resolve_index_path(index_name, purpose="migrate")
        if not index_path:
            return
        meta_path = Path(index_path).with_suffix(".leann.meta.json")
        if not meta_path.exists():
            print(f"Cannot migrate '{index_name}': metadata missing at {meta_path}.")
            return
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        current_scheme = meta.get("passage_id_scheme", "sequential")
        if current_scheme == "content-hash":
            print(f"Index '{index_name}' already uses content-hash IDs. Nothing to do.")
            return

        # Locate the sibling artifacts using the same conventions as build_index.
        index_dir = Path(index_path).parent
        index_base = Path(index_path).name
        passages_file = index_dir / f"{index_base}.passages.jsonl"
        offset_file = index_dir / f"{index_base}.passages.idx"
        base_no_leann = (
            index_base[: -len(".leann")] if index_base.endswith(".leann") else index_base
        )
        idmap_file = index_dir / f"{base_no_leann}.ids.txt"

        for p in (passages_file, offset_file):
            if not p.exists():
                print(f"Cannot migrate: required artifact missing at {p}.")
                return

        # Stream the passages to plan the rewrite and surface collision count
        # before committing to anything irreversible.
        old_ids: list[str] = []
        new_ids: list[str] = []
        with open(passages_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                old_ids.append(data["id"])
                new_ids.append(hashlib.sha256(data["text"].encode("utf-8")).hexdigest()[:16])

        unique_new = len(set(new_ids))
        collisions = len(new_ids) - unique_new
        print(
            f"Migrating '{index_name}': {len(new_ids)} passages → {unique_new} unique "
            f"content-hash IDs ({collisions} collision(s) will dedup)."
        )

        if args.dry_run:
            print("(dry-run; not writing anything)")
            return
        if not args.yes:
            confirm = input(
                "Proceed? This rewrites passages.jsonl, .idx, .ids.txt, .meta.json. [y/N] "
            )
            if confirm.strip().lower() not in ("y", "yes"):
                print("Aborted.")
                return

        # Stage writes into siblings, then atomically rename.
        new_passages = passages_file.with_suffix(passages_file.suffix + ".migrate")
        new_offsets: dict[str, int] = {}
        rewritten_passages: list[dict[str, Any]] = []
        with (
            open(passages_file, encoding="utf-8") as src,
            open(new_passages, "w", encoding="utf-8") as dst,
        ):
            idx = 0
            for line in src:
                if not line.strip():
                    continue
                data = json.loads(line)
                data["id"] = new_ids[idx]
                offset = dst.tell()
                json.dump(data, dst, ensure_ascii=False)
                dst.write("\n")
                new_offsets[new_ids[idx]] = offset
                rewritten_passages.append(data)
                idx += 1

        new_idx = offset_file.with_suffix(offset_file.suffix + ".migrate")
        with open(new_idx, "wb") as f:
            pickle.dump(new_offsets, f)

        new_idmap: Optional[Path] = None
        if idmap_file.exists():
            new_idmap = idmap_file.with_suffix(idmap_file.suffix + ".migrate")
            with open(new_idmap, "w", encoding="utf-8") as f:
                for nid in new_ids:
                    f.write(nid + "\n")

        bm25_db_name = meta.get("bm25_db")
        should_rebuild_bm25 = meta.get("bm25_backend") == "fts5"
        if should_rebuild_bm25 and not bm25_db_name:
            bm25_db_name = f"{index_base}.bm25.sqlite"
        new_bm25: Optional[Path] = None
        bm25_file: Optional[Path] = None
        if should_rebuild_bm25 and bm25_db_name:
            bm25_file = index_dir / bm25_db_name
            new_bm25 = bm25_file.with_suffix(bm25_file.suffix + ".migrate")
            bm25_index = Fts5BM25Index(str(new_bm25))
            bm25_index.fit(rewritten_passages)
            bm25_index.close()

        shutil.move(str(new_passages), str(passages_file))
        shutil.move(str(new_idx), str(offset_file))
        if new_idmap is not None:
            shutil.move(str(new_idmap), str(idmap_file))
        if new_bm25 is not None and bm25_file is not None:
            shutil.move(str(new_bm25), str(bm25_file))

        meta["passage_id_scheme"] = "content-hash"
        meta["version"] = "1.1"
        if should_rebuild_bm25 and bm25_db_name:
            meta["bm25_backend"] = "fts5"
            meta["bm25_db"] = bm25_db_name
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(
            f"✓ Migrated '{index_name}' to content-hash IDs. {collisions} collisions were deduped."
        )

    async def watch_index(self, args):
        index_name = args.index_name
        resolved = self._resolve_index_for_watch(index_name)
        if not resolved:
            return

        index_dir = resolved["index_dir"]
        sync_config_path = index_dir / "sync_roots.json"
        if not sync_config_path.exists():
            print(
                f"Sync config not found for index '{index_name}'. "
                f"Run 'leann build {index_name} --docs <dir>' first."
            )
            return

        dry_run = getattr(args, "dry_run", False)
        once = getattr(args, "once", False)
        interval = getattr(args, "interval", 5)

        if once:
            added, removed, modified = self._watch_check_changes(index_name)
            if not added and not removed and not modified:
                print("No changes detected.")
                return
            self._watch_report_changes(index_name, added, removed, modified)
            if not dry_run:
                await self._watch_trigger_build(index_name)
            return

        print(f"Watching index '{index_name}' (interval={interval}s, ctrl-c to stop)...")
        try:
            while True:
                added, removed, modified = self._watch_check_changes(index_name)
                if added or removed or modified:
                    self._watch_report_changes(index_name, added, removed, modified)
                    if not dry_run:
                        await self._watch_trigger_build(index_name)
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            print("\nWatch stopped.")

    def _resolve_index_path(
        self,
        index_name: str,
        *,
        non_interactive: bool = True,
        purpose: str = "use",
        quiet: bool = False,
    ) -> Optional[str]:
        """Resolve index path from current project or registered projects."""
        _print = (lambda *a, **kw: print(*a, file=sys.stderr, **kw)) if quiet else print

        if self.index_exists(index_name):
            return self.get_index_path(index_name)

        all_matches = self._find_all_matching_indexes(index_name)
        if not all_matches:
            _print(
                f"Index '{index_name}' not found. Use 'leann build {index_name} --docs <dir> [<dir2> ...]' to create it."
            )
            return None

        def _match_to_path(match: dict[str, Any]) -> str:
            if match["kind"] == "cli":
                return str(match["index_dir"] / "documents.leann")
            meta_file = match["meta_file"]
            file_base = match["file_base"]
            return str(meta_file.parent / f"{file_base}.leann")

        if len(all_matches) == 1:
            match = all_matches[0]
            project_info = (
                "current project"
                if match["is_current"]
                else f"project '{match['project_path'].name}'"
            )
            _print(f"Using index '{index_name}' from {project_info}")
            return _match_to_path(match)

        if non_interactive:
            current_matches = [m for m in all_matches if m["is_current"]]
            match = current_matches[0] if current_matches else all_matches[0]
            location_desc = (
                "current project"
                if match["is_current"]
                else f"project '{match['project_path'].name}'"
            )
            _print(
                f"Found {len(all_matches)} indexes named '{index_name}', using index from {location_desc}"
            )
            return _match_to_path(match)

        print(f"Found {len(all_matches)} indexes named '{index_name}':")
        for i, match in enumerate(all_matches, 1):
            project_path = match["project_path"]
            is_current = match["is_current"]
            kind = match.get("kind", "cli")
            if is_current:
                print(f"   {i}. 🏠 Current project ({'CLI' if kind == 'cli' else 'APP'})")
            else:
                print(f"   {i}. 📂 {project_path.name} ({'CLI' if kind == 'cli' else 'APP'})")

        try:
            choice = input(f"Which index to {purpose}? (1-{len(all_matches)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(all_matches):
                match = all_matches[choice_idx]
                project_info = (
                    "current project"
                    if match["is_current"]
                    else f"project '{match['project_path'].name}'"
                )
                print(f"Using index '{index_name}' from {project_info}")
                return _match_to_path(match)
            print("Invalid choice. Aborting.")
            return None
        except (ValueError, KeyboardInterrupt):
            print("Invalid input. Aborting.")
            return None

    async def search_documents(self, args):
        index_name = args.index_name
        query = args.query
        json_mode = getattr(args, "json", False)

        index_path = self._resolve_index_path(
            index_name,
            non_interactive=args.non_interactive,
            purpose="search",
            quiet=json_mode,
        )
        if not index_path:
            return

        # Build provider_options for runtime override
        provider_options = {}
        if args.embedding_prompt_template:
            provider_options["prompt_template"] = args.embedding_prompt_template

        # Parse --metadata-filters JSON string
        metadata_filters = None
        raw_filters = getattr(args, "metadata_filters", None)
        if raw_filters:
            try:
                metadata_filters = json.loads(raw_filters)
                if not isinstance(metadata_filters, dict):
                    print(
                        "Error: --metadata-filters must be a JSON object, "
                        'e.g. \'{"chapter": {"<=": 5}}\''
                    )
                    return
            except json.JSONDecodeError as e:
                print(f"Error: --metadata-filters is not valid JSON: {e}")
                return

        saved_fd = None
        if json_mode:
            sys.stdout.flush()
            try:
                saved_fd = os.dup(1)
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 1)
                os.close(devnull)
            except OSError:
                saved_fd = None

        try:
            searcher = LeannSearcher(
                index_path=index_path,
                enable_warmup=args.enable_warmup,
                use_daemon=args.use_daemon,
                daemon_ttl_seconds=args.daemon_ttl,
            )
            results = searcher.search(
                query,
                top_k=args.top_k,
                complexity=args.complexity,
                beam_width=args.beam_width,
                prune_ratio=args.prune_ratio,
                recompute_embeddings=args.recompute_embeddings,
                pruning_strategy=args.pruning_strategy,
                provider_options=provider_options if provider_options else None,
                metadata_filters=metadata_filters,
            )
        finally:
            if saved_fd is not None:
                sys.stdout.flush()
                os.dup2(saved_fd, 1)
                os.close(saved_fd)

        if json_mode:
            json_results = [
                {
                    "id": r.id,
                    "score": float(r.score),
                    "text": r.text,
                    "metadata": r.metadata,
                }
                for r in results
            ]
            print(json.dumps(json_results, ensure_ascii=False, indent=2))
            return

        print(f"Search results for '{query}' (top {len(results)}):")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.score:.3f}")

            if args.show_metadata and result.metadata:
                file_path = result.metadata.get("file_path", "")
                if file_path:
                    print(f"   File: {file_path}")

                file_name = result.metadata.get("file_name", "")
                if file_name and file_name != file_path:
                    print(f"   Name: {file_name}")

                if "creation_date" in result.metadata:
                    print(f"   Created: {result.metadata['creation_date']}")
                if "last_modified_date" in result.metadata:
                    print(f"   Modified: {result.metadata['last_modified_date']}")

            print(f"   {result.text}")
            print(f"   Source: {result.metadata.get('source', '')}")
            print()

    async def warmup_index(self, args):
        index_path = self._resolve_index_path(
            args.index_name,
            non_interactive=True,
            purpose="warm up",
        )
        if not index_path:
            return

        searcher = LeannSearcher(
            index_path=index_path,
            recompute_embeddings=True,
            enable_warmup=args.enable_warmup,
            use_daemon=args.use_daemon,
            daemon_ttl_seconds=args.daemon_ttl,
        )
        if args.enable_warmup:
            searcher.warmup()
        print(
            f"Warmed index '{args.index_name}' (daemon={'on' if args.use_daemon else 'off'}, ttl={args.daemon_ttl}s)"
        )

    async def daemon_command(self, args):
        if not args.daemon_command:
            print("Please specify one of: start, stop, status")
            return

        if args.daemon_command == "status":
            records = EmbeddingServerManager.list_daemons()
            if args.index_name:
                index_path = self._resolve_index_path(
                    args.index_name,
                    non_interactive=True,
                    purpose="check daemon status for",
                )
                if not index_path:
                    return
                meta_path = str(Path(f"{index_path}.meta.json").resolve())
                records = [
                    r
                    for r in records
                    if r.get("config_signature", {}).get("passages_file") == meta_path
                ]

            if not records:
                print("No active embedding daemons.")
                return

            print(f"Active embedding daemons: {len(records)}")
            for record in records:
                cfg = record.get("config_signature", {})
                print(
                    f"- pid={record.get('pid')} port={record.get('port')} backend={record.get('backend_module_name')} model={cfg.get('model_name')}"
                )
            return

        if args.daemon_command == "start":
            index_path = self._resolve_index_path(
                args.index_name,
                non_interactive=True,
                purpose="start daemon for",
            )
            if not index_path:
                return

            searcher = LeannSearcher(
                index_path=index_path,
                recompute_embeddings=True,
                enable_warmup=args.enable_warmup,
                use_daemon=True,
                daemon_ttl_seconds=args.daemon_ttl,
            )
            searcher.warmup()
            print(
                f"Daemon started for '{args.index_name}' (ttl={args.daemon_ttl}s, warmup={'on' if args.enable_warmup else 'off'})"
            )
            return

        if args.daemon_command == "stop":
            if args.all:
                stopped = EmbeddingServerManager.stop_daemons()
                print(f"Stopped {stopped} daemon(s).")
                return

            if not args.index_name:
                print("Provide an index name or pass --all.")
                return

            index_path = self._resolve_index_path(
                args.index_name,
                non_interactive=True,
                purpose="stop daemon for",
            )
            if not index_path:
                return
            meta_path = str(Path(f"{index_path}.meta.json").resolve())
            stopped = EmbeddingServerManager.stop_daemons(passages_file=meta_path)
            print(f"Stopped {stopped} daemon(s) for index '{args.index_name}'.")

    async def ask_questions(self, args):
        index_name = args.index_name
        index_path = self.get_index_path(index_name)

        if not self.index_exists(index_name):
            print(
                f"Index '{index_name}' not found. Use 'leann build {index_name} --docs <dir> [<dir2> ...]' to create it."
            )
            return

        print(f"Starting chat with index '{index_name}'...")
        print(f"Using {args.model} ({args.llm})")

        llm_config = {"type": args.llm, "model": args.model}
        if args.llm == "ollama":
            llm_config["host"] = resolve_ollama_host(args.host)
        elif args.llm == "openai":
            llm_config["base_url"] = resolve_openai_base_url(args.api_base)
            resolved_api_key = resolve_openai_api_key(args.api_key)
            if resolved_api_key:
                llm_config["api_key"] = resolved_api_key
        elif args.llm == "anthropic":
            # For Anthropic, pass base_url and API key if provided
            if args.api_base:
                llm_config["base_url"] = resolve_anthropic_base_url(args.api_base)
            if args.api_key:
                llm_config["api_key"] = args.api_key
        elif args.llm == "litellm":
            # LiteLLM infers the provider from the model prefix and reads that
            # provider's own env var; base_url/api_key are only for a proxy.
            base_url = resolve_litellm_base_url(args.api_base)
            if base_url:
                llm_config["base_url"] = base_url
            resolved_api_key = resolve_litellm_api_key(args.api_key)
            if resolved_api_key:
                llm_config["api_key"] = resolved_api_key
        elif args.llm == "minimax":
            llm_config["base_url"] = resolve_minimax_base_url(args.api_base)
            resolved_api_key = resolve_minimax_api_key(args.api_key)
            if resolved_api_key:
                llm_config["api_key"] = resolved_api_key
        elif args.llm in {"atlascloud", "atlas-cloud", "atlas"}:
            llm_config["type"] = "atlascloud"
            if args.model == "qwen3:8b":
                llm_config["model"] = "deepseek-ai/deepseek-v4-pro"
            llm_config["base_url"] = resolve_atlascloud_base_url(args.api_base)
            resolved_api_key = resolve_atlascloud_api_key(args.api_key)
            if resolved_api_key:
                llm_config["api_key"] = resolved_api_key

        # Parse --metadata-filters JSON string (mirrors `leann search`).
        # Run before constructing LeannChat so invalid filters fail fast without
        # spinning up an embedding server.
        metadata_filters = None
        raw_filters = getattr(args, "metadata_filters", None)
        if raw_filters:
            try:
                metadata_filters = json.loads(raw_filters)
                if not isinstance(metadata_filters, dict):
                    print(
                        "Error: --metadata-filters must be a JSON object, "
                        'e.g. \'{"chapter": {"<=": 5}}\''
                    )
                    return
            except json.JSONDecodeError as e:
                print(f"Error: --metadata-filters is not valid JSON: {e}")
                return

        chat = LeannChat(index_path=index_path, llm_config=llm_config)

        llm_kwargs: dict[str, Any] = {}
        if args.thinking_budget:
            llm_kwargs["thinking_budget"] = args.thinking_budget

        def _ask_once(prompt: str) -> None:
            query_start_time = time.time()
            response = chat.ask(
                prompt,
                top_k=args.top_k,
                complexity=args.complexity,
                beam_width=args.beam_width,
                prune_ratio=args.prune_ratio,
                recompute_embeddings=args.recompute_embeddings,
                pruning_strategy=args.pruning_strategy,
                metadata_filters=metadata_filters,
                llm_kwargs=llm_kwargs,
            )
            query_completion_time = time.time() - query_start_time
            print(f"LEANN: {response}")
            print(f"The query took {query_completion_time:.3f} seconds to finish")

        initial_query = (args.query or "").strip()

        if args.interactive:
            # Create interactive session
            session = create_cli_session(index_name)

            if initial_query:
                _ask_once(initial_query)

            session.run_interactive_loop(_ask_once)
        else:
            query = initial_query or input("Enter your question: ").strip()
            if not query:
                print("No question provided. Exiting.")
                return

            _ask_once(query)

    # ── index-* shared handler ──────────────────────────────────────

    async def _build_index_from_documents(self, args, documents: list):
        """Shared builder for all index-* commands.

        Accepts a list of llama-index Document objects (or dicts with 'text' and
        'metadata' keys), builds a LEANN index with proper embedding and
        recomputation settings.
        """
        if not documents:
            print("No documents loaded — nothing to index.")
            return

        index_name = args.index_name
        index_path = self.get_index_path(index_name)
        is_recompute = not getattr(args, "no_recompute", False)

        print(f"Building index '{index_name}' with {len(documents)} documents...")
        print(f"  Embedding: {args.embedding_model} ({args.embedding_mode})")
        print(f"  Recompute: {is_recompute}")

        for attr in (
            "embedding_prompt_template",
            "query_prompt_template",
            "embedding_host",
            "embedding_api_base",
            "embedding_api_key",
        ):
            if not hasattr(args, attr):
                setattr(args, attr, None)

        embedding_options = self._build_embedding_options(args) or None

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model=args.embedding_model,
            embedding_mode=args.embedding_mode,
            embedding_options=embedding_options,
            is_recompute=is_recompute,
        )

        for doc in documents:
            if hasattr(doc, "text"):
                builder.add_text(
                    doc.text, metadata=doc.metadata if hasattr(doc, "metadata") else {}
                )
            elif isinstance(doc, dict):
                builder.add_text(doc["text"], metadata=doc.get("metadata", {}))

        builder.build_index(index_path)
        self.register_project_dir()
        print(f"Index '{index_name}' built at {index_path}")

    async def index_browser(self, args):
        """Index browser history (Chrome or Brave)."""
        from apps.history_data.history import ChromeHistoryReader

        browser = getattr(args, "browser", "chrome")
        profile_paths = {
            "chrome": "~/Library/Application Support/Google/Chrome/Default",
            "brave": "~/Library/Application Support/BraveSoftware/Brave-Browser/Default",
        }
        profile = os.path.expanduser(profile_paths.get(browser, profile_paths["chrome"]))
        reader = ChromeHistoryReader()
        docs = reader.load_data(chrome_profile_path=profile, max_count=args.max_count)
        print(f"Loaded {len(docs)} {browser} history entries")
        await self._build_index_from_documents(args, docs)

    async def index_email(self, args):
        """Index Apple Mail."""
        from apps.email_data.LEANN_email_reader import EmlxReader, find_all_messages_directories

        msg_dirs = find_all_messages_directories()
        if not msg_dirs:
            print("No Apple Mail Messages directories found.")
            return
        reader = EmlxReader()
        docs = []
        for msg_dir in msg_dirs:
            docs.extend(reader.load_data(str(msg_dir), max_count=args.max_count))
        await self._build_index_from_documents(args, docs)

    async def index_calendar(self, args):
        """Index Apple Calendar events."""
        import shutil
        import sqlite3

        from llama_index.core import Document

        docs = []
        calendar_cache = Path.home() / "Library/Calendars/Calendar Cache"
        if not calendar_cache.exists():
            print("Apple Calendar Cache not found.")
            return

        temp_db = "/tmp/leann_calendar_index_copy"
        try:
            shutil.copy2(calendar_cache, temp_db)
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT summary, description, location,
                       datetime(start_date + 978307200, 'unixepoch', 'localtime') as start,
                       datetime(end_date + 978307200, 'unixepoch', 'localtime') as end_time
                FROM CI_EVENT ORDER BY start_date DESC LIMIT ?
                """,
                (args.max_count,),
            )
            for summary, description, location, start, end_time in cursor.fetchall():
                if not summary:
                    continue
                text = (
                    f"Event: {summary}\nStart: {start}\nEnd: {end_time}\n"
                    f"Location: {location or ''}\nDescription: {description or ''}"
                )
                docs.append(Document(text=text, metadata={"event": summary, "start": start}))
            conn.close()
        except Exception as e:
            print(f"Error reading Apple Calendar: {e}")
        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)
        await self._build_index_from_documents(args, docs)

    async def index_imessage(self, args):
        """Index iMessage conversations."""
        from apps.imessage_data.imessage_reader import IMessageReader

        reader = IMessageReader(concatenate_conversations=True)
        docs = reader.load_data()
        print(f"Loaded {len(docs)} iMessage conversations")
        await self._build_index_from_documents(args, docs)

    async def index_wechat(self, args):
        """Index WeChat chat history from exported JSON."""
        from apps.history_data.wechat_history import WeChatHistoryReader

        reader = WeChatHistoryReader()
        docs = reader.load_data(
            input_dir=args.export_dir,
            max_count=args.max_count,
            concatenate_messages=True,
        )
        print(f"Loaded {len(docs)} WeChat conversations")
        await self._build_index_from_documents(args, docs)

    async def index_chatgpt(self, args):
        """Index ChatGPT export data."""
        from apps.chatgpt_data.chatgpt_reader import ChatGPTReader

        reader = ChatGPTReader(concatenate_conversations=True)
        docs = reader.load_data(
            input_dir=args.export_path,
            max_count=args.max_count,
        )
        print(f"Loaded {len(docs)} ChatGPT conversations")
        await self._build_index_from_documents(args, docs)

    async def index_claude(self, args):
        """Index Claude export data."""
        from apps.claude_data.claude_reader import ClaudeReader

        reader = ClaudeReader(concatenate_conversations=True)
        docs = reader.load_data(
            input_dir=args.export_path,
            max_count=args.max_count,
        )
        print(f"Loaded {len(docs)} Claude conversations")
        await self._build_index_from_documents(args, docs)

    async def react_agent(self, args):
        """Run ReAct agent for multiturn retrieval."""
        index_name = args.index_name
        query = args.query

        # Find the index (similar to search_documents)
        index_path = self.get_index_path(index_name)
        if self.index_exists(index_name):
            pass
        else:
            all_matches = self._find_all_matching_indexes(index_name)
            if not all_matches:
                print(
                    f"Index '{index_name}' not found. Use 'leann build {index_name} --docs <dir> [<dir2> ...]' to create it."
                )
                return
            elif len(all_matches) == 1:
                match = all_matches[0]
                if match["kind"] == "cli":
                    index_path = str(match["index_dir"] / "documents.leann")
                else:
                    meta_file = match["meta_file"]
                    file_base = match["file_base"]
                    index_path = str(meta_file.parent / f"{file_base}.leann")
            else:
                # Multiple matches - use first one for now
                match = all_matches[0]
                if match["kind"] == "cli":
                    index_path = str(match["index_dir"] / "documents.leann")
                else:
                    meta_file = match["meta_file"]
                    file_base = match["file_base"]
                    index_path = str(meta_file.parent / f"{file_base}.leann")
                print(f"Found {len(all_matches)} indexes named '{index_name}', using first match")

        print(f"🤖 Starting ReAct agent with index '{index_name}'...")
        print(f"Using {args.model} ({args.llm})")

        llm_config = {"type": args.llm, "model": args.model}
        if args.llm == "ollama":
            llm_config["host"] = resolve_ollama_host(args.host)
        elif args.llm == "openai":
            llm_config["base_url"] = resolve_openai_base_url(args.api_base)
            resolved_api_key = resolve_openai_api_key(args.api_key)
            if resolved_api_key:
                llm_config["api_key"] = resolved_api_key
        elif args.llm == "anthropic":
            if args.api_base:
                llm_config["base_url"] = resolve_anthropic_base_url(args.api_base)
            if args.api_key:
                llm_config["api_key"] = args.api_key
        elif args.llm == "litellm":
            # LiteLLM infers the provider from the model prefix and reads that
            # provider's own env var; base_url/api_key are only for a proxy.
            base_url = resolve_litellm_base_url(args.api_base)
            if base_url:
                llm_config["base_url"] = base_url
            resolved_api_key = resolve_litellm_api_key(args.api_key)
            if resolved_api_key:
                llm_config["api_key"] = resolved_api_key
        elif args.llm == "minimax":
            llm_config["base_url"] = resolve_minimax_base_url(args.api_base)
            resolved_api_key = resolve_minimax_api_key(args.api_key)
            if resolved_api_key:
                llm_config["api_key"] = resolved_api_key
        elif args.llm in {"atlascloud", "atlas-cloud", "atlas"}:
            llm_config["type"] = "atlascloud"
            if args.model == "qwen3:8b":
                llm_config["model"] = "deepseek-ai/deepseek-v4-pro"
            llm_config["base_url"] = resolve_atlascloud_base_url(args.api_base)
            resolved_api_key = resolve_atlascloud_api_key(args.api_key)
            if resolved_api_key:
                llm_config["api_key"] = resolved_api_key

        from .react_agent import create_react_agent

        agent = create_react_agent(
            index_path=index_path,
            llm_config=llm_config,
            max_iterations=args.max_iterations,
            serper_api_key=getattr(args, "serper_api_key", None),
            jina_api_key=getattr(args, "jina_api_key", None),
        )

        if agent.web_search_available:
            print("🌐 Web search enabled (Serper API key detected)")
        else:
            print("📚 Local search only (no Serper API key)")

        print(f"\n🔍 Question: {query}\n")
        answer = agent.run(query, top_k=args.top_k)
        print(f"\n✅ Final Answer:\n{answer}\n")

        if agent.search_history:
            print(f"\n📊 Search History ({len(agent.search_history)} iterations):")
            for entry in agent.search_history:
                source_tag = f"[{entry.get('source', 'local')}]"
                print(
                    f"  {entry['iteration']}. {source_tag} {entry['action']} "
                    f"({entry['results_count']} results)"
                )

    async def serve_api(self, args):
        """Start the HTTP API server."""
        import os

        try:
            from .server import serve_async

            # Override host/port if provided via CLI args
            if args.host:
                os.environ["LEANN_SERVER_HOST"] = args.host
            if args.port:
                os.environ["LEANN_SERVER_PORT"] = str(args.port)

            # Must await uvicorn on this loop: sync uvicorn.run() starts a nested loop
            # and fails with "Cannot run the event loop while another loop is running".
            await serve_async()
        except ImportError as e:
            print(
                "❌ HTTP server dependencies not installed.\n"
                "Install them with:\n"
                "  uv pip install 'leann-core[server]'\n"
                "or:\n"
                "  uv pip install 'fastapi>=0.115' 'pydantic>=2' 'uvicorn[standard]'\n"
            )
            raise SystemExit(1) from e
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            raise SystemExit(1) from e

    async def run(self, args=None):
        parser = self.create_parser()

        if args is None:
            args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        # Determine whether to suppress C++ output
        # Default is to suppress (quiet mode), unless --verbose is specified
        suppress = not getattr(args, "verbose", False)

        if args.command == "list":
            self.list_indexes(args.max_depth)
        elif args.command == "remove":
            self.remove_index(args.index_name, args.force)
        elif args.command == "build":
            with suppress_cpp_output(suppress):
                await self.build_index(args)
        elif args.command == "watch":
            await self.watch_index(args)
        elif args.command == "migrate-ids":
            self.migrate_ids(args)
        elif args.command == "rebuild":
            with suppress_cpp_output(suppress):
                await self.rebuild_index(args)
        elif args.command == "search":
            with suppress_cpp_output(suppress):
                await self.search_documents(args)
        elif args.command == "warmup":
            with suppress_cpp_output(suppress):
                await self.warmup_index(args)
        elif args.command == "daemon":
            await self.daemon_command(args)
        elif args.command == "ask":
            with suppress_cpp_output(suppress):
                await self.ask_questions(args)
        elif args.command == "react":
            with suppress_cpp_output(suppress):
                await self.react_agent(args)
        elif args.command == "serve":
            await self.serve_api(args)
        elif args.command == "index-browser":
            with suppress_cpp_output(suppress):
                await self.index_browser(args)
        elif args.command == "index-email":
            with suppress_cpp_output(suppress):
                await self.index_email(args)
        elif args.command == "index-calendar":
            with suppress_cpp_output(suppress):
                await self.index_calendar(args)
        elif args.command == "index-wechat":
            with suppress_cpp_output(suppress):
                await self.index_wechat(args)
        elif args.command == "index-imessage":
            with suppress_cpp_output(suppress):
                await self.index_imessage(args)
        elif args.command == "index-slack":
            with suppress_cpp_output(suppress):
                await self.index_slack(args)
        elif args.command == "index-chatgpt":
            with suppress_cpp_output(suppress):
                await self.index_chatgpt(args)
        elif args.command == "index-claude":
            with suppress_cpp_output(suppress):
                await self.index_claude(args)
        else:
            parser.print_help()


def main():
    import logging

    import dotenv

    dotenv.load_dotenv()

    # Set clean logging for CLI usage
    logging.getLogger().setLevel(logging.WARNING)  # Only show warnings and errors

    cli = LeannCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
