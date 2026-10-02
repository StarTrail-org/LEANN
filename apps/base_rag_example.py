"""
Base class for unified RAG examples interface.
Provides common parameters and functionality for all RAG examples.
"""

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import dotenv
from leann.api import LeannBuilder, LeannChat

# Optional import: older PyPI builds may not include interactive_utils
try:
    from leann.interactive_utils import create_rag_session
except ImportError:

    def create_rag_session(app_name: str, data_description: str):
        class _SimpleSession:
            def run_interactive_loop(self, handler):
                print(f"Interactive session for {app_name}: {data_description}")
                print("Interactive mode not available in this build")

        return _SimpleSession()


from leann.registry import register_project_directory

# Optional import: older PyPI builds may not include settings
try:
    from leann.settings import (
        resolve_atlascloud_api_key,
        resolve_atlascloud_base_url,
        resolve_cheaperinference_api_key,
        resolve_cheaperinference_base_url,
        resolve_litellm_api_key,
        resolve_litellm_base_url,
        resolve_ollama_host,
        resolve_openai_api_key,
        resolve_openai_base_url,
    )
except ImportError:
    # Minimal fallbacks if settings helpers are unavailable
    import os

    def resolve_litellm_api_key(value: str | None) -> str | None:
        return value or os.getenv("LITELLM_API_KEY") or os.getenv("LEANN_LITELLM_API_KEY")

    def resolve_litellm_base_url(value: str | None) -> str | None:
        return value or os.getenv("LITELLM_BASE_URL") or os.getenv("LITELLM_API_BASE")

    def resolve_ollama_host(value: str | None) -> str | None:
        return value or os.getenv("LEANN_OLLAMA_HOST") or os.getenv("OLLAMA_HOST")

    def resolve_openai_api_key(value: str | None) -> str | None:
        return value or os.getenv("OPENAI_API_KEY")

    def resolve_openai_base_url(value: str | None) -> str | None:
        return value or os.getenv("OPENAI_BASE_URL")

    def resolve_atlascloud_api_key(value: str | None) -> str | None:
        return value or os.getenv("ATLASCLOUD_API_KEY") or os.getenv("ATLAS_CLOUD_API_KEY")

    def resolve_atlascloud_base_url(value: str | None) -> str | None:
        return (
            value
            or os.getenv("ATLASCLOUD_BASE_URL")
            or os.getenv("ATLAS_CLOUD_BASE_URL")
            or "https://api.atlascloud.ai/v1"
        )

    def resolve_cheaperinference_api_key(value: str | None) -> str | None:
        return value or os.getenv("CHEAPER_INFERENCE_API_KEY")

    def resolve_cheaperinference_base_url(value: str | None) -> str | None:
        return (
            value
            or os.getenv("CHEAPER_INFERENCE_BASE_URL")
            or "https://api.cheaperinference.com/v1"
        )


dotenv.load_dotenv()


class BaseRAGExample(ABC):
    """Base class for all RAG examples with unified interface."""

    def __init__(
        self,
        name: str,
        description: str,
        default_index_name: str,
    ):
        self.name = name
        self.description = description
        self.default_index_name = default_index_name
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with common parameters."""
        parser = argparse.ArgumentParser(
            description=self.description, formatter_class=argparse.RawDescriptionHelpFormatter
        )

        # Core parameters (all examples share these)
        core_group = parser.add_argument_group("Core Parameters")
        core_group.add_argument(
            "--index-dir",
            type=str,
            default=f"./{self.default_index_name}",
            help=f"Directory to store the index (default: ./{self.default_index_name})",
        )
        core_group.add_argument(
            "--query",
            type=str,
            default=None,
            help="Query to run (if not provided, will run in interactive mode)",
        )
        # Allow subclasses to override default max_items
        max_items_default = getattr(self, "max_items_default", -1)
        core_group.add_argument(
            "--max-items",
            type=int,
            default=max_items_default,
            help="Maximum number of items to process  -1 for all, means index all documents, and you should set it to a reasonable number if you have a large dataset and try at the first time)",
        )
        core_group.add_argument(
            "--force-rebuild", action="store_true", help="Force rebuild index even if it exists"
        )

        # Embedding parameters
        embedding_group = parser.add_argument_group("Embedding Parameters")
        # Allow subclasses to override default embedding_model
        embedding_model_default = getattr(self, "embedding_model_default", "facebook/contriever")
        embedding_group.add_argument(
            "--embedding-model",
            type=str,
            default=embedding_model_default,
            help=f"Embedding model to use (default: {embedding_model_default}), we provide facebook/contriever, text-embedding-3-small,mlx-community/Qwen3-Embedding-0.6B-8bit or nomic-embed-text",
        )
        embedding_group.add_argument(
            "--embedding-mode",
            type=str,
            default="sentence-transformers",
            choices=["sentence-transformers", "openai", "mlx", "ollama"],
            help="Embedding backend mode (default: sentence-transformers), we provide sentence-transformers, openai, mlx, or ollama",
        )
        embedding_group.add_argument(
            "--embedding-host",
            type=str,
            default=None,
            help="Override Ollama-compatible embedding host",
        )
        embedding_group.add_argument(
            "--embedding-api-base",
            type=str,
            default=None,
            help="Base URL for OpenAI-compatible embedding services",
        )
        embedding_group.add_argument(
            "--embedding-api-key",
            type=str,
            default=None,
            help="API key for embedding service (defaults to OPENAI_API_KEY)",
        )

        # LLM parameters
        llm_group = parser.add_argument_group("LLM Parameters")
        llm_group.add_argument(
            "--llm",
            type=str,
            default="openai",
            choices=[
                "openai",
                "ollama",
                "hf",
                "litellm",
                "simulated",
                "atlascloud",
                "atlas-cloud",
                "atlas",
                "cheaperinference",
                "cheaper-inference",
            ],
            help="LLM backend: openai, ollama, hf, litellm, atlascloud, cheaperinference, or simulated (default: openai)",
        )
        llm_group.add_argument(
            "--llm-model",
            type=str,
            default=None,
            help="Model name (default: gpt-4o) e.g., gpt-4o-mini, llama3.2:1b, Qwen/Qwen2.5-1.5B-Instruct",
        )
        llm_group.add_argument(
            "--llm-host",
            type=str,
            default=None,
            help="Host for Ollama-compatible APIs (defaults to LEANN_OLLAMA_HOST/OLLAMA_HOST)",
        )
        llm_group.add_argument(
            "--thinking-budget",
            type=str,
            choices=["low", "medium", "high"],
            default=None,
            help="Thinking budget for reasoning models (low/medium/high). Supported by GPT-Oss:20b and other reasoning models.",
        )
        llm_group.add_argument(
            "--llm-api-base",
            type=str,
            default=None,
            help="Base URL for OpenAI-compatible APIs",
        )
        llm_group.add_argument(
            "--llm-api-key",
            type=str,
            default=None,
            help="API key for OpenAI-compatible APIs (defaults to OPENAI_API_KEY)",
        )

        # AST Chunking parameters
        ast_group = parser.add_argument_group("AST Chunking Parameters")
        ast_group.add_argument(
            "--use-ast-chunking",
            action="store_true",
            help="Enable AST-aware chunking for code files (requires astchunk)",
        )
        ast_group.add_argument(
            "--ast-chunk-size",
            type=int,
            default=300,
            help="Maximum CHARACTERS per AST chunk (default: 300). Final chunks may be larger due to overlap. For 512 token models: recommended 300 chars",
        )
        ast_group.add_argument(
            "--ast-chunk-overlap",
            type=int,
            default=64,
            help="Overlap between AST chunks in CHARACTERS (default: 64). Added to chunk size, not included in it",
        )
        ast_group.add_argument(
            "--code-file-extensions",
            nargs="+",
            default=None,
            help="Additional code file extensions to process with AST chunking (e.g., .py .java .cs .ts)",
        )
        ast_group.add_argument(
            "--ast-fallback-traditional",
            action="store_true",
            default=True,
            help="Fall back to traditional chunking if AST chunking fails (default: True)",
        )

        # Search parameters
        search_group = parser.add_argument_group("Search Parameters")
        search_group.add_argument(
            "--top-k", type=int, default=20, help="Number of results to retrieve (default: 20)"
        )
        search_group.add_argument(
            "--search-complexity",
            type=int,
            default=32,
            help="Search complexity for graph traversal (default: 64)",
        )

        # Index building parameters
        index_group = parser.add_argument_group("Index Building Parameters")
        index_group.add_argument(
            "--backend-name",
            type=str,
            default="hnsw",
            choices=["hnsw", "diskann", "ivf", "flashlib"],
            help="Backend to use for index (default: hnsw). 'flashlib' requires a CUDA GPU.",
        )
        index_group.add_argument(
            "--graph-degree",
            type=int,
            default=32,
            help="Graph degree for index construction (default: 32)",
        )
        index_group.add_argument(
            "--build-complexity",
            type=int,
            default=64,
            help="Build complexity for index construction (default: 64)",
        )
        index_group.add_argument(
            "--no-compact",
            action="store_true",
            help="Disable compact index storage",
        )
        index_group.add_argument(
            "--no-recompute",
            action="store_true",
            help="Disable embedding recomputation",
        )

        # Add source-specific parameters
        self._add_specific_arguments(parser)

        return parser

    @abstractmethod
    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        """Add source-specific arguments. Override in subclasses."""
        pass

    @abstractmethod
    async def load_data(self, args) -> list[dict[str, Any]]:
        """Load data from the source. Returns list of text chunks as dicts with 'text' and 'metadata' keys."""
        pass

    def get_llm_config(self, args) -> dict[str, Any]:
        """Get LLM configuration based on arguments."""
        config = {"type": args.llm}

        if args.llm == "openai":
            config["model"] = args.llm_model or "gpt-4o"
            config["base_url"] = resolve_openai_base_url(args.llm_api_base)
            resolved_key = resolve_openai_api_key(args.llm_api_key)
            if resolved_key:
                config["api_key"] = resolved_key
        elif args.llm == "litellm":
            config["model"] = args.llm_model or "gpt-4o"
            base_url = resolve_litellm_base_url(args.llm_api_base)
            if base_url:
                config["base_url"] = base_url
            resolved_key = resolve_litellm_api_key(args.llm_api_key)
            if resolved_key:
                config["api_key"] = resolved_key
        elif args.llm in {"atlascloud", "atlas-cloud", "atlas"}:
            config["type"] = "atlascloud"
            config["model"] = args.llm_model or "deepseek-ai/deepseek-v4-pro"
            config["base_url"] = resolve_atlascloud_base_url(args.llm_api_base)
            resolved_key = resolve_atlascloud_api_key(args.llm_api_key)
            if resolved_key:
                config["api_key"] = resolved_key
        elif args.llm in {"cheaperinference", "cheaper-inference"}:
            config["type"] = "cheaperinference"
            config["model"] = args.llm_model or "gpt-5.4-mini"
            config["base_url"] = resolve_cheaperinference_base_url(args.llm_api_base)
            resolved_key = resolve_cheaperinference_api_key(args.llm_api_key)
            if resolved_key:
                config["api_key"] = resolved_key
        elif args.llm == "ollama":
            config["model"] = args.llm_model or "llama3.2:1b"
            config["host"] = resolve_ollama_host(args.llm_host)
        elif args.llm == "hf":
            config["model"] = args.llm_model or "Qwen/Qwen2.5-1.5B-Instruct"
        elif args.llm == "simulated":
            # Simulated LLM doesn't need additional configuration
            pass

        return config

    @staticmethod
    def _resolve_chunk_token_limit(args) -> int | None:
        """Resolve the embedding model's token limit for token-aware chunking.

        Returns ``None`` if the limit cannot be determined (e.g. model unknown).
        Apps can pass the result as ``max_tokens_per_chunk=`` to
        ``create_text_chunks()``.
        """
        try:
            from leann.embedding_compute import get_model_token_limit

            base_url = getattr(args, "embedding_api_base", None)
            return get_model_token_limit(args.embedding_model, base_url)
        except Exception:
            return None

    async def build_index(self, args, texts: list[dict[str, Any]]) -> str:
        """Build LEANN index from text chunks (dicts with 'text' and 'metadata' keys)."""
        index_path = str(Path(args.index_dir) / f"{self.default_index_name}.leann")

        print(f"\n[Building Index] Creating {self.name} index...")
        print(f"Total text chunks: {len(texts)}")

        # Warn if any chunks may exceed the embedding model's token limit
        limit = self._resolve_chunk_token_limit(args)
        if limit:
            try:
                from leann.chunking_utils import validate_chunk_token_limits

                _texts = [t["text"] if isinstance(t, dict) else t for t in texts]
                validate_chunk_token_limits(_texts, limit)
            except Exception:
                pass

        embedding_options: dict[str, Any] = {}
        if args.embedding_mode == "ollama":
            embedding_options["host"] = resolve_ollama_host(args.embedding_host)
        elif args.embedding_mode == "openai":
            embedding_options["base_url"] = resolve_openai_base_url(args.embedding_api_base)
            resolved_embedding_key = resolve_openai_api_key(args.embedding_api_key)
            if resolved_embedding_key:
                embedding_options["api_key"] = resolved_embedding_key

        builder = LeannBuilder(
            backend_name=args.backend_name,
            embedding_model=args.embedding_model,
            embedding_mode=args.embedding_mode,
            embedding_options=embedding_options or None,
            graph_degree=args.graph_degree,
            complexity=args.build_complexity,
            is_compact=not args.no_compact,
            is_recompute=not args.no_recompute,
            num_threads=1,  # Force single-threaded mode
        )

        # Add texts in batches for better progress tracking
        batch_size = 1000
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for item in batch:
                # Handle both dict format (from create_text_chunks) and plain strings
                if isinstance(item, dict):
                    text = item.get("text", "")
                    metadata = item.get("metadata")
                    builder.add_text(text, metadata)
                else:
                    builder.add_text(item)
            print(f"Added {min(i + batch_size, len(texts))}/{len(texts)} texts...")

        print("Building index structure...")
        builder.build_index(index_path)
        print(f"Index saved to: {index_path}")

        # Register project directory so leann list can discover this index
        # The index is saved as args.index_dir/index_name.leann
        # We want to register the current working directory where the app is run
        register_project_directory(Path.cwd())

        return index_path

    async def run_interactive_chat(self, args, index_path: str):
        """Run interactive chat with the index."""
        chat = LeannChat(
            index_path,
            llm_config=self.get_llm_config(args),
            system_prompt=f"You are a helpful assistant that answers questions about {self.name} data.",
            complexity=args.search_complexity,
        )

        # Create interactive session
        session = create_rag_session(
            app_name=self.name.lower().replace(" ", "_"), data_description=self.name
        )

        def handle_query(query: str):
            # Prepare LLM kwargs with thinking budget if specified
            llm_kwargs = {}
            if hasattr(args, "thinking_budget") and args.thinking_budget:
                llm_kwargs["thinking_budget"] = args.thinking_budget

            response = chat.ask(
                query,
                top_k=args.top_k,
                complexity=args.search_complexity,
                llm_kwargs=llm_kwargs,
            )
            print(f"\nAssistant: {response}\n")

        session.run_interactive_loop(handle_query)

    async def run_single_query(self, args, index_path: str, query: str):
        """Run a single query against the index."""
        chat = LeannChat(
            index_path,
            llm_config=self.get_llm_config(args),
            complexity=args.search_complexity,
        )

        print(f"\n[Query]: \033[36m{query}\033[0m")

        # Prepare LLM kwargs with thinking budget if specified
        llm_kwargs = {}
        if hasattr(args, "thinking_budget") and args.thinking_budget:
            llm_kwargs["thinking_budget"] = args.thinking_budget

        response = chat.ask(
            query, top_k=args.top_k, complexity=args.search_complexity, llm_kwargs=llm_kwargs
        )
        print(f"\n[Response]: \033[36m{response}\033[0m")

    async def run(self):
        """Main entry point for the example."""
        args = self.parser.parse_args()

        # Check if index exists
        index_path = str(Path(args.index_dir) / f"{self.default_index_name}.leann")
        index_exists = Path(f"{index_path}.meta.json").exists()

        if not index_exists or args.force_rebuild:
            # Load data and build index
            print(f"\n{'Rebuilding' if index_exists else 'Building'} index...")
            texts = await self.load_data(args)

            if not texts:
                print("No data found to index!")
                return

            index_path = await self.build_index(args, texts)
        else:
            print(f"\nUsing existing index in {args.index_dir}")

        # Run query or interactive mode
        if args.query:
            await self.run_single_query(args, index_path, args.query)
        else:
            await self.run_interactive_chat(args, index_path)
