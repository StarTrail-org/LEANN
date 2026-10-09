"""
Integration tests for prompt template metadata persistence and reuse.

These tests verify the complete lifecycle of prompt template persistence:
1. Template is saved to .meta.json during index build
2. Template is automatically loaded during search operations
3. Template can be overridden with explicit flag during search
4. Template is reused during chat/ask operations

These are integration tests that:
- Use real file system with temporary directories
- Run actual build and search operations
- Inspect .meta.json file contents directly
- Mock embedding servers to avoid external dependencies
- Use small test codebases for fast execution

Expected to FAIL in Red Phase because metadata persistence verification is not yet implemented.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from leann.api import LeannBuilder, LeannSearcher


class TestPromptTemplateMetadataPersistence:
    """Tests for prompt template storage in .meta.json during build."""

    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary directory for test indexes."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_embeddings(self):
        """Mock compute_embeddings to return dummy embeddings."""
        with patch("leann.api.compute_embeddings") as mock_compute:
            # Return dummy embeddings as numpy array
            mock_compute.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
            yield mock_compute

    def test_prompt_template_saved_to_metadata(self, temp_index_dir, mock_embeddings):
        """
        Verify that when build is run with embedding_options containing prompt_template,
        the template value is saved to .meta.json file.

        This is the core persistence requirement - templates must be saved to allow
        reuse in subsequent search operations without re-specifying the flag.

        Expected failure: .meta.json exists but doesn't contain embedding_options
        with prompt_template, or the value is not persisted correctly.
        """
        # Setup test data
        index_path = temp_index_dir / "test_index.leann"
        template = "search_document: "

        # Build index with prompt template in embedding_options
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="text-embedding-3-small",
            embedding_mode="openai",
            embedding_options={"prompt_template": template},
        )

        # Add a simple document
        builder.add_text("This is a test document for indexing")

        # Build the index
        builder.build_index(str(index_path))

        # Verify .meta.json was created and contains the template
        meta_path = temp_index_dir / "test_index.leann.meta.json"
        assert meta_path.exists(), ".meta.json file should be created during build"

        # Read and parse metadata
        with open(meta_path, encoding="utf-8") as f:
            meta_data = json.load(f)

        # Verify embedding_options exists in metadata
        assert "embedding_options" in meta_data, (
            "embedding_options should be saved to .meta.json when provided"
        )

        # Verify prompt_template is in embedding_options
        embedding_options = meta_data["embedding_options"]
        assert "prompt_template" in embedding_options, (
            "prompt_template should be saved within embedding_options"
        )

        # Verify the template value matches what we provided
        assert embedding_options["prompt_template"] == template, (
            f"Template should be '{template}', got '{embedding_options.get('prompt_template')}'"
        )

    def test_prompt_template_absent_when_not_provided(self, temp_index_dir, mock_embeddings):
        """
        Verify that when no prompt template is provided during build,
        .meta.json either doesn't have embedding_options or prompt_template key.

        This ensures clean metadata without unnecessary keys when features aren't used.

        Expected behavior: Build succeeds, .meta.json doesn't contain prompt_template.
        """
        index_path = temp_index_dir / "test_no_template.leann"

        # Build index WITHOUT prompt template
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="text-embedding-3-small",
            embedding_mode="openai",
            # No embedding_options provided
        )

        builder.add_text("Document without template")
        builder.build_index(str(index_path))

        # Verify metadata
        meta_path = temp_index_dir / "test_no_template.leann.meta.json"
        assert meta_path.exists()

        with open(meta_path, encoding="utf-8") as f:
            meta_data = json.load(f)

        # If embedding_options exists, it should not contain prompt_template
        if "embedding_options" in meta_data:
            embedding_options = meta_data["embedding_options"]
            assert "prompt_template" not in embedding_options, (
                "prompt_template should not be in metadata when not provided"
            )


class TestPromptTemplateAutoLoadOnSearch:
    """Tests for automatic loading of prompt template during search operations.

    NOTE: Over-mocked test removed (test_prompt_template_auto_loaded_on_search).
    This functionality is now comprehensively tested by TestQueryPromptTemplateAutoLoad
    which uses simpler mocking and doesn't hang.
    """

    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary directory for test indexes."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_embeddings(self):
        """Mock compute_embeddings to capture calls and return dummy embeddings."""
        with patch("leann.api.compute_embeddings") as mock_compute:
            mock_compute.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
            yield mock_compute

    def test_search_without_template_in_metadata(self, temp_index_dir, mock_embeddings):
        """
        Verify that searching an index built WITHOUT a prompt template
        works correctly (backward compatibility).

        The searcher should handle missing prompt_template gracefully.

        Expected behavior: Search succeeds, no template is used.
        """
        # Build index without template
        index_path = temp_index_dir / "no_template.leann"

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="text-embedding-3-small",
            embedding_mode="openai",
        )
        builder.add_text("Document without template")
        builder.build_index(str(index_path))

        # Reset mocks
        mock_embeddings.reset_mock()

        # Create searcher and search
        searcher = LeannSearcher(index_path=str(index_path))

        # Verify no template in embedding_options
        assert "prompt_template" not in searcher.embedding_options, (
            "Searcher should not have prompt_template when not in metadata"
        )


class TestQueryPromptTemplateAutoLoad:
    """Tests for automatic loading of separate query_prompt_template during search (R2).

    These tests verify the new two-template system where:
    - build_prompt_template: Applied during index building
    - query_prompt_template: Applied during search operations

    Expected to FAIL in Red Phase (R2) because query template extraction
    and application is not yet implemented in LeannSearcher.search().
    """

    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary directory for test indexes."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_compute_embeddings(self):
        """Mock compute_embeddings to capture calls and return dummy embeddings."""
        with patch("leann.embedding_compute.compute_embeddings") as mock_compute:
            mock_compute.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
            yield mock_compute

    def test_search_auto_loads_query_template(self, temp_index_dir, mock_compute_embeddings):
        """
        Verify that search() automatically loads and applies query_prompt_template from .meta.json.

        Given: Index built with separate build_prompt_template and query_prompt_template
        When: LeannSearcher.search("my query") is called
        Then: Query embedding is computed with "query: my query" (query template applied)

        This is the core R2 requirement - query templates must be auto-loaded and applied
        during search without user intervention.

        Expected failure: compute_embeddings called with raw "my query" instead of
        "query: my query" because query template extraction is not implemented.
        """
        # Setup: Build index with separate templates in new format
        index_path = temp_index_dir / "query_template.leann"

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="text-embedding-3-small",
            embedding_mode="openai",
            embedding_options={
                "build_prompt_template": "doc: ",
                "query_prompt_template": "query: ",
            },
        )
        builder.add_text("Test document")
        builder.build_index(str(index_path))

        # Reset mock to ignore build calls
        mock_compute_embeddings.reset_mock()

        # Act: Search with query
        searcher = LeannSearcher(index_path=str(index_path))

        # Mock the backend search to avoid actual search
        with patch.object(searcher.backend_impl, "search") as mock_backend_search:
            mock_backend_search.return_value = {
                "labels": [["test_id_0"]],  # IDs (nested list for batch support)
                "distances": [[0.9]],  # Distances (nested list for batch support)
            }

            searcher.search("my query", top_k=1, recompute_embeddings=False)

        # Assert: compute_embeddings was called with query template applied
        assert mock_compute_embeddings.called, "compute_embeddings should be called during search"

        # Get the actual text passed to compute_embeddings
        call_args = mock_compute_embeddings.call_args
        texts_arg = call_args[0][0]  # First positional arg (list of texts)

        assert len(texts_arg) == 1, "Should compute embedding for one query"
        assert texts_arg[0] == "query: my query", (
            f"Query template should be applied: expected 'query: my query', got '{texts_arg[0]}'"
        )

    def test_search_backward_compat_single_template(self, temp_index_dir, mock_compute_embeddings):
        """
        Verify backward compatibility with old single prompt_template format.

        Given: Index with old format (single prompt_template, no query_prompt_template)
        When: LeannSearcher.search("my query") is called
        Then: Query embedding computed with "doc: my query" (old template applied)

        This ensures indexes built with the old single-template system continue
        to work correctly with the new search implementation.

        Expected failure: Old template not recognized/applied because backward
        compatibility logic is not implemented.
        """
        # Setup: Build index with old single-template format
        index_path = temp_index_dir / "old_template.leann"

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="text-embedding-3-small",
            embedding_mode="openai",
            embedding_options={"prompt_template": "doc: "},  # Old format
        )
        builder.add_text("Test document")
        builder.build_index(str(index_path))

        # Reset mock
        mock_compute_embeddings.reset_mock()

        # Act: Search
        searcher = LeannSearcher(index_path=str(index_path))

        with patch.object(searcher.backend_impl, "search") as mock_backend_search:
            mock_backend_search.return_value = {"labels": [["test_id_0"]], "distances": [[0.9]]}

            searcher.search("my query", top_k=1, recompute_embeddings=False)

        # Assert: Old template was applied
        call_args = mock_compute_embeddings.call_args
        texts_arg = call_args[0][0]

        assert texts_arg[0] == "doc: my query", (
            f"Old prompt_template should be applied for backward compatibility: "
            f"expected 'doc: my query', got '{texts_arg[0]}'"
        )

    def test_search_backward_compat_no_template(self, temp_index_dir, mock_compute_embeddings):
        """
        Verify backward compatibility when no template is present in .meta.json.

        Given: Index with no template in .meta.json (very old indexes)
        When: LeannSearcher.search("my query") is called
        Then: Query embedding computed with "my query" (no template, raw query)

        This ensures the most basic backward compatibility - indexes without
        any template support continue to work as before.

        Expected failure: May fail if default template is incorrectly applied,
        or if missing template causes error.
        """
        # Setup: Build index without any template
        index_path = temp_index_dir / "no_template.leann"

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="text-embedding-3-small",
            embedding_mode="openai",
            # No embedding_options at all
        )
        builder.add_text("Test document")
        builder.build_index(str(index_path))

        # Reset mock
        mock_compute_embeddings.reset_mock()

        # Act: Search
        searcher = LeannSearcher(index_path=str(index_path))

        with patch.object(searcher.backend_impl, "search") as mock_backend_search:
            mock_backend_search.return_value = {"labels": [["test_id_0"]], "distances": [[0.9]]}

            searcher.search("my query", top_k=1, recompute_embeddings=False)

        # Assert: No template applied (raw query)
        call_args = mock_compute_embeddings.call_args
        texts_arg = call_args[0][0]

        assert texts_arg[0] == "my query", (
            f"No template should be applied when missing from metadata: "
            f"expected 'my query', got '{texts_arg[0]}'"
        )

    def test_search_override_via_provider_options(self, temp_index_dir, mock_compute_embeddings):
        """
        Verify that explicit provider_options can override stored query template.

        Given: Index with query_prompt_template: "query: "
        When: search() called with provider_options={"prompt_template": "override: "}
        Then: Query embedding computed with "override: test" (override takes precedence)

        This enables users to experiment with different query templates without
        rebuilding the index, or to handle special query types differently.

        Expected failure: provider_options parameter is accepted via **kwargs but
        not used. Query embedding computed with raw "test" instead of "override: test"
        because override logic is not implemented.
        """
        # Setup: Build index with query template
        index_path = temp_index_dir / "override_template.leann"

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="text-embedding-3-small",
            embedding_mode="openai",
            embedding_options={
                "build_prompt_template": "doc: ",
                "query_prompt_template": "query: ",
            },
        )
        builder.add_text("Test document")
        builder.build_index(str(index_path))

        # Reset mock
        mock_compute_embeddings.reset_mock()

        # Act: Search with override
        searcher = LeannSearcher(index_path=str(index_path))

        with patch.object(searcher.backend_impl, "search") as mock_backend_search:
            mock_backend_search.return_value = {"labels": [["test_id_0"]], "distances": [[0.9]]}

            # This should accept provider_options parameter
            searcher.search(
                "test",
                top_k=1,
                recompute_embeddings=False,
                provider_options={"prompt_template": "override: "},
            )

        # Assert: Override template was applied
        call_args = mock_compute_embeddings.call_args
        texts_arg = call_args[0][0]

        assert texts_arg[0] == "override: test", (
            f"Override template should take precedence: "
            f"expected 'override: test', got '{texts_arg[0]}'"
        )


class TestPromptTemplateReuseInChat:
    """Tests for prompt template reuse in chat/ask operations."""

    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary directory for test indexes."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_embeddings(self):
        """Mock compute_embeddings to return dummy embeddings."""
        with patch("leann.api.compute_embeddings") as mock_compute:
            mock_compute.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
            yield mock_compute

    @pytest.fixture
    def mock_embedding_server_manager(self):
        """Mock EmbeddingServerManager for chat tests."""
        with patch("leann.searcher_base.EmbeddingServerManager") as mock_manager_class:
            mock_manager = Mock()
            mock_manager.start_server.return_value = (True, 5557)
            mock_manager_class.return_value = mock_manager
            yield mock_manager

    @pytest.fixture
    def index_with_template(self, temp_index_dir, mock_embeddings):
        """Build an index with a prompt template."""
        index_path = temp_index_dir / "chat_template_index.leann"
        template = "document_query: "

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="text-embedding-3-small",
            embedding_mode="openai",
            embedding_options={"prompt_template": template},
        )

        builder.add_text("Test document for chat")
        builder.build_index(str(index_path))

        return str(index_path), template


class TestPromptTemplateIntegrationWithEmbeddingModes:
    """Tests for prompt template compatibility with different embedding modes."""

    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary directory for test indexes."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.parametrize(
        "mode,model,template,filename_prefix",
        [
            (
                "openai",
                "text-embedding-3-small",
                "Represent this for searching: ",
                "openai_template",
            ),
            ("ollama", "nomic-embed-text", "search_query: ", "ollama_template"),
            ("sentence-transformers", "facebook/contriever", "query: ", "st_template"),
        ],
    )
    def test_prompt_template_metadata_with_embedding_modes(
        self, temp_index_dir, mode, model, template, filename_prefix
    ):
        """Verify prompt template is saved correctly across different embedding modes.

        Tests that prompt templates are persisted to .meta.json for:
        - OpenAI mode (primary use case)
        - Ollama mode (also supports templates)
        - Sentence-transformers mode (saved for forward compatibility)

        Expected behavior: Template is saved to .meta.json regardless of mode.
        """
        with patch("leann.api.compute_embeddings") as mock_compute:
            mock_compute.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

            index_path = temp_index_dir / f"{filename_prefix}.leann"

            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model=model,
                embedding_mode=mode,
                embedding_options={"prompt_template": template},
            )

            builder.add_text(f"{mode.capitalize()} test document")
            builder.build_index(str(index_path))

            # Verify metadata
            meta_path = temp_index_dir / f"{filename_prefix}.leann.meta.json"
            with open(meta_path, encoding="utf-8") as f:
                meta_data = json.load(f)

            assert meta_data["embedding_mode"] == mode
            # Template should be saved for all modes (even if not used by some)
            if "embedding_options" in meta_data:
                assert meta_data["embedding_options"]["prompt_template"] == template


class TestQueryTemplateApplicationInComputeEmbedding:
    """Tests for query template application in compute_query_embedding() (Bug Fix).

    These tests verify that query templates are applied consistently in BOTH
    code paths (server and fallback) when computing query embeddings.

    This addresses the bug where query templates were only applied in the
    fallback path, not when using the embedding server (the default path).

    Bug Context:
    - Issue: Query templates were stored in metadata but only applied during
      fallback (direct) computation, not when using embedding server
    - Fix: Move template application to BEFORE any computation path in
      compute_query_embedding() (searcher_base.py:107-110)
    - Impact: Critical for models like EmbeddingGemma that require task-specific
      templates for optimal performance

    These tests ensure the fix works correctly and prevent regression.
    """

    @pytest.fixture
    def temp_index_with_template(self):
        """Create a temporary index with query template in metadata"""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            index_dir = Path(tmpdir)
            index_file = index_dir / "test.leann"
            meta_file = index_dir / "test.leann.meta.json"

            # Create minimal metadata with query template
            metadata = {
                "version": "1.0",
                "backend_name": "hnsw",
                "embedding_model": "text-embedding-embeddinggemma-300m-qat",
                "dimensions": 768,
                "embedding_mode": "openai",
                "backend_kwargs": {
                    "graph_degree": 32,
                    "complexity": 64,
                    "distance_metric": "cosine",
                },
                "embedding_options": {
                    "base_url": "http://localhost:1234/v1",
                    "api_key": "test-key",
                    "build_prompt_template": "title: none | text: ",
                    "query_prompt_template": "task: search result | query: ",
                },
            }

            meta_file.write_text(json.dumps(metadata, indent=2))

            # Create minimal HNSW index file (empty is okay for this test)
            index_file.write_bytes(b"")

            yield str(index_file)

    def test_query_template_applied_in_fallback_path(self, temp_index_with_template):
        """Test that query template is applied when using fallback (direct) path"""
        from leann.searcher_base import BaseSearcher

        # Create a concrete implementation for testing
        class TestSearcher(BaseSearcher):
            def search(
                self,
                query,
                top_k,
                complexity=64,
                beam_width=1,
                prune_ratio=0.0,
                recompute_embeddings=False,
                pruning_strategy="global",
                zmq_port=None,
                **kwargs,
            ):
                return {"labels": [], "distances": []}

        searcher = object.__new__(TestSearcher)
        searcher.index_path = Path(temp_index_with_template)
        searcher.index_dir = searcher.index_path.parent

        # Load metadata
        meta_file = searcher.index_dir / f"{searcher.index_path.name}.meta.json"
        with open(meta_file) as f:
            searcher.meta = json.load(f)

        searcher.embedding_model = searcher.meta["embedding_model"]
        searcher.embedding_mode = searcher.meta.get("embedding_mode", "sentence-transformers")
        searcher.embedding_options = searcher.meta.get("embedding_options", {})
        searcher.enable_warmup = False
        searcher.use_daemon = False
        searcher.daemon_ttl_seconds = 0

        # Initialize query cache for tests
        searcher.query_cache = Mock()
        searcher.query_cache.get.return_value = None

        # Mock compute_embeddings to capture the query text
        captured_queries = []

        def mock_compute_embeddings(texts, model, mode, provider_options=None):
            captured_queries.extend(texts)
            return np.random.rand(len(texts), 768).astype(np.float32)

        with patch(
            "leann.embedding_compute.compute_embeddings", side_effect=mock_compute_embeddings
        ):
            # Call compute_query_embedding with template (fallback path)
            result = searcher.compute_query_embedding(
                query="vector database",
                use_server_if_available=False,  # Force fallback path
                query_template="task: search result | query: ",
            )

        # Verify template was applied
        assert len(captured_queries) == 1
        assert captured_queries[0] == "task: search result | query: vector database"
        assert result.shape == (1, 768)

    def test_query_template_applied_in_server_path(self, temp_index_with_template):
        """Test that query template is applied when using server path"""
        from leann.searcher_base import BaseSearcher

        # Create a concrete implementation for testing
        class TestSearcher(BaseSearcher):
            def search(
                self,
                query,
                top_k,
                complexity=64,
                beam_width=1,
                prune_ratio=0.0,
                recompute_embeddings=False,
                pruning_strategy="global",
                zmq_port=None,
                **kwargs,
            ):
                return {"labels": [], "distances": []}

        searcher = object.__new__(TestSearcher)
        searcher.index_path = Path(temp_index_with_template)
        searcher.index_dir = searcher.index_path.parent

        # Load metadata
        meta_file = searcher.index_dir / f"{searcher.index_path.name}.meta.json"
        with open(meta_file) as f:
            searcher.meta = json.load(f)

        searcher.embedding_model = searcher.meta["embedding_model"]
        searcher.embedding_mode = searcher.meta.get("embedding_mode", "sentence-transformers")
        searcher.embedding_options = searcher.meta.get("embedding_options", {})
        searcher.enable_warmup = False
        searcher.use_daemon = False
        searcher.daemon_ttl_seconds = 0

        # Initialize query cache for tests
        searcher.query_cache = Mock()
        searcher.query_cache.get.return_value = None

        # Mock the server methods to capture the query text
        captured_queries = []

        def mock_ensure_server_running(passages_file, port, **kwargs):
            return port

        def mock_compute_embedding_via_server(chunks, port):
            captured_queries.extend(chunks)
            return np.random.rand(len(chunks), 768).astype(np.float32)

        searcher._ensure_server_running = mock_ensure_server_running
        searcher._compute_embedding_via_server = mock_compute_embedding_via_server

        # Call compute_query_embedding with template (server path)
        result = searcher.compute_query_embedding(
            query="vector database",
            use_server_if_available=True,  # Use server path
            query_template="task: search result | query: ",
        )

        # Verify template was applied BEFORE calling server
        assert len(captured_queries) == 1
        assert captured_queries[0] == "task: search result | query: vector database"
        assert result.shape == (1, 768)

    def test_query_template_without_template_parameter(self, temp_index_with_template):
        """Test that query is unchanged when no template is provided"""
        from leann.searcher_base import BaseSearcher

        class TestSearcher(BaseSearcher):
            def search(
                self,
                query,
                top_k,
                complexity=64,
                beam_width=1,
                prune_ratio=0.0,
                recompute_embeddings=False,
                pruning_strategy="global",
                zmq_port=None,
                **kwargs,
            ):
                return {"labels": [], "distances": []}

        searcher = object.__new__(TestSearcher)
        searcher.index_path = Path(temp_index_with_template)
        searcher.index_dir = searcher.index_path.parent

        meta_file = searcher.index_dir / f"{searcher.index_path.name}.meta.json"
        with open(meta_file) as f:
            searcher.meta = json.load(f)

        searcher.embedding_model = searcher.meta["embedding_model"]
        searcher.embedding_mode = searcher.meta.get("embedding_mode", "sentence-transformers")
        searcher.embedding_options = searcher.meta.get("embedding_options", {})
        searcher.enable_warmup = False
        searcher.use_daemon = False
        searcher.daemon_ttl_seconds = 0

        # Initialize query cache for tests
        searcher.query_cache = Mock()
        searcher.query_cache.get.return_value = None

        captured_queries = []

        def mock_compute_embeddings(texts, model, mode, provider_options=None):
            captured_queries.extend(texts)
            return np.random.rand(len(texts), 768).astype(np.float32)

        with patch(
            "leann.embedding_compute.compute_embeddings", side_effect=mock_compute_embeddings
        ):
            searcher.compute_query_embedding(
                query="vector database",
                use_server_if_available=False,
                query_template=None,  # No template
            )

        # Verify query is unchanged
        assert len(captured_queries) == 1
        assert captured_queries[0] == "vector database"

    def test_query_template_consistency_between_paths(self, temp_index_with_template):
        """Test that both paths apply template identically"""
        from leann.searcher_base import BaseSearcher

        class TestSearcher(BaseSearcher):
            def search(
                self,
                query,
                top_k,
                complexity=64,
                beam_width=1,
                prune_ratio=0.0,
                recompute_embeddings=False,
                pruning_strategy="global",
                zmq_port=None,
                **kwargs,
            ):
                return {"labels": [], "distances": []}

        searcher = object.__new__(TestSearcher)
        searcher.index_path = Path(temp_index_with_template)
        searcher.index_dir = searcher.index_path.parent

        meta_file = searcher.index_dir / f"{searcher.index_path.name}.meta.json"
        with open(meta_file) as f:
            searcher.meta = json.load(f)

        searcher.embedding_model = searcher.meta["embedding_model"]
        searcher.embedding_mode = searcher.meta.get("embedding_mode", "sentence-transformers")
        searcher.embedding_options = searcher.meta.get("embedding_options", {})
        searcher.enable_warmup = False
        searcher.use_daemon = False
        searcher.daemon_ttl_seconds = 0

        # Initialize query cache for tests
        searcher.query_cache = Mock()
        searcher.query_cache.get.return_value = None

        query_template = "task: search result | query: "
        original_query = "vector database"

        # Capture queries from fallback path
        fallback_queries = []

        def mock_compute_embeddings(texts, model, mode, provider_options=None):
            fallback_queries.extend(texts)
            return np.random.rand(len(texts), 768).astype(np.float32)

        with patch(
            "leann.embedding_compute.compute_embeddings", side_effect=mock_compute_embeddings
        ):
            searcher.compute_query_embedding(
                query=original_query,
                use_server_if_available=False,
                query_template=query_template,
            )

        # Capture queries from server path
        server_queries = []

        def mock_ensure_server_running(passages_file, port, **kwargs):
            return port

        def mock_compute_embedding_via_server(chunks, port):
            server_queries.extend(chunks)
            return np.random.rand(len(chunks), 768).astype(np.float32)

        searcher._ensure_server_running = mock_ensure_server_running
        searcher._compute_embedding_via_server = mock_compute_embedding_via_server

        searcher.compute_query_embedding(
            query=original_query,
            use_server_if_available=True,
            query_template=query_template,
        )

        # Verify both paths produced identical templated queries
        assert len(fallback_queries) == 1
        assert len(server_queries) == 1
        assert fallback_queries[0] == server_queries[0]
        assert fallback_queries[0] == f"{query_template}{original_query}"

    def test_query_template_with_empty_string(self, temp_index_with_template):
        """Test behavior with empty template string"""
        from leann.searcher_base import BaseSearcher

        class TestSearcher(BaseSearcher):
            def search(
                self,
                query,
                top_k,
                complexity=64,
                beam_width=1,
                prune_ratio=0.0,
                recompute_embeddings=False,
                pruning_strategy="global",
                zmq_port=None,
                **kwargs,
            ):
                return {"labels": [], "distances": []}

        searcher = object.__new__(TestSearcher)
        searcher.index_path = Path(temp_index_with_template)
        searcher.index_dir = searcher.index_path.parent

        meta_file = searcher.index_dir / f"{searcher.index_path.name}.meta.json"
        with open(meta_file) as f:
            searcher.meta = json.load(f)

        searcher.embedding_model = searcher.meta["embedding_model"]
        searcher.embedding_mode = searcher.meta.get("embedding_mode", "sentence-transformers")
        searcher.embedding_options = searcher.meta.get("embedding_options", {})
        searcher.enable_warmup = False
        searcher.use_daemon = False
        searcher.daemon_ttl_seconds = 0

        # Initialize query cache for tests
        searcher.query_cache = Mock()
        searcher.query_cache.get.return_value = None

        captured_queries = []

        def mock_compute_embeddings(texts, model, mode, provider_options=None):
            captured_queries.extend(texts)
            return np.random.rand(len(texts), 768).astype(np.float32)

        with patch(
            "leann.embedding_compute.compute_embeddings", side_effect=mock_compute_embeddings
        ):
            searcher.compute_query_embedding(
                query="vector database",
                use_server_if_available=False,
                query_template="",  # Empty string
            )

        # Empty string is falsy, so no template should be applied
        assert captured_queries[0] == "vector database"
