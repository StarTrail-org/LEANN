"""
Tests for CLI argument integration of --embedding-prompt-template.

These tests verify that:
1. The --embedding-prompt-template flag is properly registered on build and search commands
2. The template value flows from CLI args to embedding_options dict
3. The template is passed through to compute_embeddings() function
4. Default behavior (no flag) is handled correctly
"""

from unittest.mock import Mock, patch

from leann.cli import LeannCLI


class TestCLIPromptTemplateArgument:
    """Tests for --embedding-prompt-template on build and search commands."""

    def test_commands_accept_prompt_template_argument(self):
        """Verify that build and search parsers accept --embedding-prompt-template flag."""
        cli = LeannCLI()
        parser = cli.create_parser()
        template_value = "search_query: "

        # Test build command
        build_args = parser.parse_args(
            [
                "build",
                "test-index",
                "--docs",
                "/tmp/test-docs",
                "--embedding-prompt-template",
                template_value,
            ]
        )
        assert build_args.command == "build"
        assert hasattr(build_args, "embedding_prompt_template"), (
            "build command should have embedding_prompt_template attribute"
        )
        assert build_args.embedding_prompt_template == template_value

        # Test search command
        search_args = parser.parse_args(
            ["search", "test-index", "my query", "--embedding-prompt-template", template_value]
        )
        assert search_args.command == "search"
        assert hasattr(search_args, "embedding_prompt_template"), (
            "search command should have embedding_prompt_template attribute"
        )
        assert search_args.embedding_prompt_template == template_value

    def test_commands_default_to_none(self):
        """Verify default value is None when flag not provided (backward compatibility)."""
        cli = LeannCLI()
        parser = cli.create_parser()

        # Test build command default
        build_args = parser.parse_args(["build", "test-index", "--docs", "/tmp/test-docs"])
        assert hasattr(build_args, "embedding_prompt_template"), (
            "build command should have embedding_prompt_template attribute"
        )
        assert build_args.embedding_prompt_template is None, (
            "Build default value should be None when flag not provided"
        )

        # Test search command default
        search_args = parser.parse_args(["search", "test-index", "my query"])
        assert hasattr(search_args, "embedding_prompt_template"), (
            "search command should have embedding_prompt_template attribute"
        )
        assert search_args.embedding_prompt_template is None, (
            "Search default value should be None when flag not provided"
        )


class TestBuildCommandPromptTemplateArgumentExtras:
    """Additional build-specific tests for prompt template argument."""

    def test_build_command_prompt_template_with_multiword_value(self):
        """
        Verify that template values with spaces are handled correctly.

        Templates like "search_document: " or "Represent this sentence for searching: "
        should be accepted as a single string argument.
        """
        cli = LeannCLI()
        parser = cli.create_parser()

        template = "Represent this sentence for searching: "
        args = parser.parse_args(
            [
                "build",
                "test-index",
                "--docs",
                "/tmp/test-docs",
                "--embedding-prompt-template",
                template,
            ]
        )

        assert args.embedding_prompt_template == template


class TestPromptTemplateStoredInEmbeddingOptions:
    """Tests for template storage in embedding_options dict."""

    @patch("leann.cli.LeannBuilder")
    def test_prompt_template_stored_in_embedding_options_on_build(
        self, mock_builder_class, tmp_path
    ):
        """
        Verify that when --embedding-prompt-template is provided to build command,
        the value is stored in embedding_options dict passed to LeannBuilder.

        This test will fail because the CLI doesn't currently process this argument
        and add it to embedding_options.
        """
        # Setup mocks
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        # Create CLI and run build command
        cli = LeannCLI()

        # Mock load_documents to return a document so builder is created
        cli.load_documents = Mock(return_value=[{"text": "test content", "metadata": {}}])  # type: ignore[assignment]

        parser = cli.create_parser()

        template = "search_query: "
        args = parser.parse_args(
            [
                "build",
                "test-index",
                "--docs",
                str(tmp_path),
                "--embedding-prompt-template",
                template,
                "--force",  # Force rebuild to ensure LeannBuilder is called
            ]
        )

        # Run the build command
        import asyncio

        asyncio.run(cli.build_index(args))

        # Check that LeannBuilder was called with embedding_options containing prompt_template
        call_kwargs = mock_builder_class.call_args.kwargs
        assert "embedding_options" in call_kwargs, "LeannBuilder should receive embedding_options"

        embedding_options = call_kwargs["embedding_options"]
        assert embedding_options is not None, (
            "embedding_options should not be None when template provided"
        )
        assert "prompt_template" in embedding_options, (
            "embedding_options should contain 'prompt_template' key"
        )
        assert embedding_options["prompt_template"] == template, (
            f"Template should be '{template}', got {embedding_options.get('prompt_template')}"
        )

    @patch("leann.cli.LeannBuilder")
    def test_prompt_template_not_in_options_when_not_provided(self, mock_builder_class, tmp_path):
        """
        Verify that when --embedding-prompt-template is NOT provided,
        embedding_options either doesn't have the key or it's None.

        This ensures we don't pass empty/None values unnecessarily.
        """
        # Setup mocks
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        cli = LeannCLI()

        # Mock load_documents to return a document so builder is created
        cli.load_documents = Mock(return_value=[{"text": "test content", "metadata": {}}])  # type: ignore[assignment]

        parser = cli.create_parser()

        args = parser.parse_args(
            [
                "build",
                "test-index",
                "--docs",
                str(tmp_path),
                "--force",  # Force rebuild to ensure LeannBuilder is called
            ]
        )

        import asyncio

        asyncio.run(cli.build_index(args))

        # Check that if embedding_options is passed, it doesn't have prompt_template
        call_kwargs = mock_builder_class.call_args.kwargs
        if call_kwargs.get("embedding_options"):
            embedding_options = call_kwargs["embedding_options"]
            # Either the key shouldn't exist, or it should be None
            assert (
                "prompt_template" not in embedding_options
                or embedding_options["prompt_template"] is None
            ), "prompt_template should not be set when flag not provided"

    # R1 Tests: Build-time separate template storage
    @patch("leann.cli.LeannBuilder")
    def test_build_stores_separate_templates(self, mock_builder_class, tmp_path):
        """
        R1 Test 1: Verify that when both --embedding-prompt-template and
        --query-prompt-template are provided to build command, both values
        are stored separately in embedding_options dict as build_prompt_template
        and query_prompt_template.

        This test will fail because:
        1. CLI doesn't accept --query-prompt-template flag yet
        2. CLI doesn't store templates as separate build_prompt_template and
           query_prompt_template keys

        Expected behavior after implementation:
        - .meta.json contains: {"embedding_options": {
            "build_prompt_template": "doc: ",
            "query_prompt_template": "query: "
          }}
        """
        # Setup mocks
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        cli = LeannCLI()

        # Mock load_documents to return a document so builder is created
        cli.load_documents = Mock(return_value=[{"text": "test content", "metadata": {}}])  # type: ignore[assignment]

        parser = cli.create_parser()

        build_template = "doc: "
        query_template = "query: "
        args = parser.parse_args(
            [
                "build",
                "test-index",
                "--docs",
                str(tmp_path),
                "--embedding-prompt-template",
                build_template,
                "--query-prompt-template",
                query_template,
                "--force",
            ]
        )

        # Run the build command
        import asyncio

        asyncio.run(cli.build_index(args))

        # Check that LeannBuilder was called with separate template keys
        call_kwargs = mock_builder_class.call_args.kwargs
        assert "embedding_options" in call_kwargs, "LeannBuilder should receive embedding_options"

        embedding_options = call_kwargs["embedding_options"]
        assert embedding_options is not None, (
            "embedding_options should not be None when templates provided"
        )

        assert "build_prompt_template" in embedding_options, (
            "embedding_options should contain 'build_prompt_template' key"
        )
        assert embedding_options["build_prompt_template"] == build_template, (
            f"build_prompt_template should be '{build_template}'"
        )

        assert "query_prompt_template" in embedding_options, (
            "embedding_options should contain 'query_prompt_template' key"
        )
        assert embedding_options["query_prompt_template"] == query_template, (
            f"query_prompt_template should be '{query_template}'"
        )

        # Old key should NOT be present when using new separate template format
        assert "prompt_template" not in embedding_options, (
            "Old 'prompt_template' key should not be present with separate templates"
        )

    @patch("leann.cli.LeannBuilder")
    def test_build_backward_compat_single_template(self, mock_builder_class, tmp_path):
        """
        R1 Test 2: Verify backward compatibility - when only
        --embedding-prompt-template is provided (old behavior), it should
        still be stored as 'prompt_template' in embedding_options.

        This ensures existing workflows continue to work unchanged.

        This test currently passes because it matches existing behavior, but it
        documents the requirement that this behavior must be preserved after
        implementing the separate template feature.

        Expected behavior:
        - .meta.json contains: {"embedding_options": {"prompt_template": "prompt: "}}
        - No build_prompt_template or query_prompt_template keys
        """
        # Setup mocks
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        cli = LeannCLI()

        # Mock load_documents to return a document so builder is created
        cli.load_documents = Mock(return_value=[{"text": "test content", "metadata": {}}])  # type: ignore[assignment]

        parser = cli.create_parser()

        template = "prompt: "
        args = parser.parse_args(
            [
                "build",
                "test-index",
                "--docs",
                str(tmp_path),
                "--embedding-prompt-template",
                template,
                "--force",
            ]
        )

        # Run the build command
        import asyncio

        asyncio.run(cli.build_index(args))

        # Check that LeannBuilder was called with old format
        call_kwargs = mock_builder_class.call_args.kwargs
        assert "embedding_options" in call_kwargs, "LeannBuilder should receive embedding_options"

        embedding_options = call_kwargs["embedding_options"]
        assert embedding_options is not None, (
            "embedding_options should not be None when template provided"
        )

        assert "prompt_template" in embedding_options, (
            "embedding_options should contain old 'prompt_template' key for backward compat"
        )
        assert embedding_options["prompt_template"] == template, (
            f"prompt_template should be '{template}'"
        )

        # New keys should NOT be present in backward compat mode
        assert "build_prompt_template" not in embedding_options, (
            "build_prompt_template should not be present with single template flag"
        )
        assert "query_prompt_template" not in embedding_options, (
            "query_prompt_template should not be present with single template flag"
        )

    @patch("leann.cli.LeannBuilder")
    def test_build_no_templates(self, mock_builder_class, tmp_path):
        """
        R1 Test 3: Verify that when no template flags are provided,
        embedding_options has no prompt template keys.

        This ensures clean defaults and no unnecessary keys in .meta.json.

        This test currently passes because it matches existing behavior, but it
        documents the requirement that this behavior must be preserved after
        implementing the separate template feature.

        Expected behavior:
        - .meta.json has no prompt_template, build_prompt_template, or
          query_prompt_template keys (or embedding_options is empty/None)
        """
        # Setup mocks
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        cli = LeannCLI()

        # Mock load_documents to return a document so builder is created
        cli.load_documents = Mock(return_value=[{"text": "test content", "metadata": {}}])  # type: ignore[assignment]

        parser = cli.create_parser()

        args = parser.parse_args(["build", "test-index", "--docs", str(tmp_path), "--force"])

        # Run the build command
        import asyncio

        asyncio.run(cli.build_index(args))

        # Check that no template keys are present
        call_kwargs = mock_builder_class.call_args.kwargs
        if call_kwargs.get("embedding_options"):
            embedding_options = call_kwargs["embedding_options"]

            # None of the template keys should be present
            assert "prompt_template" not in embedding_options, (
                "prompt_template should not be present when no flags provided"
            )
            assert "build_prompt_template" not in embedding_options, (
                "build_prompt_template should not be present when no flags provided"
            )
            assert "query_prompt_template" not in embedding_options, (
                "query_prompt_template should not be present when no flags provided"
            )


class TestPromptTemplateFlowsToComputeEmbeddings:
    """Tests for template flowing through to compute_embeddings function."""

    @patch("leann.api.compute_embeddings")
    def test_prompt_template_flows_to_compute_embeddings_via_provider_options(
        self, mock_compute_embeddings, tmp_path
    ):
        """
        Verify that the prompt template flows from CLI args through LeannBuilder
        to compute_embeddings() function via provider_options parameter.

        This is an integration test that verifies the complete flow:
        CLI → embedding_options → LeannBuilder → compute_embeddings(provider_options)

        Native backend graph construction is mocked: this test only asserts the
        embedding provider_options wiring. Building a real HNSW graph with the
        mocked 3-D embeddings is outside the scope of the prompt-template contract
        and has been observed to abort the Linux CI process with exit 127.
        """
        # Mock compute_embeddings to return dummy embeddings as numpy array
        import numpy as np

        mock_compute_embeddings.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        # Use real LeannBuilder (not mocked) to test the actual embedding_options flow.
        # Stub only the native backend build + optional BM25 side effects.
        cli = LeannCLI()

        # Mock load_documents to return a simple document
        cli.load_documents = Mock(return_value=[{"text": "test content", "metadata": {}}])  # type: ignore[assignment]

        parser = cli.create_parser()

        template = "search_document: "
        args = parser.parse_args(
            [
                "build",
                "test-index",
                "--docs",
                str(tmp_path),
                "--embedding-prompt-template",
                template,
                "--backend-name",
                "hnsw",  # Use hnsw backend
                "--force",  # Force rebuild to ensure index is created
            ]
        )

        import asyncio
        from unittest.mock import MagicMock

        mock_builder_instance = MagicMock()
        mock_factory = MagicMock()
        mock_factory.builder = MagicMock(return_value=mock_builder_instance)

        # Avoid native HNSW graph build / BM25 side-effects; this test only
        # asserts CLI → LeannBuilder → compute_embeddings(provider_options).
        cli.register_project_dir = Mock()  # type: ignore[method-assign]
        with patch.dict("leann.api.BACKEND_REGISTRY", {"hnsw": mock_factory}, clear=False):
            with patch("leann.api.Fts5BM25Index") as mock_bm25_cls:
                mock_bm25_cls.return_value = MagicMock()
                asyncio.run(cli.build_index(args))

        # Verify compute_embeddings was called with provider_options containing prompt_template
        assert mock_compute_embeddings.called, "compute_embeddings should have been called"

        # Prefer kwargs; fall back to inspecting all calls (positional vs keyword)
        provider_options = None
        for call in mock_compute_embeddings.call_args_list:
            if call.kwargs.get("provider_options") is not None:
                provider_options = call.kwargs["provider_options"]
                break
            # provider_options may be passed positionally depending on call site
            if len(call.args) >= 5 and isinstance(call.args[4], dict):
                provider_options = call.args[4]
                break

        assert provider_options is not None, (
            "compute_embeddings should receive provider_options parameter"
        )
        assert "prompt_template" in provider_options, (
            "provider_options should contain prompt_template key"
        )
        assert provider_options["prompt_template"] == template, (
            f"Template should be '{template}', got {provider_options.get('prompt_template')}"
        )
        # Native build should have been invoked with the mocked embeddings
        mock_builder_instance.build.assert_called()


class TestPromptTemplateArgumentHelp:
    """Tests for argument help text and documentation."""

    def test_build_command_prompt_template_has_help_text(self):
        """
        Verify that --embedding-prompt-template has descriptive help text.

        Good help text is crucial for CLI usability.
        """
        cli = LeannCLI()
        parser = cli.create_parser()

        # Get the build subparser
        # This is a bit tricky - we need to parse to get the help
        # We'll check that the help includes relevant keywords
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        try:
            with redirect_stdout(f):
                parser.parse_args(["build", "--help"])
        except SystemExit:
            pass  # --help causes sys.exit(0)

        help_text = f.getvalue()
        assert "--embedding-prompt-template" in help_text, (
            "Help text should mention --embedding-prompt-template"
        )
        # Check for keywords that should be in the help
        help_lower = help_text.lower()
        assert any(keyword in help_lower for keyword in ["template", "prompt", "prepend"]), (
            "Help text should explain what the prompt template does"
        )

    def test_search_command_prompt_template_has_help_text(self):
        """
        Verify that search command also has help text for --embedding-prompt-template.
        """
        cli = LeannCLI()
        parser = cli.create_parser()

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        try:
            with redirect_stdout(f):
                parser.parse_args(["search", "--help"])
        except SystemExit:
            pass  # --help causes sys.exit(0)

        help_text = f.getvalue()
        assert "--embedding-prompt-template" in help_text, (
            "Search help text should mention --embedding-prompt-template"
        )
