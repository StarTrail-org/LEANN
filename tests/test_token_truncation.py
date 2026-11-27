"""Unit tests for token-aware truncation functionality.

This test suite defines the contract for token truncation functions that prevent
500 errors from Ollama when text exceeds model token limits. These tests verify:

1. Model token limit retrieval (known and unknown models)
2. Text truncation behavior for single and multiple texts
3. Token counting and truncation accuracy using tiktoken

All tests are written in Red Phase - they should FAIL initially because the
implementation does not exist yet.
"""

import pytest
import tiktoken
from leann.embedding_compute import (
    EMBEDDING_MODEL_LIMITS,
    get_model_token_limit,
    truncate_to_token_limit,
)


class TestModelTokenLimits:
    """Tests for retrieving model-specific token limits."""

    def test_get_model_token_limit_known_model(self):
        """Verify correct token limit is returned for known models.

        Known models should return their specific token limits from
        EMBEDDING_MODEL_LIMITS dictionary.
        """
        # Test nomic-embed-text (2048 tokens)
        limit = get_model_token_limit("nomic-embed-text")
        assert limit == 2048, "nomic-embed-text should have 2048 token limit"

        # Test nomic-embed-text-v1.5 (2048 tokens)
        limit = get_model_token_limit("nomic-embed-text-v1.5")
        assert limit == 2048, "nomic-embed-text-v1.5 should have 2048 token limit"

        # Test nomic-embed-text-v2 (512 tokens)
        limit = get_model_token_limit("nomic-embed-text-v2")
        assert limit == 512, "nomic-embed-text-v2 should have 512 token limit"

        # Test OpenAI models (8192 tokens)
        limit = get_model_token_limit("text-embedding-3-small")
        assert limit == 8192, "text-embedding-3-small should have 8192 token limit"

    def test_get_model_token_limit_unknown_model(self):
        """Verify default token limit is returned for unknown models.

        Unknown models should return the default limit (2048) to allow
        operation with reasonable safety margin.
        """
        # Test with completely unknown model
        limit = get_model_token_limit("unknown-model-xyz")
        assert limit == 2048, "Unknown models should return default 2048"

        # Test with empty string
        limit = get_model_token_limit("")
        assert limit == 2048, "Empty model name should return default 2048"

    def test_get_model_token_limit_custom_default(self):
        """Verify custom default can be specified for unknown models.

        Allow callers to specify their own default token limit when
        model is not in the known models dictionary.
        """
        limit = get_model_token_limit("unknown-model", default=4096)
        assert limit == 4096, "Should return custom default for unknown models"

        # Known model should ignore custom default
        limit = get_model_token_limit("nomic-embed-text", default=4096)
        assert limit == 2048, "Known model should ignore custom default"

    def test_embedding_model_limits_dictionary_exists(self):
        """Verify EMBEDDING_MODEL_LIMITS dictionary contains expected models.

        The dictionary should be importable and contain at least the
        known nomic models with correct token limits.
        """
        assert isinstance(EMBEDDING_MODEL_LIMITS, dict), "Should be a dictionary"
        assert "nomic-embed-text" in EMBEDDING_MODEL_LIMITS, "Should contain nomic-embed-text"
        assert "nomic-embed-text-v1.5" in EMBEDDING_MODEL_LIMITS, (
            "Should contain nomic-embed-text-v1.5"
        )
        assert EMBEDDING_MODEL_LIMITS["nomic-embed-text"] == 2048
        assert EMBEDDING_MODEL_LIMITS["nomic-embed-text-v1.5"] == 2048
        assert EMBEDDING_MODEL_LIMITS["nomic-embed-text-v2"] == 512
        # OpenAI models
        assert EMBEDDING_MODEL_LIMITS["text-embedding-3-small"] == 8192


class TestTokenTruncation:
    """Tests for truncating texts to token limits."""

    @pytest.fixture
    def tokenizer(self):
        """Provide tiktoken tokenizer for token counting verification."""
        return tiktoken.get_encoding("cl100k_base")

    def test_truncate_single_text_under_limit(self, tokenizer):
        """Verify text under token limit remains unchanged.

        When text is already within the token limit, it should be
        returned unchanged with no truncation.
        """
        text = "This is a short text that is well under the token limit."
        token_count = len(tokenizer.encode(text))
        assert token_count < 100, f"Test setup: text should be short (has {token_count} tokens)"

        # Truncate with generous limit
        result = truncate_to_token_limit([text], token_limit=512)

        assert len(result) == 1, "Should return same number of texts"
        assert result[0] == text, "Text under limit should be unchanged"

    def test_truncate_single_text_over_limit(self, tokenizer):
        """Verify text over token limit is truncated correctly.

        When text exceeds the token limit, it should be truncated to
        fit within the limit while maintaining valid token boundaries.
        """
        # Create a text that definitely exceeds limit
        text = "word " * 200  # ~200 tokens (each "word " is typically 1-2 tokens)
        original_token_count = len(tokenizer.encode(text))
        assert original_token_count > 50, (
            f"Test setup: text should be long (has {original_token_count} tokens)"
        )

        # Truncate to 50 tokens
        result = truncate_to_token_limit([text], token_limit=50)

        assert len(result) == 1, "Should return same number of texts"
        assert result[0] != text, "Text over limit should be truncated"
        assert len(result[0]) < len(text), "Truncated text should be shorter"

        # Verify truncated text is within token limit
        truncated_token_count = len(tokenizer.encode(result[0]))
        assert truncated_token_count <= 50, (
            f"Truncated text should be ≤50 tokens, got {truncated_token_count}"
        )

    def test_truncate_multiple_texts_mixed_lengths(self, tokenizer):
        """Verify multiple texts with mixed lengths are handled correctly.

        When processing multiple texts:
        - Texts under limit should remain unchanged
        - Texts over limit should be truncated independently
        - Output list should maintain same order and length
        """
        texts = [
            "Short text.",  # Under limit
            "word " * 200,  # Over limit
            "Another short one.",  # Under limit
            "token " * 150,  # Over limit
        ]

        # Verify test setup
        for i, text in enumerate(texts):
            token_count = len(tokenizer.encode(text))
            if i in [1, 3]:
                assert token_count > 50, f"Text {i} should be over limit (has {token_count} tokens)"
            else:
                assert token_count < 50, (
                    f"Text {i} should be under limit (has {token_count} tokens)"
                )

        # Truncate with 50 token limit
        result = truncate_to_token_limit(texts, token_limit=50)

        assert len(result) == len(texts), "Should return same number of texts"

        # Verify each text individually
        for i, (original, truncated) in enumerate(zip(texts, result)):
            token_count = len(tokenizer.encode(truncated))
            assert token_count <= 50, f"Text {i} should be ≤50 tokens, got {token_count}"

            # Short texts should be unchanged
            if i in [0, 2]:
                assert truncated == original, f"Short text {i} should be unchanged"
            # Long texts should be truncated
            else:
                assert len(truncated) < len(original), f"Long text {i} should be truncated"

    def test_truncate_empty_list(self):
        """Verify empty input list returns empty output list.

        Edge case: empty list should return empty list without errors.
        """
        result = truncate_to_token_limit([], token_limit=512)
        assert result == [], "Empty input should return empty output"

    def test_truncate_preserves_order(self, tokenizer):
        """Verify truncation preserves original text order.

        Output list should maintain the same order as input list,
        regardless of which texts were truncated.
        """
        texts = [
            "First text " * 50,  # Will be truncated
            "Second text.",  # Won't be truncated
            "Third text " * 50,  # Will be truncated
        ]

        result = truncate_to_token_limit(texts, token_limit=20)

        assert len(result) == 3, "Should preserve list length"
        # Check that order is maintained by looking for distinctive words
        assert "First" in result[0], "First text should remain in first position"
        assert "Second" in result[1], "Second text should remain in second position"
        assert "Third" in result[2], "Third text should remain in third position"

    def test_truncate_extremely_long_text(self, tokenizer):
        """Verify extremely long texts are truncated efficiently.

        Test with text that far exceeds token limit to ensure
        truncation handles extreme cases without performance issues.
        """
        # Create very long text (simulate real-world scenario)
        text = "token " * 5000  # ~5000+ tokens
        original_token_count = len(tokenizer.encode(text))
        assert original_token_count > 1000, "Test setup: text should be very long"

        # Truncate to small limit
        result = truncate_to_token_limit([text], token_limit=100)

        assert len(result) == 1
        truncated_token_count = len(tokenizer.encode(result[0]))
        assert truncated_token_count <= 100, (
            f"Should truncate to ≤100 tokens, got {truncated_token_count}"
        )
        assert len(result[0]) < len(text) // 10, "Should significantly reduce text length"

    def test_truncate_exact_token_limit(self, tokenizer):
        """Verify text at exactly token limit is handled correctly.

        Edge case: text with exactly the token limit should either
        remain unchanged or be safely truncated by 1 token.
        """
        # Create text with approximately 50 tokens
        # We'll adjust to get exactly 50
        target_tokens = 50
        text = "word " * 50
        tokens = tokenizer.encode(text)

        # Adjust to get exactly target_tokens
        if len(tokens) > target_tokens:
            tokens = tokens[:target_tokens]
            text = tokenizer.decode(tokens)
        elif len(tokens) < target_tokens:
            # Add more words
            while len(tokenizer.encode(text)) < target_tokens:
                text += "word "
            tokens = tokenizer.encode(text)[:target_tokens]
            text = tokenizer.decode(tokens)

        # Verify we have exactly target_tokens
        assert len(tokenizer.encode(text)) == target_tokens, (
            "Test setup: should have exactly 50 tokens"
        )

        result = truncate_to_token_limit([text], token_limit=target_tokens)

        assert len(result) == 1
        result_tokens = len(tokenizer.encode(result[0]))
        assert result_tokens <= target_tokens, (
            f"Should be ≤{target_tokens} tokens, got {result_tokens}"
        )

    def test_truncate_literal_special_token_under_limit(self):
        """Verify a literal special-token spelling does not raise.

        A document containing the literal text "<|endoftext|>" (e.g. a
        tokenizer tutorial or pasted chat log) must be encoded as ordinary
        text, not rejected: tiktoken defaults to disallowed_special="all",
        which raises ValueError and would abort the whole embedding batch.
        """
        text = "<|endoftext|>"
        result = truncate_to_token_limit([text], token_limit=2048)

        assert len(result) == 1, "Should return same number of texts"
        assert result[0] == text, "Literal special-token text should be unchanged"

    def test_truncate_literal_special_token_over_limit(self, tokenizer):
        """Verify long text with literal special-token spellings truncates.

        The same spelling repeated past the token limit must truncate
        cleanly to the limit instead of raising ValueError.
        """
        text = "<|endoftext|> " * 20  # 122 tokens with disallowed_special=()
        original_token_count = len(tokenizer.encode(text, disallowed_special=()))
        assert original_token_count == 122, (
            f"Test setup: text should be 122 tokens (has {original_token_count})"
        )

        result = truncate_to_token_limit([text], token_limit=16)

        assert len(result) == 1, "Should return same number of texts"
        assert result[0] == "<|endoftext|> <|endoftext|> <|endoft", (
            "Should truncate to the first 16 tokens of literal text"
        )
        truncated_token_count = len(tokenizer.encode(result[0], disallowed_special=()))
        assert truncated_token_count == 16, (
            f"Truncated text should be 16 tokens, got {truncated_token_count}"
        )

    def test_truncate_mixed_text_with_literal_special_token(self):
        """Verify ordinary text mentioning a special token stays verbatim.

        A short sentence that merely mentions "<|endoftext|>" is within
        the limit and must be returned unchanged.
        """
        text = "A tokenizer example: <|endoftext|> is literal."
        result = truncate_to_token_limit([text], token_limit=2048)

        assert len(result) == 1, "Should return same number of texts"
        assert result[0] == text, "Mixed text with literal token should be unchanged"


class TestLMStudioHybridDiscovery:
    """Tests for LM Studio integration in get_model_token_limit() hybrid discovery.

    These tests verify that get_model_token_limit() properly integrates with
    the LM Studio SDK bridge for dynamic token limit discovery. The integration
    should:

    1. Detect LM Studio URLs (port 1234 or 'lmstudio'/'lm.studio' in URL)
    2. Convert HTTP URLs to WebSocket format for SDK queries
    3. Query LM Studio SDK and use discovered limit
    4. Fall back to registry when SDK returns None
    5. Execute AFTER Ollama detection but BEFORE registry fallback

    All tests are written in Red Phase - they should FAIL initially because the
    LM Studio detection and integration logic does not exist yet in get_model_token_limit().
    """

    def test_get_model_token_limit_lmstudio_success(self, monkeypatch):
        """Verify LM Studio SDK query succeeds and returns detected limit.

        When a LM Studio base_url is detected and the SDK query succeeds,
        get_model_token_limit() should return the dynamically discovered
        context length without falling back to the registry.
        """

        # Mock _query_lmstudio_context_limit to return successful SDK query
        def mock_query_lmstudio(model_name, base_url):
            # Verify WebSocket URL was passed (not HTTP)
            assert base_url.startswith("ws://"), (
                f"Should convert HTTP to WebSocket format, got: {base_url}"
            )
            return 8192  # Successful SDK query

        monkeypatch.setattr(
            "leann.embedding_compute._query_lmstudio_context_limit",
            mock_query_lmstudio,
        )

        # Test with HTTP URL that should be converted to WebSocket
        limit = get_model_token_limit(
            model_name="custom-model", base_url="http://localhost:1234/v1"
        )

        assert limit == 8192, "Should return limit from LM Studio SDK query"

    def test_get_model_token_limit_lmstudio_fallback_to_registry(self, monkeypatch):
        """Verify fallback to registry when LM Studio SDK returns None.

        When LM Studio SDK query fails (returns None), get_model_token_limit()
        should fall back to the EMBEDDING_MODEL_LIMITS registry.
        """

        # Mock _query_lmstudio_context_limit to return None (SDK failure)
        def mock_query_lmstudio(model_name, base_url):
            return None  # SDK query failed

        monkeypatch.setattr(
            "leann.embedding_compute._query_lmstudio_context_limit",
            mock_query_lmstudio,
        )

        # Test with known model that exists in registry
        limit = get_model_token_limit(
            model_name="nomic-embed-text", base_url="http://localhost:1234/v1"
        )

        # Should fall back to registry value
        assert limit == 2048, "Should fall back to registry when SDK returns None"

    def test_get_model_token_limit_lmstudio_port_detection(self, monkeypatch):
        """Verify detection of LM Studio via port 1234.

        get_model_token_limit() should recognize port 1234 as a LM Studio
        server and attempt SDK query, regardless of hostname.
        """
        query_called = False

        def mock_query_lmstudio(model_name, base_url):
            nonlocal query_called
            query_called = True
            return 4096

        monkeypatch.setattr(
            "leann.embedding_compute._query_lmstudio_context_limit",
            mock_query_lmstudio,
        )

        # Test with port 1234 (default LM Studio port)
        limit = get_model_token_limit(model_name="test-model", base_url="http://127.0.0.1:1234/v1")

        assert query_called, "Should detect port 1234 and call LM Studio SDK query"
        assert limit == 4096, "Should return SDK query result"

    @pytest.mark.parametrize(
        "test_url,expected_limit,keyword",
        [
            ("http://lmstudio.local:8080/v1", 16384, "lmstudio"),
            ("http://api.lm.studio:5000/v1", 32768, "lm.studio"),
        ],
    )
    def test_get_model_token_limit_lmstudio_url_keyword_detection(
        self, monkeypatch, test_url, expected_limit, keyword
    ):
        """Verify detection of LM Studio via keywords in URL.

        get_model_token_limit() should recognize 'lmstudio' or 'lm.studio'
        in the URL as indicating a LM Studio server.
        """
        query_called = False

        def mock_query_lmstudio(model_name, base_url):
            nonlocal query_called
            query_called = True
            return expected_limit

        monkeypatch.setattr(
            "leann.embedding_compute._query_lmstudio_context_limit",
            mock_query_lmstudio,
        )

        limit = get_model_token_limit(model_name="test-model", base_url=test_url)

        assert query_called, f"Should detect '{keyword}' keyword and call SDK query"
        assert limit == expected_limit, f"Should return SDK query result for {keyword}"

    @pytest.mark.parametrize(
        "input_url,expected_protocol,expected_host",
        [
            ("http://localhost:1234/v1", "ws://", "localhost:1234"),
            ("https://lmstudio.example.com:1234/v1", "wss://", "lmstudio.example.com:1234"),
        ],
    )
    def test_get_model_token_limit_protocol_conversion(
        self, monkeypatch, input_url, expected_protocol, expected_host
    ):
        """Verify HTTP/HTTPS URL is converted to WebSocket format for SDK query.

        LM Studio SDK requires WebSocket URLs. get_model_token_limit() should:
        1. Convert 'http://' to 'ws://'
        2. Convert 'https://' to 'wss://'
        3. Remove '/v1' or other path suffixes (SDK expects base URL)
        4. Preserve host and port
        """
        conversions_tested = []

        def mock_query_lmstudio(model_name, base_url):
            conversions_tested.append(base_url)
            return 8192

        monkeypatch.setattr(
            "leann.embedding_compute._query_lmstudio_context_limit",
            mock_query_lmstudio,
        )

        get_model_token_limit(model_name="test-model", base_url=input_url)

        # Verify conversion happened
        assert len(conversions_tested) == 1, "Should have called SDK query once"
        assert conversions_tested[0].startswith(expected_protocol), (
            f"Should convert to {expected_protocol}"
        )
        assert expected_host in conversions_tested[0], (
            f"Should preserve host and port: {expected_host}"
        )

    def test_get_model_token_limit_lmstudio_executes_after_ollama(self, monkeypatch):
        """Verify LM Studio detection happens AFTER Ollama detection.

        The hybrid discovery order should be:
        1. Ollama dynamic discovery (port 11434 or 'ollama' in URL)
        2. LM Studio dynamic discovery (port 1234 or 'lmstudio' in URL)
        3. Registry fallback

        If both Ollama and LM Studio patterns match, Ollama should take precedence.
        This test verifies that LM Studio is checked but doesn't interfere with Ollama.
        """
        ollama_called = False
        lmstudio_called = False

        def mock_query_ollama(model_name, base_url):
            nonlocal ollama_called
            ollama_called = True
            return 2048  # Ollama query succeeds

        def mock_query_lmstudio(model_name, base_url):
            nonlocal lmstudio_called
            lmstudio_called = True
            return None  # Should not be reached if Ollama succeeds

        monkeypatch.setattr(
            "leann.embedding_compute._query_ollama_context_limit",
            mock_query_ollama,
        )
        monkeypatch.setattr(
            "leann.embedding_compute._query_lmstudio_context_limit",
            mock_query_lmstudio,
        )

        # Test with Ollama URL
        limit = get_model_token_limit(
            model_name="test-model", base_url="http://localhost:11434/api"
        )

        assert ollama_called, "Should attempt Ollama query first"
        assert not lmstudio_called, "Should not attempt LM Studio query when Ollama succeeds"
        assert limit == 2048, "Should return Ollama result"

    def test_get_model_token_limit_lmstudio_not_detected_for_non_lmstudio_urls(self, monkeypatch):
        """Verify LM Studio SDK query is NOT called for non-LM Studio URLs.

        Only URLs with port 1234 or 'lmstudio'/'lm.studio' keywords should
        trigger LM Studio SDK queries. Other URLs should skip to registry fallback.
        """
        lmstudio_called = False

        def mock_query_lmstudio(model_name, base_url):
            nonlocal lmstudio_called
            lmstudio_called = True
            return 8192

        monkeypatch.setattr(
            "leann.embedding_compute._query_lmstudio_context_limit",
            mock_query_lmstudio,
        )

        # Test with non-LM Studio URLs
        test_cases = [
            "http://localhost:8080/v1",  # Different port
            "http://openai.example.com/v1",  # Different service
            "http://localhost:3000/v1",  # Another port
        ]

        for base_url in test_cases:
            lmstudio_called = False  # Reset for each test
            get_model_token_limit(model_name="nomic-embed-text", base_url=base_url)
            assert not lmstudio_called, f"Should NOT call LM Studio SDK for URL: {base_url}"

    def test_get_model_token_limit_lmstudio_case_insensitive_detection(self, monkeypatch):
        """Verify LM Studio detection is case-insensitive for keywords.

        Keywords 'lmstudio' and 'lm.studio' should be detected regardless
        of case (LMStudio, LMSTUDIO, LmStudio, etc.).
        """
        query_called = False

        def mock_query_lmstudio(model_name, base_url):
            nonlocal query_called
            query_called = True
            return 8192

        monkeypatch.setattr(
            "leann.embedding_compute._query_lmstudio_context_limit",
            mock_query_lmstudio,
        )

        # Test various case variations
        test_cases = [
            "http://LMStudio.local:8080/v1",
            "http://LMSTUDIO.example.com/v1",
            "http://LmStudio.local/v1",
            "http://api.LM.STUDIO:5000/v1",
        ]

        for base_url in test_cases:
            query_called = False  # Reset for each test
            limit = get_model_token_limit(model_name="test-model", base_url=base_url)
            assert query_called, f"Should detect LM Studio in URL: {base_url}"
            assert limit == 8192, f"Should return SDK result for URL: {base_url}"


class TestTokenLimitCaching:
    """Tests for token limit caching to prevent repeated SDK/API calls.

    Caching prevents duplicate SDK/API calls within the same Python process,
    which is important because:
    1. LM Studio SDK load() can load duplicate model instances
    2. Ollama /api/show queries add latency
    3. Registry lookups are pure overhead

    Cache is process-scoped and resets between leann build invocations.
    """

    def setup_method(self):
        """Clear cache before each test."""
        from leann.embedding_compute import _token_limit_cache

        _token_limit_cache.clear()

    def test_registry_lookup_is_cached(self):
        """Verify that registry lookups are cached."""
        from leann.embedding_compute import _token_limit_cache

        # First call
        limit1 = get_model_token_limit("text-embedding-3-small")
        assert limit1 == 8192

        # Verify it's in cache
        cache_key = ("text-embedding-3-small", "")
        assert cache_key in _token_limit_cache
        assert _token_limit_cache[cache_key] == 8192

        # Second call should use cache
        limit2 = get_model_token_limit("text-embedding-3-small")
        assert limit2 == 8192

    def test_default_fallback_is_cached(self):
        """Verify that default fallbacks are cached."""
        from leann.embedding_compute import _token_limit_cache

        # First call with unknown model
        limit1 = get_model_token_limit("unknown-model-xyz", default=512)
        assert limit1 == 512

        # Verify it's in cache
        cache_key = ("unknown-model-xyz", "")
        assert cache_key in _token_limit_cache
        assert _token_limit_cache[cache_key] == 512

        # Second call should use cache
        limit2 = get_model_token_limit("unknown-model-xyz", default=512)
        assert limit2 == 512

    def test_different_urls_create_separate_cache_entries(self):
        """Verify that different base_urls create separate cache entries."""
        from leann.embedding_compute import _token_limit_cache

        # Same model, different URLs
        limit1 = get_model_token_limit("nomic-embed-text", base_url="http://localhost:11434")
        limit2 = get_model_token_limit("nomic-embed-text", base_url="http://localhost:1234/v1")

        # Both should find the model in registry (2048)
        assert limit1 == 2048
        assert limit2 == 2048

        # But they should be separate cache entries
        cache_key1 = ("nomic-embed-text", "http://localhost:11434")
        cache_key2 = ("nomic-embed-text", "http://localhost:1234/v1")

        assert cache_key1 in _token_limit_cache
        assert cache_key2 in _token_limit_cache
        assert len(_token_limit_cache) == 2

    def test_cache_prevents_repeated_lookups(self):
        """Verify that cache prevents repeated registry/API lookups."""
        from leann.embedding_compute import _token_limit_cache

        model_name = "text-embedding-ada-002"

        # First call - should add to cache
        assert len(_token_limit_cache) == 0
        limit1 = get_model_token_limit(model_name)

        cache_size_after_first = len(_token_limit_cache)
        assert cache_size_after_first == 1

        # Multiple subsequent calls - cache size should not change
        for _ in range(5):
            limit = get_model_token_limit(model_name)
            assert limit == limit1
            assert len(_token_limit_cache) == cache_size_after_first

    def test_versioned_model_names_cached_correctly(self):
        """Verify that versioned model names (e.g., model:tag) are cached."""
        from leann.embedding_compute import _token_limit_cache

        # Model with version tag
        limit = get_model_token_limit("nomic-embed-text:latest", base_url="http://localhost:11434")
        assert limit == 2048

        # Should be cached with full name including version
        cache_key = ("nomic-embed-text:latest", "http://localhost:11434")
        assert cache_key in _token_limit_cache
        assert _token_limit_cache[cache_key] == 2048
