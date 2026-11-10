"""
Tests for UberLLM provider integration.

These tests validate UberLLM provider settings, chat class, and factory
integration without triggering LEANN's compiled backend imports.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_LEANN_SRC = os.path.join(os.path.dirname(__file__), "..", "packages", "leann-core", "src")
if _LEANN_SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_LEANN_SRC))

if "leann" not in sys.modules:
    import types

    _stub = types.ModuleType("leann")
    _stub.__path__ = [os.path.join(os.path.abspath(_LEANN_SRC), "leann")]
    sys.modules["leann"] = _stub

from leann.settings import (  # noqa: E402
    resolve_uberllm_api_key,
    resolve_uberllm_base_url,
)


class TestUberLLMSettings:
    """Test UberLLM settings resolver functions."""

    def test_resolve_uberllm_api_key_explicit(self):
        assert resolve_uberllm_api_key("test-key") == "test-key"

    def test_resolve_uberllm_api_key_from_env(self):
        with patch.dict(os.environ, {"UBERLLM_API_KEY": "uberllm-key"}, clear=True):
            assert resolve_uberllm_api_key() == "uberllm-key"

    def test_resolve_uberllm_api_key_does_not_fallback_to_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            assert resolve_uberllm_api_key() is None

    def test_resolve_uberllm_base_url_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_uberllm_base_url() == "https://api.uberllm.dev/v1"

    def test_resolve_uberllm_base_url_explicit(self):
        assert resolve_uberllm_base_url("https://custom.url/v1") == "https://custom.url/v1"

    def test_resolve_uberllm_base_url_env_precedence(self):
        with patch.dict(
            os.environ,
            {
                "LEANN_UBERLLM_BASE_URL": "https://leann.url/v1",
                "UBERLLM_BASE_URL": "https://fallback.url/v1",
            },
            clear=True,
        ):
            assert resolve_uberllm_base_url() == "https://leann.url/v1"

    def test_resolve_uberllm_base_url_strips_trailing_slash(self):
        assert (
            resolve_uberllm_base_url("https://api.uberllm.dev/v1/") == "https://api.uberllm.dev/v1"
        )


class TestUberLLMChat:
    """Test UberLLMChat class."""

    def test_init_requires_api_key(self):
        from leann.chat import UberLLMChat

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="UberLLM API key is required"):
                UberLLMChat(api_key=None)

    @patch("openai.OpenAI")
    def test_init_with_api_key(self, mock_openai_cls):
        from leann.chat import UberLLMChat

        with patch.dict(os.environ, {}, clear=True):
            chat = UberLLMChat(api_key="test-key")
        assert chat.model == "deepseek/deepseek-v4-pro"
        assert chat.api_key == "test-key"
        assert chat.base_url == "https://api.uberllm.dev/v1"
        mock_openai_cls.assert_called_once_with(
            api_key="test-key", base_url="https://api.uberllm.dev/v1"
        )

    @patch("openai.OpenAI")
    def test_init_custom_model(self, mock_openai_cls):
        from leann.chat import UberLLMChat

        chat = UberLLMChat(model="anthropic/claude-sonnet-4.6", api_key="test-key")
        assert chat.model == "anthropic/claude-sonnet-4.6"

    @patch("openai.OpenAI")
    def test_init_custom_base_url(self, mock_openai_cls):
        from leann.chat import UberLLMChat

        chat = UberLLMChat(api_key="test-key", base_url="https://custom.uberllm.url/v1")
        assert chat.base_url == "https://custom.uberllm.url/v1"

    @patch("openai.OpenAI")
    def test_ask_returns_response(self, mock_openai_cls):
        from leann.chat import UberLLMChat

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from UberLLM!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 100
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response

        chat = UberLLMChat(api_key="test-key")
        result = chat.ask("Hello")

        assert result == "Hello from UberLLM!"
        mock_client.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_ask_with_kwargs(self, mock_openai_cls):
        from leann.chat import UberLLMChat

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 50
        mock_response.usage.prompt_tokens = 25
        mock_response.usage.completion_tokens = 25
        mock_client.chat.completions.create.return_value = mock_response

        chat = UberLLMChat(api_key="test-key")
        chat.ask("Hello", temperature=0.5, max_tokens=500, top_p=0.9)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "deepseek/deepseek-v4-pro"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["top_p"] == 0.9

    @patch("openai.OpenAI")
    def test_ask_handles_error(self, mock_openai_cls):
        from leann.chat import UberLLMChat

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        chat = UberLLMChat(api_key="test-key")
        result = chat.ask("Hello")

        assert "Error" in result
        assert "UberLLM" in result


class TestGetLLMFactory:
    """Test get_llm factory function with the UberLLM type."""

    @patch("openai.OpenAI")
    def test_get_llm_uberllm(self, mock_openai_cls):
        from leann.chat import UberLLMChat, get_llm

        llm = get_llm({"type": "uberllm", "api_key": "test-key"})
        assert isinstance(llm, UberLLMChat)
        assert llm.model == "deepseek/deepseek-v4-pro"

    @patch("openai.OpenAI")
    def test_get_llm_uberllm_custom_model(self, mock_openai_cls):
        from leann.chat import UberLLMChat, get_llm

        llm = get_llm(
            {
                "type": "uberllm",
                "model": "deepseek/deepseek-v4.1-flash",
                "api_key": "test-key",
            }
        )
        assert isinstance(llm, UberLLMChat)
        assert llm.model == "deepseek/deepseek-v4.1-flash"

    @patch("openai.OpenAI")
    def test_get_llm_uberllm_custom_base_url(self, mock_openai_cls):
        from leann.chat import UberLLMChat, get_llm

        llm = get_llm(
            {
                "type": "uberllm",
                "api_key": "test-key",
                "base_url": "https://custom.uberllm.url/v1",
            }
        )
        assert isinstance(llm, UberLLMChat)
        assert llm.base_url == "https://custom.uberllm.url/v1"


@pytest.mark.skipif(
    not os.getenv("UBERLLM_API_KEY"),
    reason="UBERLLM_API_KEY not set; skipping live API test",
)
class TestUberLLMLiveAPI:
    """Live API tests for UberLLM provider."""

    def test_uberllm_deepseek_live(self):
        from leann.chat import UberLLMChat

        chat = UberLLMChat(model="deepseek/deepseek-v4.1-flash")
        response = chat.ask("Say hello in one word.", max_tokens=30)
        assert isinstance(response, str)
        assert len(response) > 0
        assert not response.startswith("Error:")

    def test_uberllm_via_get_llm_live(self):
        from leann.chat import get_llm

        llm = get_llm({"type": "uberllm", "model": "deepseek/deepseek-v4.1-flash"})
        response = llm.ask("What is 1+1? Reply with just the number.", max_tokens=30)
        assert isinstance(response, str)
        assert len(response) > 0
        assert not response.startswith("Error:")
