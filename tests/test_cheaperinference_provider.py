"""
Tests for Cheaper Inference provider integration.

These tests validate Cheaper Inference provider settings, chat class, and
factory integration without triggering LEANN's compiled backend imports.
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
    resolve_cheaperinference_api_key,
    resolve_cheaperinference_base_url,
)


class TestCheaperInferenceSettings:
    """Test Cheaper Inference settings resolver functions."""

    def test_resolve_cheaperinference_api_key_explicit(self):
        assert resolve_cheaperinference_api_key("test-key") == "test-key"

    def test_resolve_cheaperinference_api_key_from_env(self):
        with patch.dict(os.environ, {"CHEAPER_INFERENCE_API_KEY": "ci-key"}, clear=True):
            assert resolve_cheaperinference_api_key() == "ci-key"

    def test_resolve_cheaperinference_api_key_does_not_fallback_to_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            assert resolve_cheaperinference_api_key() is None

    def test_resolve_cheaperinference_base_url_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_cheaperinference_base_url() == "https://api.cheaperinference.com/v1"

    def test_resolve_cheaperinference_base_url_explicit(self):
        assert resolve_cheaperinference_base_url("https://custom.url/v1") == "https://custom.url/v1"

    def test_resolve_cheaperinference_base_url_env_precedence(self):
        with patch.dict(
            os.environ,
            {
                "LEANN_CHEAPER_INFERENCE_BASE_URL": "https://leann.url/v1",
                "CHEAPER_INFERENCE_BASE_URL": "https://fallback.url/v1",
            },
            clear=True,
        ):
            assert resolve_cheaperinference_base_url() == "https://leann.url/v1"

    def test_resolve_cheaperinference_base_url_strips_trailing_slash(self):
        assert (
            resolve_cheaperinference_base_url("https://api.cheaperinference.com/v1/")
            == "https://api.cheaperinference.com/v1"
        )


class TestCheaperInferenceChat:
    """Test CheaperInferenceChat class."""

    def test_init_requires_api_key(self):
        from leann.chat import CheaperInferenceChat

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Cheaper Inference API key is required"):
                CheaperInferenceChat(api_key=None)

    @patch("openai.OpenAI")
    def test_init_with_api_key(self, mock_openai_cls):
        from leann.chat import CheaperInferenceChat

        chat = CheaperInferenceChat(api_key="test-key")
        assert chat.model == "gpt-5.4-mini"
        assert chat.api_key == "test-key"
        assert chat.base_url == "https://api.cheaperinference.com/v1"
        mock_openai_cls.assert_called_once_with(
            api_key="test-key", base_url="https://api.cheaperinference.com/v1"
        )

    @patch("openai.OpenAI")
    def test_init_custom_model(self, mock_openai_cls):
        from leann.chat import CheaperInferenceChat

        chat = CheaperInferenceChat(model="deepseek-v4-flash", api_key="test-key")
        assert chat.model == "deepseek-v4-flash"

    @patch("openai.OpenAI")
    def test_init_custom_base_url(self, mock_openai_cls):
        from leann.chat import CheaperInferenceChat

        chat = CheaperInferenceChat(api_key="test-key", base_url="https://custom.ci.url/v1")
        assert chat.base_url == "https://custom.ci.url/v1"

    @patch("openai.OpenAI")
    def test_ask_returns_response(self, mock_openai_cls):
        from leann.chat import CheaperInferenceChat

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from Cheaper Inference!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 100
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response

        chat = CheaperInferenceChat(api_key="test-key")
        result = chat.ask("Hello")

        assert result == "Hello from Cheaper Inference!"
        mock_client.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_ask_with_kwargs(self, mock_openai_cls):
        from leann.chat import CheaperInferenceChat

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

        chat = CheaperInferenceChat(api_key="test-key")
        chat.ask("Hello", temperature=0.5, max_tokens=500, top_p=0.9)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["top_p"] == 0.9

    @patch("openai.OpenAI")
    def test_ask_handles_error(self, mock_openai_cls):
        from leann.chat import CheaperInferenceChat

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        chat = CheaperInferenceChat(api_key="test-key")
        result = chat.ask("Hello")

        assert "Error" in result
        assert "Cheaper Inference" in result


class TestGetLLMFactory:
    """Test get_llm factory function with Cheaper Inference types."""

    @patch("openai.OpenAI")
    def test_get_llm_cheaperinference(self, mock_openai_cls):
        from leann.chat import CheaperInferenceChat, get_llm

        llm = get_llm({"type": "cheaperinference", "api_key": "test-key"})
        assert isinstance(llm, CheaperInferenceChat)
        assert llm.model == "gpt-5.4-mini"

    @patch("openai.OpenAI")
    @pytest.mark.parametrize("provider_type", ["cheaperinference", "cheaper-inference"])
    def test_get_llm_cheaperinference_aliases(self, mock_openai_cls, provider_type):
        from leann.chat import CheaperInferenceChat, get_llm

        llm = get_llm({"type": provider_type, "api_key": "test-key"})
        assert isinstance(llm, CheaperInferenceChat)

    @patch("openai.OpenAI")
    def test_get_llm_cheaperinference_custom_model(self, mock_openai_cls):
        from leann.chat import CheaperInferenceChat, get_llm

        llm = get_llm(
            {
                "type": "cheaperinference",
                "model": "deepseek-v4-flash",
                "api_key": "test-key",
            }
        )
        assert isinstance(llm, CheaperInferenceChat)
        assert llm.model == "deepseek-v4-flash"

    @patch("openai.OpenAI")
    def test_get_llm_cheaperinference_custom_base_url(self, mock_openai_cls):
        from leann.chat import CheaperInferenceChat, get_llm

        llm = get_llm(
            {
                "type": "cheaperinference",
                "api_key": "test-key",
                "base_url": "https://custom.ci.url/v1",
            }
        )
        assert isinstance(llm, CheaperInferenceChat)
        assert llm.base_url == "https://custom.ci.url/v1"
