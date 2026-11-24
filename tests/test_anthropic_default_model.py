"""
Tests for the default model of the Anthropic provider.

These tests build the LLM through ``get_llm`` without triggering LEANN's compiled
backend imports, and without any network access.
"""

import os
import sys
from unittest.mock import patch

import pytest

_LEANN_SRC = os.path.join(os.path.dirname(__file__), "..", "packages", "leann-core", "src")
if _LEANN_SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_LEANN_SRC))

if "leann" not in sys.modules:
    import types

    _stub = types.ModuleType("leann")
    _stub.__path__ = [os.path.join(os.path.abspath(_LEANN_SRC), "leann")]
    sys.modules["leann"] = _stub

pytest.importorskip("anthropic")


class TestAnthropicDefaultModel:
    @patch("anthropic.Anthropic")
    def test_factory_default_matches_the_chat_class_default(self, mock_anthropic_cls):
        from leann.chat import AnthropicChat, get_llm

        llm = get_llm({"type": "anthropic", "api_key": "test-key"})

        assert isinstance(llm, AnthropicChat)
        assert llm.model == AnthropicChat(api_key="test-key").model
        assert "claude-3" not in llm.model

    @patch("anthropic.Anthropic")
    def test_factory_keeps_an_explicit_model(self, mock_anthropic_cls):
        from leann.chat import AnthropicChat, get_llm

        llm = get_llm({"type": "anthropic", "model": "claude-sonnet-5-5", "api_key": "test-key"})

        assert isinstance(llm, AnthropicChat)
        assert llm.model == "claude-sonnet-5-5"


class TestAnthropicAskThinkingBlock:
    @patch("anthropic.Anthropic")
    def test_ask_skips_leading_thinking_block(self, mock_anthropic_cls):
        from types import SimpleNamespace

        from leann.chat import AnthropicChat

        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="let me think"),
                SimpleNamespace(type="text", text=" the answer "),
            ],
            usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            stop_reason="end_turn",
        )
        mock_anthropic_cls.return_value.messages.create.return_value = response

        llm = AnthropicChat(model="claude-sonnet-5-5", api_key="test-key")

        assert llm.ask("hi") == "the answer"
