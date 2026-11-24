"""Tests for resolve_ollama_host handling of Ollama's host:port OLLAMA_HOST form."""

import os
import sys
import types

import pytest

# Import leann.settings without triggering leann.__init__ (needs C++ backends).
_LEANN_SRC = os.path.join(os.path.dirname(__file__), "..", "packages", "leann-core", "src")
if "leann" not in sys.modules:
    _LEANN_SRC = os.path.abspath(_LEANN_SRC)
    _stub = types.ModuleType("leann")
    _stub.__path__ = [os.path.join(_LEANN_SRC, "leann")]
    sys.modules["leann"] = _stub

from leann.settings import resolve_ollama_host  # noqa: E402

_ENV_VARS = ("LEANN_LOCAL_LLM_HOST", "LEANN_OLLAMA_HOST", "OLLAMA_HOST", "LOCAL_LLM_ENDPOINT")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_default_is_unchanged():
    assert resolve_ollama_host() == "http://localhost:11434"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("0.0.0.0:11434", "http://0.0.0.0:11434"),
        ("localhost:11434/", "http://localhost:11434"),
        ("myhost", "http://myhost"),
        ("http://h:1/", "http://h:1"),
        ("https://ollama.example.com", "https://ollama.example.com"),
    ],
)
def test_ollama_host_env_gets_a_scheme(monkeypatch, value, expected):
    monkeypatch.setenv("OLLAMA_HOST", value)
    assert resolve_ollama_host() == expected


def test_explicit_host_without_scheme_gets_a_scheme():
    assert resolve_ollama_host("10.0.0.5:11434") == "http://10.0.0.5:11434"
