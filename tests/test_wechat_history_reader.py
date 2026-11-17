import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest


def _stub_module(name: str) -> Any:
    """A stand-in module object whose attributes are set dynamically below."""
    return types.ModuleType(name)


try:
    from llama_index.core import Document as _Document
except ModuleNotFoundError:
    llama_index = _stub_module("llama_index")
    llama_index_core = _stub_module("llama_index.core")
    llama_index_readers = _stub_module("llama_index.core.readers")
    llama_index_base = _stub_module("llama_index.core.readers.base")

    class _Document:
        def __init__(self, text, metadata):
            self.text = text
            self.metadata = metadata

    class _BaseReader:
        pass

    llama_index_core.Document = _Document
    llama_index_base.BaseReader = _BaseReader
    sys.modules["llama_index"] = llama_index
    sys.modules["llama_index.core"] = llama_index_core
    sys.modules["llama_index.core.readers"] = llama_index_readers
    sys.modules["llama_index.core.readers.base"] = llama_index_base

readers_path = Path(__file__).parents[1] / "packages/leann-core/src/leann/readers.py"
readers_spec = importlib.util.spec_from_file_location("leann_readers", readers_path)
assert readers_spec is not None and readers_spec.loader is not None
readers = importlib.util.module_from_spec(readers_spec)
readers_spec.loader.exec_module(readers)
WeChatHistoryReader = readers.WeChatHistoryReader


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Reminder: buy milk tomorrow", "Reminder: buy milk tomorrow"),
        ("10:30 meeting at noon", "10:30 meeting at noon"),
        ("Q: is this covered?", "Q: is this covered?"),
        ("Please note: the meeting moved", "Please note: the meeting moved"),
        # A genuine wxid_ sender prefix is still stripped.
        ("wxid_abc123: hello there", "hello there"),
        ("Note:", "Note:"),
    ],
)
def test_extract_readable_text_keeps_colons_in_message_body(raw, expected):
    assert WeChatHistoryReader()._extract_readable_text(raw) == expected


def test_is_text_message_keeps_short_colon_message():
    # "Note:" used to be stripped to "" and the message dropped entirely.
    assert WeChatHistoryReader()._is_text_message("Note:") is True
    assert WeChatHistoryReader()._is_text_message("Hi:") is True


def test_load_data_keeps_colons_in_message_body(tmp_path):
    export_dir = tmp_path / "wechat-export"
    export_dir.mkdir()
    (export_dir / "Alice.json").write_text(
        json.dumps(
            [
                {
                    "content": "Reminder: buy milk tomorrow",
                    "createTime": 1763000000,
                    "isSentFromSelf": False,
                },
                {
                    "content": "10:30 meeting at noon",
                    "createTime": 1763000060,
                    "isSentFromSelf": True,
                },
                {
                    "content": "wxid_abc123: hello there",
                    "createTime": 1763000120,
                    "isSentFromSelf": False,
                },
                {"content": "Note:", "createTime": 1763000180, "isSentFromSelf": False},
            ]
        ),
        encoding="utf-8",
    )

    docs = WeChatHistoryReader().load_data(str(export_dir))

    assert len(docs) == 1
    text = docs[0].text
    assert "Reminder: buy milk tomorrow" in text
    assert "10:30 meeting at noon" in text
    assert "hello there" in text
    assert "Note:" in text
