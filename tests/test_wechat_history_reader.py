"""Regression tests for the WeChat reader that `leann index-wechat` actually runs.

`cli.py:index_wechat` imports `apps.history_data.wechat_history.WeChatHistoryReader`,
so that is the class under test here. The copy in
`packages/leann-core/src/leann/readers.py` has no importer.
"""

import json
import sys
from pathlib import Path

import pytest

# Add the parent directory to the path so we can import from apps
sys.path.append(str(Path(__file__).parent.parent))

from apps.history_data.wechat_history import WeChatHistoryReader


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

    docs = WeChatHistoryReader().load_data(
        wechat_export_dir=str(export_dir), concatenate_messages=True
    )

    assert len(docs) == 1
    text = docs[0].text
    assert "Reminder: buy milk tomorrow" in text
    assert "10:30 meeting at noon" in text
    assert "hello there" in text
    assert "Note:" in text
