"""Exercise index command parsing and dispatch without accessing personal data."""

import asyncio
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, call

import pytest
from leann import cli as cli_module
from leann.cli import LeannCLI
from llama_index.core import Document

SOURCES = {
    "browser": ("apps.history_data.history", "ChromeHistoryReader", "browser_history"),
    "email": ("apps.email_data.LEANN_email_reader", "EmlxReader", "email"),
    "calendar": (None, None, "calendar"),
    "imessage": ("apps.imessage_data.imessage_reader", "IMessageReader", "imessage"),
    "wechat": ("apps.history_data.wechat_history", "WeChatHistoryReader", "wechat"),
    "chatgpt": ("apps.chatgpt_data.chatgpt_reader", "ChatGPTReader", "chatgpt"),
    "claude": ("apps.claude_data.claude_reader", "ClaudeReader", "claude"),
}


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        os.path,
        "expanduser",
        lambda path: str(tmp_path / path[2:]) if path.startswith("~/") else path,
    )
    return LeannCLI()


def command_args(source, tmp_path):
    argv = [f"index-{source}"]
    if source == "browser":
        argv.append("brave")
    elif source == "wechat":
        argv.extend(["--export-dir", str(tmp_path / "export")])
    elif source in {"chatgpt", "claude"}:
        argv.extend(["--export-path", str(tmp_path / "export.json")])
    return argv


@pytest.mark.parametrize("source", SOURCES)
def test_index_command_defaults(cli, source, tmp_path):
    args = cli.create_parser().parse_args(command_args(source, tmp_path))
    assert args.index_name == SOURCES[source][2]
    assert args.max_count == 1000
    assert not args.no_recompute


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("empty", [False, True], ids=["documents", "empty"])
@pytest.mark.parametrize("no_recompute", [False, True], ids=["recompute", "stored"])
def test_index_command_dispatch(cli, source, empty, no_recompute, tmp_path, monkeypatch):
    # Install fake modules before dispatch: no application reader is imported or run.
    documents = [] if empty else [Document(text="Synthetic text", metadata={"source": source})]
    module_name, reader_name, _ = SOURCES[source]
    reader_class = Mock()
    reader_class.return_value.load_data.return_value = documents
    if module_name:
        assert reader_name is not None
        parts = module_name.split(".")
        for end in range(1, len(parts) + 1):
            name = ".".join(parts[:end])
            module = ModuleType(name)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, name, module)
        module = sys.modules[module_name]
        setattr(module, reader_name, reader_class)
        if source == "email":
            monkeypatch.setattr(
                module,
                "find_all_messages_directories",
                Mock(return_value=[tmp_path / "mail"]),
                raising=False,
            )
    else:
        # The legacy calendar handler embeds its reader. Exercise its actual SQL
        # against a synthetic database, redirecting its fixed scratch path.
        calendar_cache = tmp_path / "Library" / "Calendars" / "Calendar Cache"
        calendar_cache.parent.mkdir(parents=True)
        connection = sqlite3.connect(calendar_cache)
        connection.execute(
            "CREATE TABLE CI_EVENT (summary, description, location, start_date, end_date)"
        )
        if not empty:
            connection.executemany(
                "INSERT INTO CI_EVENT VALUES (?, ?, ?, ?, ?)",
                [(f"Synthetic event {i}", "Details", "Room", i, i + 1) for i in range(3)],
            )
        connection.commit()
        monkeypatch.setattr(shutil, "copy2", Mock())
        monkeypatch.setattr(sqlite3, "connect", Mock(return_value=connection))
        original_exists = os.path.exists
        monkeypatch.setattr(
            os.path,
            "exists",
            lambda path: False
            if path == "/tmp/leann_calendar_index_copy"
            else original_exists(path),
        )

    builder_class = Mock()
    register = Mock()
    monkeypatch.setattr(cli_module, "LeannBuilder", builder_class)
    monkeypatch.setattr(cli, "register_project_dir", register)
    argv = [
        *command_args(source, tmp_path),
        "--index-name",
        "synthetic-index",
        "--max-count",
        "2",
        "--embedding-model",
        "synthetic-model",
        "--embedding-mode",
        "ollama",
        "--embedding-host",
        "http://127.0.0.1:9999",
    ]
    if no_recompute:
        argv.append("--no-recompute")
    args = cli.create_parser().parse_args(argv)
    asyncio.run(cli.run(args))

    if source == "browser":
        reader_class.return_value.load_data.assert_called_once_with(
            chrome_profile_path=str(
                tmp_path / "Library/Application Support/BraveSoftware/Brave-Browser/Default"
            ),
            max_count=2,
        )
    elif source == "email":
        reader_class.return_value.load_data.assert_called_once_with(
            str(tmp_path / "mail"), max_count=2
        )
    elif source == "imessage":
        reader_class.assert_called_once_with(concatenate_conversations=True)
        reader_class.return_value.load_data.assert_called_once_with()
    elif source == "wechat":
        reader_class.return_value.load_data.assert_called_once_with(
            input_dir=str(tmp_path / "export"),
            max_count=2,
            concatenate_messages=True,
        )
    elif source in {"chatgpt", "claude"}:
        reader_class.assert_called_once_with(concatenate_conversations=True)
        reader_class.return_value.load_data.assert_called_once_with(
            input_dir=str(tmp_path / "export.json"),
            max_count=2,
        )

    if empty:
        builder_class.assert_not_called()
        register.assert_not_called()
        return

    builder_class.assert_called_once_with(
        backend_name="hnsw",
        embedding_model="synthetic-model",
        embedding_mode="ollama",
        embedding_options={"host": "http://127.0.0.1:9999"},
        is_recompute=not no_recompute,
    )
    builder = builder_class.return_value
    if source == "calendar":
        assert builder.add_text.call_count == 2
        assert "Synthetic event 2" in builder.add_text.call_args_list[0].args[0]
        assert "Synthetic event 1" in builder.add_text.call_args_list[1].args[0]
    else:
        assert builder.add_text.call_args_list == [
            call(doc.text, metadata=doc.metadata) for doc in documents
        ]
    builder.build_index.assert_called_once_with(cli.get_index_path("synthetic-index"))
    register.assert_called_once_with()


def test_top_level_help(cli, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.create_parser().parse_args(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    for source in SOURCES:
        assert f"index-{source}" in output


def test_unrelated_list_dispatch(cli, monkeypatch):
    list_indexes = Mock()
    monkeypatch.setattr(cli, "list_indexes", list_indexes)
    args = cli.create_parser().parse_args(["list", "--max-depth", "2"])
    asyncio.run(cli.run(args))
    list_indexes.assert_called_once_with(2)
