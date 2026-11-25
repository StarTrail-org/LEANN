import asyncio

import leann.cli
import leann.react_agent
import pytest
from leann.cli import LeannCLI


class _CapturedError(Exception):
    pass


@pytest.mark.parametrize(
    "command, handler, module, factory",
    [
        ("ask", "ask_questions", leann.cli, "LeannChat"),
        ("react", "react_agent", leann.react_agent, "create_react_agent"),
    ],
)
def test_cli_novita_forwards_api_options(tmp_path, monkeypatch, command, handler, module, factory):
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_factory(**kwargs):
        captured.update(kwargs["llm_config"])
        raise _CapturedError

    monkeypatch.setattr(module, factory, fake_factory)

    cli = LeannCLI()
    monkeypatch.setattr(cli, "index_exists", lambda name: True)
    args = cli.create_parser().parse_args(
        [
            command,
            "my-docs",
            "a question",
            "--llm",
            "novita",
            "--api-key",
            "cli-key",
            "--api-base",
            "https://proxy.example/v1",
        ]
    )

    with pytest.raises(_CapturedError):
        asyncio.run(getattr(cli, handler)(args))

    assert captured["api_key"] == "cli-key"
    assert captured["base_url"] == "https://proxy.example/v1"
