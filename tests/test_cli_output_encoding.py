"""The CLI must write UTF-8 when its output is piped, whatever the locale.

Before Python 3.15, Windows encodes piped stdout with the ANSI code page (cp1252 on
English systems). The CLI prints emoji and passage text, so ``leann build``,
``leann list`` and ``leann search --json`` raised UnicodeEncodeError whenever their
output was captured, which is how the MCP server runs them. PYTHONIOENCODING=cp1252
reproduces that code page on any platform.
"""

import json
import os
import subprocess
import sys

from leann import mcp


def _legacy_code_page_env(home):
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "HOME": str(home), "USERPROFILE": str(home)}
    env.pop("PYTHONUTF8", None)
    return env


def test_piped_cli_output_survives_a_legacy_code_page(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "leann", "list"],
        capture_output=True,
        encoding="utf-8",
        timeout=120,
        cwd=tmp_path,
        env=_legacy_code_page_env(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    assert "📚 LEANN Indexes" in result.stdout


def test_mcp_list_works_under_a_legacy_code_page(tmp_path, monkeypatch):
    for key, value in _legacy_code_page_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    monkeypatch.chdir(tmp_path)

    response = mcp.handle_list(1)

    text = json.dumps(response, ensure_ascii=False)
    assert "UnicodeEncodeError" not in text
    assert "📚 LEANN Indexes" in response["result"]["content"][0]["text"]
