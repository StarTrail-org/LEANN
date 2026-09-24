"""MCP tools should use the directory selected at server startup."""

import json
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

module_path = Path(__file__).resolve().parents[1] / "packages/leann-core/src/leann/mcp.py"
mcp = ModuleType("leann_mcp_base_dir_test")
SourceFileLoader(mcp.__name__, str(module_path)).exec_module(mcp)


@pytest.mark.parametrize("relative_base_dir", [False, True])
def test_base_dir_applies_to_cli_and_local_index_reads(tmp_path, monkeypatch, relative_base_dir):
    project_dir = tmp_path / "project"
    launch_dir = tmp_path / "launcher"
    index_dir = project_dir / ".leann" / "indexes" / "docs"
    index_dir.mkdir(parents=True)
    launch_dir.mkdir()
    (index_dir / "documents.leann.meta.json").write_text(
        json.dumps(
            {"backend_name": "ivf", "embedding_model": "existing-model", "embedding_mode": "local"}
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(launch_dir)
    base_dir = Path("..") / "project" if relative_base_dir else project_dir
    monkeypatch.setattr(mcp, "_base_dir", str(base_dir))

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output = "[]" if command[3] == "search" else "ok"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(mcp.subprocess, "run", fake_run)

    status = mcp.handle_status(1, {"index_name": "docs"})
    expected_index_dir = base_dir / ".leann" / "indexes" / "docs"
    assert f"Location: {expected_index_dir}" in status["result"]["content"][0]["text"]

    mcp.handle_build(2, {"index_name": "docs", "docs": ["source.txt"]})
    mcp.handle_list(3)
    mcp.handle_search(4, {"index_name": "docs", "query": "example"})

    assert all(kwargs["cwd"] == base_dir for _, kwargs in calls)
    assert "--embedding-model=existing-model" in calls[0][0]
    assert "--embedding-mode=local" in calls[0][0]
