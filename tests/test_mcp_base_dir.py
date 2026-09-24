"""MCP tools should use the directory selected at server startup."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

module_path = Path(__file__).resolve().parents[1] / "packages/leann-core/src/leann/mcp.py"
spec = importlib.util.spec_from_file_location("leann_mcp_base_dir_test", module_path)
mcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp)


def test_base_dir_applies_to_cli_and_local_index_reads(tmp_path, monkeypatch):
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
    monkeypatch.setattr(mcp, "_base_dir", str(project_dir))

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output = "[]" if command[3] == "search" else "ok"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(mcp.subprocess, "run", fake_run)

    status = mcp.handle_status(1, {"index_name": "docs"})
    assert f"Location: {index_dir}" in status["result"]["content"][0]["text"]

    mcp.handle_build(2, {"index_name": "docs", "docs": ["source.txt"]})
    mcp.handle_list(3)
    mcp.handle_search(4, {"index_name": "docs", "query": "example"})

    assert all(kwargs["cwd"] == project_dir for _, kwargs in calls)
    assert "--embedding-model=existing-model" in calls[0][0]
    assert "--embedding-mode=local" in calls[0][0]
