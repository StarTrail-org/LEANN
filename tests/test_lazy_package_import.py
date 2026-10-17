"""Importing light submodules must not pull in the whole LEANN stack.

Regression tests for eager package initialisation: ``import leann.mcp`` used to
execute ``leann/__init__.py``, which imported ``leann.api`` and auto-discovered
every backend (faiss included), costing gigabytes for processes that never run
a search.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _run(code: str, extra_path: Path | None = None) -> subprocess.CompletedProcess:
    env = None
    if extra_path is not None:
        paths = [str(extra_path), *filter(None, [os.environ.get("PYTHONPATH")])]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def test_importing_mcp_does_not_load_heavy_modules():
    result = _run(
        """
        import sys
        import leann.mcp
        heavy = [m for m in ("leann.api", "leann.chat", "leann.embedding_server_manager",
                             "leann.react_agent", "faiss") if m in sys.modules]
        print(",".join(heavy))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_public_api_is_still_importable_from_package():
    result = _run(
        """
        import leann
        from leann import LeannBuilder, LeannChat, LeannSearcher, ReActAgent
        from leann import create_react_agent, BACKEND_REGISTRY
        import leann.api
        assert LeannBuilder is leann.api.LeannBuilder
        assert LeannSearcher is leann.api.LeannSearcher
        assert LeannChat is leann.api.LeannChat
        assert BACKEND_REGISTRY is leann.api.BACKEND_REGISTRY
        assert set(leann.__all__) <= set(dir(leann))
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_unknown_attribute_raises_attribute_error():
    result = _run(
        """
        import leann
        try:
            leann.does_not_exist
        except AttributeError as e:
            print("AttributeError", "does_not_exist" in str(e))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "AttributeError True"


def test_backends_are_discovered_when_api_is_imported(tmp_path):
    """Backend auto-discovery must still run for ``from leann.api import ...``."""
    dist = tmp_path / "leann_backend_lazyprobe-0.0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: leann-backend-lazyprobe\nVersion: 0.0.0\n"
    )
    (tmp_path / "leann_backend_lazyprobe.py").write_text(
        "from leann.registry import register_backend\n\n\n"
        "@register_backend('lazyprobe')\n"
        "class Probe:\n"
        "    pass\n"
    )
    for statement in (
        "from leann.api import LeannSearcher",
        "from leann import LeannSearcher",
        "from leann import BACKEND_REGISTRY",
    ):
        result = _run(
            f"""
            {statement}
            from leann.registry import BACKEND_REGISTRY
            print("lazyprobe" in BACKEND_REGISTRY)
            """,
            extra_path=tmp_path,
        )
        assert result.returncode == 0, f"{statement}: {result.stderr}"
        assert result.stdout.strip() == "True", statement
