import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from leann.embedding_server_manager import EmbeddingServerManager, _pid_is_alive


def _configure_fake_module_env(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    support_path = str(root / "tests" / "support")
    existing = os.environ.get("PYTHONPATH", "")
    joined = support_path if not existing else f"{support_path}{os.pathsep}{existing}"
    monkeypatch.setenv("PYTHONPATH", joined)


def _wait_until(predicate, timeout=5.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_daemon_reuse_with_real_subprocess(tmp_path, monkeypatch):
    _configure_fake_module_env(monkeypatch)
    monkeypatch.setattr(
        EmbeddingServerManager, "_registry_dir", staticmethod(lambda: tmp_path / "servers")
    )

    manager1 = EmbeddingServerManager("fake_embedding_server_module")
    ok1, port1 = manager1.start_server(
        port=6151,
        model_name="fake-model",
        use_daemon=True,
        daemon_ttl_seconds=60,
        enable_warmup=True,
    )
    assert ok1 and port1 == 6151
    assert manager1.server_process is not None
    pid1 = manager1.server_process.pid

    manager2 = EmbeddingServerManager("fake_embedding_server_module")
    ok2, port2 = manager2.start_server(
        port=6151,
        model_name="fake-model",
        use_daemon=True,
        daemon_ttl_seconds=60,
    )
    assert ok2 and port2 == 6151
    # Adoption path: no newly spawned process object attached.
    assert manager2.server_process is None

    records = EmbeddingServerManager.list_daemons()
    assert len(records) == 1
    assert int(records[0]["pid"]) == pid1

    stopped = EmbeddingServerManager.stop_daemons(
        backend_module_name="fake_embedding_server_module"
    )
    assert stopped == 1


def test_daemon_ttl_expiry_with_real_subprocess(tmp_path, monkeypatch):
    _configure_fake_module_env(monkeypatch)
    monkeypatch.setattr(
        EmbeddingServerManager, "_registry_dir", staticmethod(lambda: tmp_path / "servers")
    )

    manager = EmbeddingServerManager("fake_embedding_server_module")
    ok, port = manager.start_server(
        port=6152,
        model_name="fake-model",
        use_daemon=True,
        daemon_ttl_seconds=1,
        enable_warmup=False,
    )
    assert ok and port == 6152

    # Fake daemon should self-exit after idle TTL.
    expired = _wait_until(lambda: len(EmbeddingServerManager.list_daemons()) == 0, timeout=4.0)
    assert expired


def test_daemon_does_not_keep_the_callers_output_open(tmp_path, monkeypatch):
    _configure_fake_module_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    registry_dir = tmp_path / ".leann" / "servers"
    monkeypatch.setattr(EmbeddingServerManager, "_registry_dir", staticmethod(lambda: registry_dir))

    driver = textwrap.dedent(
        """
        from leann.embedding_server_manager import EmbeddingServerManager

        ok, port = EmbeddingServerManager("fake_embedding_server_module").start_server(
            port=6153,
            model_name="fake-model",
            use_daemon=True,
            daemon_ttl_seconds=60,
            enable_warmup=False,
        )
        print(port if ok else "failed")
        """
    )
    try:
        # Without detaching, this waits for the 60 s daemon TTL and hits the timeout.
        result = subprocess.run(
            [sys.executable, "-c", driver], capture_output=True, text=True, timeout=20
        )
        assert result.returncode == 0, result.stderr

        # list_daemons drops records it considers dead, so capture them first.
        state = {
            path.name: path.read_text(encoding="utf-8", errors="replace")[-500:]
            for path in registry_dir.glob("*")
            if path.suffix in (".json", ".log")
        }
        records = EmbeddingServerManager.list_daemons()
        assert len(records) == 1, f"driver={result.stdout!r} {result.stderr!r} registry={state}"
        assert result.stdout.strip() == str(records[0]["port"])
        assert "fake server" not in result.stdout + result.stderr

        logs = list(registry_dir.glob("*.log"))
        assert len(logs) == 1
        assert _wait_until(lambda: "fake server stderr" in logs[0].read_text(encoding="utf-8"))
        assert "fake server stdout" in logs[0].read_text(encoding="utf-8")
        assert records[0]["log_path"] == str(logs[0])

        if hasattr(os, "getsid"):
            assert os.getsid(records[0]["pid"]) == records[0]["pid"]
    finally:
        EmbeddingServerManager.stop_daemons(backend_module_name="fake_embedding_server_module")


def test_pid_is_alive_follows_a_real_process():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert _pid_is_alive(process.pid)
    finally:
        process.kill()
        process.wait()
    assert not _pid_is_alive(process.pid)


def test_failed_daemon_start_keeps_its_output(tmp_path, monkeypatch):
    monkeypatch.setattr(
        EmbeddingServerManager, "_registry_dir", staticmethod(lambda: tmp_path / "servers")
    )

    manager = EmbeddingServerManager("no_such_embedding_server_module")
    ok, _port = manager.start_server(
        port=6154,
        model_name="fake-model",
        use_daemon=True,
        daemon_ttl_seconds=60,
        enable_warmup=False,
    )

    assert not ok
    assert "No module named no_such_embedding_server_module" in manager.daemon_startup_output()
