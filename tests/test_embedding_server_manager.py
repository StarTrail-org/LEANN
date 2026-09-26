import json
import os
import threading
import time
from typing import Any, cast

import pytest
from leann.embedding_server_manager import EmbeddingServerManager


class DummyProcess:
    def __init__(self, pid=12345):
        self.pid = pid
        self._terminated = False

    def poll(self):
        return 0 if self._terminated else None

    def terminate(self):
        self._terminated = True

    def kill(self):
        self._terminated = True

    def wait(self, timeout=None):
        self._terminated = True
        return 0


@pytest.fixture
def embedding_manager(monkeypatch):
    manager = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")

    def fake_get_available_port(start_port):
        return start_port

    monkeypatch.setattr(
        "leann.embedding_server_manager._get_available_port",
        fake_get_available_port,
    )

    start_calls = []

    def fake_start_new_server(self, port, model_name, embedding_mode, **kwargs):
        config_signature = kwargs.get("config_signature")
        start_calls.append(config_signature)
        self.server_process = DummyProcess()
        self.server_port = port
        self._server_config = config_signature
        return True, port

    monkeypatch.setattr(
        EmbeddingServerManager,
        "_start_new_server",
        fake_start_new_server,
    )

    # Ensure stop_server doesn't try to operate on real subprocesses
    def fake_stop_server(self):
        self.server_process = None
        self.server_port = None
        self._server_config = None

    monkeypatch.setattr(EmbeddingServerManager, "stop_server", fake_stop_server)

    return manager, start_calls


def _write_meta(meta_path, passages_name, index_name, total):
    meta_path.write_text(
        json.dumps(
            {
                "backend_name": "hnsw",
                "embedding_model": "test-model",
                "embedding_mode": "sentence-transformers",
                "dimensions": 3,
                "backend_kwargs": {},
                "passage_sources": [
                    {
                        "type": "jsonl",
                        "path": passages_name,
                        "index_path": index_name,
                    }
                ],
                "total_passages": total,
            }
        ),
        encoding="utf-8",
    )


def test_server_restarts_when_metadata_changes(tmp_path, embedding_manager):
    manager, start_calls = embedding_manager

    meta_path = tmp_path / "example.meta.json"
    passages_path = tmp_path / "example.passages.jsonl"
    index_path = tmp_path / "example.passages.idx"

    passages_path.write_text("first\n", encoding="utf-8")
    index_path.write_bytes(b"index")
    _write_meta(meta_path, passages_path.name, index_path.name, total=1)

    # Initial start populates signature
    ok, port = manager.start_server(
        port=6000,
        model_name="test-model",
        passages_file=str(meta_path),
        use_daemon=False,
    )
    assert ok
    assert port == 6000
    assert len(start_calls) == 1

    initial_signature = start_calls[0]["passages_signature"]

    # No metadata change => reuse existing server
    ok, port_again = manager.start_server(
        port=6000,
        model_name="test-model",
        passages_file=str(meta_path),
        use_daemon=False,
    )
    assert ok
    assert port_again == 6000
    assert len(start_calls) == 1

    # Modify passage data and metadata to force signature change
    time.sleep(0.01)  # Ensure filesystem timestamps move forward
    passages_path.write_text("second\n", encoding="utf-8")
    _write_meta(meta_path, passages_path.name, index_path.name, total=2)

    ok, port_third = manager.start_server(
        port=6000,
        model_name="test-model",
        passages_file=str(meta_path),
        use_daemon=False,
    )
    assert ok
    assert port_third == 6000
    assert len(start_calls) == 2

    updated_signature = start_calls[1]["passages_signature"]
    assert updated_signature != initial_signature


def test_list_daemons_ignores_stale_records(tmp_path, monkeypatch):
    registry_dir = tmp_path / "servers"
    registry_dir.mkdir()
    monkeypatch.setattr(EmbeddingServerManager, "_registry_dir", staticmethod(lambda: registry_dir))

    stale = registry_dir / "stale.json"
    stale.write_text(
        json.dumps(
            {
                "pid": 999999,
                "port": 65531,
                "backend_module_name": "leann_backend_hnsw.hnsw_embedding_server",
                "config_signature": {"passages_file": "/tmp/a.meta.json"},
            }
        ),
        encoding="utf-8",
    )

    records = EmbeddingServerManager.list_daemons()
    assert records == []
    assert not stale.exists()


def test_stop_daemons_filters_by_backend_and_passages(tmp_path, monkeypatch):
    registry_dir = tmp_path / "servers"
    registry_dir.mkdir()
    monkeypatch.setattr(EmbeddingServerManager, "_registry_dir", staticmethod(lambda: registry_dir))

    meta_path = (tmp_path / "x.meta.json").resolve()
    record = registry_dir / "daemon.json"
    record.write_text(
        json.dumps(
            {
                "pid": 12345,
                "port": 6001,
                "backend_module_name": "leann_backend_hnsw.hnsw_embedding_server",
                "config_signature": {"passages_file": str(meta_path)},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        EmbeddingServerManager,
        "list_daemons",
        classmethod(
            lambda cls: [
                {
                    "pid": 12345,
                    "port": 6001,
                    "backend_module_name": "leann_backend_hnsw.hnsw_embedding_server",
                    "config_signature": {"passages_file": str(meta_path)},
                    "record_path": str(record),
                }
            ]
        ),
    )

    killed: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int):
        killed.append((pid, sig))

    monkeypatch.setattr(os, "kill", fake_kill)

    stopped = EmbeddingServerManager.stop_daemons(
        backend_module_name="leann_backend_hnsw.hnsw_embedding_server",
        passages_file=str(meta_path),
    )
    assert stopped == 1
    assert killed == [(12345, 15)]
    assert not record.exists()


def test_daemon_registry_reuse_across_manager_instances(tmp_path, monkeypatch):
    """Second manager should adopt the daemon started by first manager."""
    registry_dir = tmp_path / "servers"
    registry_dir.mkdir()
    monkeypatch.setattr(EmbeddingServerManager, "_registry_dir", staticmethod(lambda: registry_dir))
    monkeypatch.setattr(
        "leann.embedding_server_manager._get_available_port",
        lambda start_port: start_port,
    )
    monkeypatch.setattr("leann.embedding_server_manager._check_port", lambda port: True)
    monkeypatch.setattr("leann.embedding_server_manager._pid_is_alive", lambda pid: pid == 22222)

    starts = []

    def fake_start_new_server(self, port, model_name, embedding_mode, **kwargs):
        starts.append((port, model_name, embedding_mode))
        self.server_process = DummyProcess(pid=22222)
        self.server_port = port
        self._server_config = kwargs.get("config_signature")
        return True, port

    monkeypatch.setattr(EmbeddingServerManager, "_start_new_server", fake_start_new_server)

    manager1 = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    ok1, port1 = manager1.start_server(
        port=6011,
        model_name="test-model",
        use_daemon=True,
        daemon_ttl_seconds=120,
    )
    assert ok1 and port1 == 6011
    assert len(starts) == 1

    manager2 = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    ok2, port2 = manager2.start_server(
        port=6011,
        model_name="test-model",
        use_daemon=True,
        daemon_ttl_seconds=120,
    )
    assert ok2 and port2 == 6011
    # No second process spawn: adopted from registry.
    assert len(starts) == 1
    assert manager2.server_process is None


def test_stale_registry_falls_back_to_fresh_start(tmp_path, monkeypatch):
    """If registry points to dead daemon, manager should start a new process."""
    registry_dir = tmp_path / "servers"
    registry_dir.mkdir()
    monkeypatch.setattr(EmbeddingServerManager, "_registry_dir", staticmethod(lambda: registry_dir))
    monkeypatch.setattr(
        "leann.embedding_server_manager._get_available_port",
        lambda start_port: start_port,
    )

    starts = []

    def fake_start_new_server(self, port, model_name, embedding_mode, **kwargs):
        starts.append(port)
        self.server_process = DummyProcess(pid=33333)
        self.server_port = port
        self._server_config = kwargs.get("config_signature")
        return True, port

    monkeypatch.setattr(EmbeddingServerManager, "_start_new_server", fake_start_new_server)

    manager = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    signature = manager._build_config_signature(
        model_name="test-model",
        embedding_mode="sentence-transformers",
        provider_options=None,
        passages_file=None,
        distance_metric=None,
    )
    stale_file = registry_dir / f"{manager._registry_key(signature)}.json"
    stale_file.write_text(
        json.dumps(
            {
                "pid": 999999,
                "port": 6012,
                "backend_module_name": "leann_backend_hnsw.hnsw_embedding_server",
                "config_signature": signature,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("leann.embedding_server_manager._pid_is_alive", lambda pid: False)
    monkeypatch.setattr("leann.embedding_server_manager._check_port", lambda port: False)

    ok, port = manager.start_server(
        port=6012,
        model_name="test-model",
        use_daemon=True,
    )
    assert ok and port == 6012
    assert starts == [6012]
    assert stale_file.exists()
    refreshed = json.loads(stale_file.read_text(encoding="utf-8"))
    assert refreshed["pid"] == 33333


def test_build_server_command_includes_daemon_and_warmup_flags():
    manager = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    command = manager._build_server_command(
        port=6020,
        model_name="m",
        embedding_mode="sentence-transformers",
        distance_metric="mips",
        enable_warmup=True,
        use_daemon=True,
        daemon_ttl_seconds=321,
    )
    assert "--enable-warmup" in command
    assert "--daemon-mode" in command
    assert "--daemon-ttl" in command
    ttl_idx = command.index("--daemon-ttl")
    assert command[ttl_idx + 1] == "321"

    command_no_daemon = manager._build_server_command(
        port=6020,
        model_name="m",
        embedding_mode="sentence-transformers",
        enable_warmup=False,
        use_daemon=False,
    )
    assert "--daemon-mode" not in command_no_daemon
    assert "--enable-warmup" not in command_no_daemon


def test_corrupted_registry_file_is_recovered_on_start(tmp_path, monkeypatch):
    """Invalid registry json should not block startup; file is replaced."""
    registry_dir = tmp_path / "servers"
    registry_dir.mkdir()
    monkeypatch.setattr(EmbeddingServerManager, "_registry_dir", staticmethod(lambda: registry_dir))
    monkeypatch.setattr(
        "leann.embedding_server_manager._get_available_port",
        lambda start_port: start_port,
    )

    starts = []

    def fake_start_new_server(self, port, model_name, embedding_mode, **kwargs):
        starts.append(port)
        self.server_process = DummyProcess(pid=44444)
        self.server_port = port
        self._server_config = kwargs.get("config_signature")
        return True, port

    monkeypatch.setattr(EmbeddingServerManager, "_start_new_server", fake_start_new_server)

    manager = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    signature = manager._build_config_signature(
        model_name="test-model",
        embedding_mode="sentence-transformers",
        provider_options=None,
        passages_file=None,
        distance_metric=None,
    )
    record_path = registry_dir / f"{manager._registry_key(signature)}.json"
    record_path.write_text("{invalid-json", encoding="utf-8")

    monkeypatch.setattr("leann.embedding_server_manager._pid_is_alive", lambda pid: False)
    monkeypatch.setattr("leann.embedding_server_manager._check_port", lambda port: False)

    ok, port = manager.start_server(port=6030, model_name="test-model", use_daemon=True)
    assert ok and port == 6030
    assert starts == [6030]
    data = json.loads(record_path.read_text(encoding="utf-8"))
    assert data["pid"] == 44444


def test_stop_server_detaches_when_daemon_mode(monkeypatch):
    """Daemon mode should detach manager without terminating shared process."""
    manager = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    manager.server_process = cast(Any, DummyProcess(pid=55555))
    manager.server_port = 6031
    manager._server_config = {"model_name": "m"}
    manager._daemon_mode = True

    # If terminate is called this test should fail.
    called = {"terminate": 0}

    def fail_terminate():
        called["terminate"] += 1
        raise AssertionError("terminate should not be called in daemon detach path")

    manager.server_process.terminate = fail_terminate

    manager.stop_server()
    assert called["terminate"] == 0
    assert manager.server_process is None
    assert manager.server_port is None
    assert manager._server_config is None


def test_concurrent_daemon_start_only_spawns_once(tmp_path, monkeypatch):
    """Concurrent calls should be serialized by registry lock for same signature."""
    registry_dir = tmp_path / "servers"
    registry_dir.mkdir()
    monkeypatch.setattr(EmbeddingServerManager, "_registry_dir", staticmethod(lambda: registry_dir))
    monkeypatch.setattr(
        "leann.embedding_server_manager._get_available_port",
        lambda start_port: start_port,
    )
    monkeypatch.setattr("leann.embedding_server_manager._check_port", lambda port: True)
    monkeypatch.setattr("leann.embedding_server_manager._pid_is_alive", lambda pid: pid in (77777,))

    starts = []

    def fake_start_new_server(self, port, model_name, embedding_mode, **kwargs):
        # Force overlap window between two starters.
        time.sleep(0.05)
        starts.append((self, port))
        self.server_process = DummyProcess(pid=77777)
        self.server_port = port
        self._server_config = kwargs.get("config_signature")
        return True, port

    monkeypatch.setattr(EmbeddingServerManager, "_start_new_server", fake_start_new_server)

    results = []

    def runner():
        manager = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
        ok, port = manager.start_server(port=6040, model_name="test-model", use_daemon=True)
        results.append((ok, port))

    t1 = threading.Thread(target=runner)
    t2 = threading.Thread(target=runner)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(results) == 2
    assert all(ok and port == 6040 for ok, port in results)
    # Exactly one actual process start, one adopts registry record.
    assert len(starts) == 1


def test_registry_record_write_is_atomic(tmp_path, monkeypatch):
    """Registry writes should go through temp file + os.replace."""
    registry_dir = tmp_path / "servers"
    registry_dir.mkdir()
    monkeypatch.setattr(EmbeddingServerManager, "_registry_dir", staticmethod(lambda: registry_dir))

    manager = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    manager.server_process = cast(Any, DummyProcess(pid=88888))

    calls = []
    real_replace = os.replace

    def tracked_replace(src, dst):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", tracked_replace)

    path = manager._write_registry_record(
        port=6060,
        config_signature={"model_name": "m"},
        daemon_ttl_seconds=123,
    )
    assert path.exists()
    assert calls, "os.replace should be used for atomic write"
    src, dst = calls[0]
    assert src.endswith(".json.tmp")
    assert dst.endswith(".json")


def test_different_passages_files_start_separate_daemons(tmp_path, monkeypatch):
    """Two managers with different passages files must NOT share a daemon (issue #281).

    Simulates sequential test runs where each test builds its own index in a
    separate directory but uses the same model and embedding mode.  A stale
    daemon from run A should never be adopted by run B.
    """
    registry_dir = tmp_path / "servers"
    registry_dir.mkdir()
    monkeypatch.setattr(EmbeddingServerManager, "_registry_dir", staticmethod(lambda: registry_dir))

    # Each call to _get_available_port returns a strictly increasing port so
    # that the two daemons land on different ports when they are not sharing.
    port_seq = [6200]

    def fake_get_available_port(start_port):
        p = port_seq[0]
        port_seq[0] += 1
        return p

    monkeypatch.setattr(
        "leann.embedding_server_manager._get_available_port",
        fake_get_available_port,
    )

    alive_pids: set[int] = set()
    monkeypatch.setattr(
        "leann.embedding_server_manager._pid_is_alive", lambda pid: pid in alive_pids
    )
    monkeypatch.setattr("leann.embedding_server_manager._check_port", lambda port: True)

    pid_seq = [55001]
    starts: list[dict] = []

    def fake_start_new_server(self, port, model_name, embedding_mode, **kwargs):
        pid = pid_seq[0]
        pid_seq[0] += 1
        alive_pids.add(pid)
        starts.append({"port": port, "config": kwargs.get("config_signature")})
        self.server_process = DummyProcess(pid=pid)
        self.server_port = port
        self._server_config = kwargs.get("config_signature")
        return True, port

    monkeypatch.setattr(EmbeddingServerManager, "_start_new_server", fake_start_new_server)

    # Simulate two separate test runs, each in its own temp subdirectory,
    # using the *same* filename patterns but *different* parent directories.
    run_a = tmp_path / "run_a"
    run_a.mkdir()
    meta_a = run_a / "myindex.meta.json"
    passages_a = run_a / "myindex.passages.jsonl"
    passages_a.write_text("passage: hello world\n", encoding="utf-8")
    _write_meta(meta_a, passages_a.name, "myindex.passages.idx", total=1)

    run_b = tmp_path / "run_b"
    run_b.mkdir()
    meta_b = run_b / "myindex.meta.json"
    passages_b = run_b / "myindex.passages.jsonl"
    passages_b.write_text("passage: different content\n", encoding="utf-8")
    _write_meta(meta_b, passages_b.name, "myindex.passages.idx", total=1)

    # Run A: start daemon with index_a passages
    manager_a = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    ok_a, port_a = manager_a.start_server(
        port=6200,
        model_name="test-model",
        passages_file=str(meta_a),
        use_daemon=True,
    )
    assert ok_a
    assert len(starts) == 1, "First manager should start one daemon"

    # Run B: a fresh manager with a DIFFERENT passages file must not adopt run A's daemon
    manager_b = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    ok_b, port_b = manager_b.start_server(
        port=6200,
        model_name="test-model",
        passages_file=str(meta_b),
        use_daemon=True,
    )
    assert ok_b
    assert len(starts) == 2, (
        "A separate daemon must be spawned for each distinct passages file; "
        "sharing daemons across different indices causes 'Failed to fetch embeddings' errors"
    )
    assert port_a != port_b, "Each index must communicate with its own daemon port"


def test_same_passages_file_reuses_existing_daemon(tmp_path, monkeypatch):
    """Two managers with identical passages files SHOULD share the same daemon."""
    registry_dir = tmp_path / "servers"
    registry_dir.mkdir()
    monkeypatch.setattr(EmbeddingServerManager, "_registry_dir", staticmethod(lambda: registry_dir))
    monkeypatch.setattr(
        "leann.embedding_server_manager._get_available_port", lambda start_port: start_port
    )

    alive_pids: set[int] = {99001}
    monkeypatch.setattr(
        "leann.embedding_server_manager._pid_is_alive", lambda pid: pid in alive_pids
    )
    monkeypatch.setattr("leann.embedding_server_manager._check_port", lambda port: True)

    starts: list[int] = []

    def fake_start_new_server(self, port, model_name, embedding_mode, **kwargs):
        starts.append(port)
        self.server_process = DummyProcess(pid=99001)
        self.server_port = port
        self._server_config = kwargs.get("config_signature")
        return True, port

    monkeypatch.setattr(EmbeddingServerManager, "_start_new_server", fake_start_new_server)

    meta = tmp_path / "shared.meta.json"
    passages = tmp_path / "shared.passages.jsonl"
    passages.write_text("shared passage\n", encoding="utf-8")
    _write_meta(meta, passages.name, "shared.passages.idx", total=1)

    manager1 = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    ok1, port1 = manager1.start_server(
        port=6300, model_name="test-model", passages_file=str(meta), use_daemon=True
    )
    assert ok1 and len(starts) == 1

    manager2 = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    ok2, port2 = manager2.start_server(
        port=6300, model_name="test-model", passages_file=str(meta), use_daemon=True
    )
    assert ok2
    assert len(starts) == 1, "Second manager should reuse the existing daemon, not start a new one"
    assert port1 == port2, "Both managers should connect to the same daemon port"


def test_stale_lock_info_removed_when_pid_dead(tmp_path, monkeypatch):
    registry_dir = tmp_path / "servers"
    registry_dir.mkdir()
    monkeypatch.setattr(EmbeddingServerManager, "_registry_dir", staticmethod(lambda: registry_dir))
    monkeypatch.setattr("leann.embedding_server_manager._pid_is_alive", lambda pid: False)

    manager = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    signature = manager._build_config_signature(
        model_name="x",
        embedding_mode="sentence-transformers",
        provider_options=None,
        passages_file=None,
        distance_metric=None,
    )
    key = manager._registry_key(signature)
    lock_info = registry_dir / f"{key}.lockinfo.json"
    lock_info.write_text(json.dumps({"pid": 999999, "ts": time.time()}), encoding="utf-8")

    with manager._registry_lock(signature):
        pass

    assert not lock_info.exists()
