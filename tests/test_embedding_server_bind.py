from pathlib import Path

import pytest
from leann.embedding_server_endpoint import (
    bind_zmq_embedding_socket,
    zmq_bind_endpoint,
)


class RecordingSocket:
    def __init__(self):
        self.endpoint: str | None = None

    def bind(self, endpoint: str) -> None:
        self.endpoint = endpoint


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, "tcp://127.0.0.1:5555"),
        ({"LEANN_EMBEDDING_SERVER_HOST": "0.0.0.0"}, "tcp://0.0.0.0:5555"),
        ({"LEANN_EMBEDDING_SERVER_HOST": "::1"}, "tcp://[::1]:5555"),
        ({"LEANN_EMBEDDING_SERVER_HOST": "[::1]"}, "tcp://[::1]:5555"),
        ({"LEANN_EMBEDDING_SERVER_HOST": ""}, "tcp://127.0.0.1:5555"),
    ],
)
def test_zmq_bind_endpoint_stays_local_by_default(environment, expected):
    assert zmq_bind_endpoint(5555, environment) == expected


def test_bind_zmq_embedding_socket_uses_resolved_loopback_endpoint(monkeypatch):
    monkeypatch.delenv("LEANN_EMBEDDING_SERVER_HOST", raising=False)
    socket = RecordingSocket()

    endpoint = bind_zmq_embedding_socket(socket, 6123)

    assert endpoint == "tcp://127.0.0.1:6123"
    assert socket.endpoint == endpoint


def test_bind_zmq_embedding_socket_creates_a_real_loopback_listener():
    zmq = pytest.importorskip("zmq")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    try:
        bind_zmq_embedding_socket(socket, 0)
        actual_endpoint = socket.getsockopt(zmq.LAST_ENDPOINT).decode()
        assert actual_endpoint.startswith("tcp://127.0.0.1:")
        assert not actual_endpoint.endswith(":*")
    finally:
        socket.close(linger=0)
        context.term()


def test_all_embedding_servers_use_the_shared_bind_helper():
    root = Path(__file__).resolve().parents[1] / "packages"
    servers = [
        root / "leann-backend-hnsw/leann_backend_hnsw/hnsw_embedding_server.py",
        root / "leann-backend-diskann/leann_backend_diskann/diskann_embedding_server.py",
    ]

    for server in servers:
        source = server.read_text(encoding="utf-8")
        assert "bind_zmq_embedding_socket(" in source
        assert '.bind(f"tcp://' not in source
