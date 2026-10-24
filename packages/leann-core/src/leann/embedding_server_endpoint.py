"""Shared bind-endpoint handling for LEANN's local ZMQ embedding servers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Protocol

DEFAULT_EMBEDDING_SERVER_HOST = "127.0.0.1"
EMBEDDING_SERVER_HOST_ENV = "LEANN_EMBEDDING_SERVER_HOST"


class BindableSocket(Protocol):
    def bind(self, endpoint: str) -> None: ...


def zmq_bind_endpoint(port: int, environ: Mapping[str, str] | None = None) -> str:
    """Return a ZMQ endpoint that stays on loopback unless explicitly overridden."""
    env = os.environ if environ is None else environ
    host = env.get(EMBEDDING_SERVER_HOST_ENV, DEFAULT_EMBEDDING_SERVER_HOST).strip()
    host = host or DEFAULT_EMBEDDING_SERVER_HOST
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"tcp://{host}:{port}"


def bind_zmq_embedding_socket(
    socket: BindableSocket,
    port: int,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Bind an embedding-server socket and return the endpoint for logging/tests."""
    endpoint = zmq_bind_endpoint(port, environ)
    socket.bind(endpoint)
    return endpoint
