import atexit
import contextlib
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .settings import encode_provider_options

# Lightweight, self-contained server manager with no cross-process inspection

# Set up logging based on environment variable
LOG_LEVEL = os.getenv("LEANN_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
_LOCK_STALE_SECONDS = 600
_FLOCK_TIMEOUT_SECONDS = 300
_REGISTRY_LOCKS_GUARD = threading.Lock()
_REGISTRY_LOCKS: dict[str, threading.Lock] = {}


def _flock_acquire(lock_file) -> None:
    """Acquire an exclusive file lock for cross-process synchronisation.

    Uses ``fcntl.flock`` on POSIX and ``msvcrt.locking`` on Windows.  Both are
    auto-released when the file descriptor is closed or the owning process
    exits, so a holder crash does not permanently block other waiters.
    """
    try:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return
    except ImportError:
        pass
    except OSError:
        return

    if sys.platform != "win32":
        return
    import msvcrt

    # msvcrt.locking operates on byte ranges; ensure the file has content.
    lock_file.seek(0, 2)
    if lock_file.tell() == 0:
        lock_file.write("\n")
        lock_file.flush()
    lock_file.seek(0)

    deadline = time.monotonic() + _FLOCK_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            time.sleep(0.5)
    logger.warning(
        "Cross-process file lock timed out after %ds; proceeding without lock",
        _FLOCK_TIMEOUT_SECONDS,
    )


def _flock_release(lock_file) -> None:
    """Release the file lock acquired by :func:`_flock_acquire`."""
    try:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return
    except ImportError:
        pass
    except OSError:
        return

    if sys.platform != "win32":
        return
    try:
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    except (ImportError, OSError):
        pass


def _is_colab_environment() -> bool:
    """Check if we're running in Google Colab environment."""
    return "COLAB_GPU" in os.environ or "COLAB_TPU" in os.environ


def _get_available_port(start_port: int = 5557) -> int:
    """Get an available port starting from start_port."""
    port = start_port
    while port < start_port + 100:  # Try up to 100 ports
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            port += 1
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + 100}")


def _check_port(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _pid_is_alive(pid: int) -> bool:
    """Best-effort liveness check for a process id."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# Note: All cross-process scanning helpers removed for simplicity


def _safe_resolve(path: Path) -> str:
    """Resolve paths safely even if the target does not yet exist."""
    try:
        return str(path.resolve(strict=False))
    except Exception:
        return str(path)


def _safe_stat_signature(path: Path) -> dict:
    """Return a lightweight signature describing the current state of a path."""
    signature: dict[str, object] = {"path": _safe_resolve(path)}
    try:
        stat = path.stat()
    except FileNotFoundError:
        signature["missing"] = True
    except Exception as exc:  # pragma: no cover - unexpected filesystem errors
        signature["error"] = str(exc)
    else:
        signature["mtime_ns"] = stat.st_mtime_ns
        signature["size"] = stat.st_size
    return signature


def _build_passages_signature(passages_file: Optional[str]) -> Optional[dict]:
    """Collect modification signatures for metadata and referenced passage files."""
    if not passages_file:
        return None

    meta_path = Path(passages_file)
    signature: dict[str, object] = {"meta": _safe_stat_signature(meta_path)}

    try:
        with meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)
    except FileNotFoundError:
        signature["meta_missing"] = True
        signature["sources"] = []
        return signature
    except json.JSONDecodeError as exc:
        signature["meta_error"] = f"json_error:{exc}"
        signature["sources"] = []
        return signature
    except Exception as exc:  # pragma: no cover - unexpected errors
        signature["meta_error"] = str(exc)
        signature["sources"] = []
        return signature

    base_dir = meta_path.parent
    seen_paths: set[str] = set()
    source_signatures: list[dict[str, object]] = []

    for source in meta.get("passage_sources", []):
        for key, kind in (
            ("path", "passages"),
            ("path_relative", "passages"),
            ("index_path", "index"),
            ("index_path_relative", "index"),
        ):
            raw_path = source.get(key)
            if not raw_path:
                continue
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            resolved = _safe_resolve(candidate)
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            sig = _safe_stat_signature(candidate)
            sig["kind"] = kind
            source_signatures.append(sig)

    signature["sources"] = source_signatures
    return signature


# Note: All cross-process scanning helpers removed for simplicity


class EmbeddingServerManager:
    """
    A simplified manager for embedding server processes that avoids complex update mechanisms.
    """

    def __init__(self, backend_module_name: str):
        """
        Initializes the manager for a specific backend.

        Args:
            backend_module_name (str): The full module name of the backend's server script.
                                       e.g., "leann_backend_diskann.embedding_server"
        """
        self.backend_module_name = backend_module_name
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port: Optional[int] = None
        # Track last-started config for in-process reuse only
        self._server_config: Optional[dict] = None
        self._daemon_mode = False
        self._registry_path: Optional[Path] = None
        self._atexit_registered = False
        # Also register a weakref finalizer to ensure cleanup when manager is GC'ed
        try:
            import weakref

            self._finalizer = weakref.finalize(self, self._finalize_process)
        except Exception:
            self._finalizer = None

    def start_server(
        self,
        port: int,
        model_name: str,
        embedding_mode: str = "sentence-transformers",
        **kwargs,
    ) -> tuple[bool, int]:
        """Start the embedding server."""
        # passages_file may be present in kwargs for server CLI, but we don't need it here
        provider_options = kwargs.pop("provider_options", None)
        passages_file = kwargs.get("passages_file", "")
        distance_metric = kwargs.get("distance_metric", "")
        use_daemon = bool(kwargs.get("use_daemon", True))
        daemon_ttl_seconds = int(kwargs.get("daemon_ttl_seconds", 900))

        config_signature = self._build_config_signature(
            model_name=model_name,
            embedding_mode=embedding_mode,
            provider_options=provider_options,
            passages_file=passages_file,
            distance_metric=distance_metric,
        )

        # If this manager already has a live server, just reuse it
        if (
            self.server_process
            and self.server_process.poll() is None
            and self.server_port
            and self._server_config == config_signature
        ):
            logger.info("Reusing in-process server")
            return True, self.server_port

        # Configuration changed, stop existing ephemeral server before starting a new one
        if self.server_process and self.server_process.poll() is None and not self._daemon_mode:
            logger.info("Existing server configuration differs; restarting embedding server")
            self.stop_server()

        # Reuse an already-running daemon from registry if possible.
        if use_daemon and not _is_colab_environment():
            with self._registry_lock(config_signature):
                adopted = self._adopt_registered_server(config_signature)
                if adopted is not None:
                    self.server_process = None
                    self.server_port = adopted
                    self._server_config = config_signature
                    self._daemon_mode = True
                    return True, adopted

        # For Colab environment, use a different strategy
        if _is_colab_environment():
            logger.info("Detected Colab environment, using alternative startup strategy")
            return self._start_server_colab(
                port,
                model_name,
                embedding_mode,
                config_signature=config_signature,
                provider_options=provider_options,
                **kwargs,
            )

        # Always pick a fresh available port
        try:
            actual_port = _get_available_port(port)
        except RuntimeError:
            logger.error("No available ports found")
            return False, port

        # Start a new server (guarded by lock in daemon mode to avoid race).
        if use_daemon and not _is_colab_environment():
            with self._registry_lock(config_signature):
                adopted = self._adopt_registered_server(config_signature)
                if adopted is not None:
                    self.server_process = None
                    self.server_port = adopted
                    self._server_config = config_signature
                    self._daemon_mode = True
                    return True, adopted
                started, actual_port = self._start_new_server(
                    actual_port,
                    model_name,
                    embedding_mode,
                    provider_options=provider_options,
                    config_signature=config_signature,
                    **kwargs,
                )
                if started:
                    self._daemon_mode = True
                    self._registry_path = self._write_registry_record(
                        port=actual_port,
                        config_signature=config_signature,
                        daemon_ttl_seconds=daemon_ttl_seconds,
                    )
                return started, actual_port

        started, actual_port = self._start_new_server(
            actual_port,
            model_name,
            embedding_mode,
            provider_options=provider_options,
            config_signature=config_signature,
            **kwargs,
        )
        if started:
            self._daemon_mode = False
        return started, actual_port

    def _build_config_signature(
        self,
        *,
        model_name: str,
        embedding_mode: str,
        provider_options: Optional[dict],
        passages_file: Optional[str],
        distance_metric: Optional[str],
    ) -> dict:
        """Create a signature describing the current server configuration."""
        passages_path = ""
        if passages_file:
            passages_path = _safe_resolve(Path(passages_file))
        return {
            "model_name": model_name,
            "passages_file": passages_path,
            "embedding_mode": embedding_mode,
            "distance_metric": distance_metric or "",
            "provider_options": provider_options or {},
            "passages_signature": _build_passages_signature(passages_file),
        }

    def _start_server_colab(
        self,
        port: int,
        model_name: str,
        embedding_mode: str = "sentence-transformers",
        *,
        config_signature: Optional[dict] = None,
        provider_options: Optional[dict] = None,
        **kwargs,
    ) -> tuple[bool, int]:
        """Start server with Colab-specific configuration."""
        # Try to find an available port
        try:
            actual_port = _get_available_port(port)
        except RuntimeError:
            logger.error("No available ports found")
            return False, port

        logger.info(f"Starting server on port {actual_port} for Colab environment")

        # Use a simpler startup strategy for Colab
        command = self._build_server_command(actual_port, model_name, embedding_mode, **kwargs)

        try:
            # In Colab, we'll use a more direct approach
            self._launch_server_process_colab(
                command,
                actual_port,
                provider_options=provider_options,
                config_signature=config_signature,
            )
            started, ready_port = self._wait_for_server_ready_colab(actual_port)
            if started:
                self._server_config = config_signature or {
                    "model_name": model_name,
                    "passages_file": kwargs.get("passages_file", ""),
                    "embedding_mode": embedding_mode,
                    "provider_options": provider_options or {},
                }
            return started, ready_port
        except Exception as e:
            logger.error(f"Failed to start embedding server in Colab: {e}")
            return False, actual_port

    # Note: No compatibility check needed; manager is per-searcher and configs are stable per instance

    def _start_new_server(
        self,
        port: int,
        model_name: str,
        embedding_mode: str,
        provider_options: Optional[dict] = None,
        config_signature: Optional[dict] = None,
        **kwargs,
    ) -> tuple[bool, int]:
        """Start a new embedding server on the given port."""
        logger.info(f"Starting embedding server on port {port}...")

        command = self._build_server_command(port, model_name, embedding_mode, **kwargs)

        try:
            self._launch_server_process(
                command,
                port,
                provider_options=provider_options,
                config_signature=config_signature,
            )
            started, ready_port = self._wait_for_server_ready(port)
            if started:
                self._server_config = config_signature or {
                    "model_name": model_name,
                    "passages_file": kwargs.get("passages_file", ""),
                    "embedding_mode": embedding_mode,
                    "provider_options": provider_options or {},
                }
            return started, ready_port
        except Exception as e:
            logger.error(f"Failed to start embedding server: {e}")
            return False, port

    def _build_server_command(
        self, port: int, model_name: str, embedding_mode: str, **kwargs
    ) -> list:
        """Build the command to start the embedding server."""
        command = [
            sys.executable,
            "-m",
            self.backend_module_name,
            "--zmq-port",
            str(port),
            "--model-name",
            model_name,
        ]

        if kwargs.get("passages_file"):
            # Convert to absolute path to ensure subprocess can find the file
            passages_file = Path(kwargs["passages_file"]).resolve()
            command.extend(["--passages-file", str(passages_file)])
        if embedding_mode != "sentence-transformers":
            command.extend(["--embedding-mode", embedding_mode])
        if kwargs.get("distance_metric"):
            command.extend(["--distance-metric", kwargs["distance_metric"]])
        if kwargs.get("enable_warmup", True):
            command.append("--enable-warmup")
        if kwargs.get("use_daemon", True):
            command.append("--daemon-mode")
            ttl = int(kwargs.get("daemon_ttl_seconds", 900))
            command.extend(["--daemon-ttl", str(ttl)])

        return command

    def _launch_server_process(
        self,
        command: list,
        port: int,
        *,
        provider_options: Optional[dict] = None,
        config_signature: Optional[dict] = None,
    ) -> None:
        """Launch the server process."""
        project_root = Path(__file__).parent.parent.parent.parent.parent
        logger.info(f"Command: {' '.join(command)}")

        # In CI environment, redirect stdout to avoid buffer deadlock but keep stderr for debugging
        # Embedding servers use many print statements that can fill stdout buffers
        is_ci = os.environ.get("CI") == "true"
        if is_ci:
            stdout_target = subprocess.DEVNULL
            stderr_target = None  # Keep stderr for error debugging in CI
            logger.info(
                "CI environment detected, redirecting embedding server stdout to DEVNULL, keeping stderr"
            )
        else:
            stdout_target = None  # Direct to console for visible logs
            stderr_target = None  # Direct to console for visible logs

        # Start embedding server subprocess
        logger.info(f"Starting server process with command: {' '.join(command)}")
        env = os.environ.copy()
        encoded_options = encode_provider_options(provider_options)
        if encoded_options:
            env["LEANN_EMBEDDING_OPTIONS"] = encoded_options

        self.server_process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=stdout_target,
            stderr=stderr_target,
            env=env,
        )
        self.server_port = port
        # Record config for in-process reuse (best effort; refined later when ready)
        if config_signature is not None:
            self._server_config = config_signature
        else:  # Fallback for unexpected code paths
            try:
                self._server_config = {
                    "model_name": command[command.index("--model-name") + 1]
                    if "--model-name" in command
                    else "",
                    "passages_file": command[command.index("--passages-file") + 1]
                    if "--passages-file" in command
                    else "",
                    "embedding_mode": command[command.index("--embedding-mode") + 1]
                    if "--embedding-mode" in command
                    else "sentence-transformers",
                    "provider_options": provider_options or {},
                }
            except Exception:
                self._server_config = {
                    "model_name": "",
                    "passages_file": "",
                    "embedding_mode": "sentence-transformers",
                    "provider_options": provider_options or {},
                }
        logger.info(f"Server process started with PID: {self.server_process.pid}")

        # Register atexit callback only when we actually start a process
        if not self._atexit_registered:
            # Always attempt best-effort finalize at interpreter exit
            atexit.register(self._finalize_process)
            self._atexit_registered = True
        # Touch finalizer so it knows there is a live process
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None and getattr(finalizer, "alive", False) is False:
            try:
                import weakref

                self._finalizer = weakref.finalize(self, self._finalize_process)
            except Exception:
                pass

    def _wait_for_server_ready(self, port: int) -> tuple[bool, int]:
        """Wait for the server to be ready."""
        max_wait, wait_interval = 120, 0.5
        for _ in range(int(max_wait / wait_interval)):
            if _check_port(port):
                logger.info("Embedding server is ready!")
                return True, port

            if self.server_process and self.server_process.poll() is not None:
                logger.error("Server terminated during startup.")
                return False, port

            time.sleep(wait_interval)

        logger.error(f"Server failed to start within {max_wait} seconds.")
        self.stop_server()
        return False, port

    def stop_server(self):
        """Stops the embedding server process if it's running."""
        if self._daemon_mode:
            # Daemon is intentionally process-global: this manager detaches by default.
            self.server_process = None
            self.server_port = None
            self._server_config = None
            self._daemon_mode = False
            self._registry_path = None
            return

        if not self.server_process:
            return

        if self.server_process and self.server_process.poll() is not None:
            # Process already terminated
            self.server_process = None
            self.server_port = None
            self._server_config = None
            return

        logger.info(
            f"Terminating server process (PID: {self.server_process.pid}) for backend {self.backend_module_name}..."
        )

        # Use simple termination first; if the server installed signal handlers,
        # it will exit cleanly. Otherwise escalate to kill after a short wait.
        try:
            self.server_process.terminate()
        except Exception:
            pass

        try:
            self.server_process.wait(timeout=5)  # Give more time for graceful shutdown
            logger.info(f"Server process {self.server_process.pid} terminated gracefully.")
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Server process {self.server_process.pid} did not terminate within 5 seconds, force killing..."
            )
            try:
                self.server_process.kill()
            except Exception:
                pass
            try:
                self.server_process.wait(timeout=2)
                logger.info(f"Server process {self.server_process.pid} killed successfully.")
            except subprocess.TimeoutExpired:
                logger.error(
                    f"Failed to kill server process {self.server_process.pid} - it may be hung"
                )

        # Clean up process resources with timeout to avoid CI hang
        try:
            # Use shorter timeout in CI environments
            is_ci = os.environ.get("CI") == "true"
            timeout = 3 if is_ci else 10
            self.server_process.wait(timeout=timeout)
            logger.info(f"Server process {self.server_process.pid} cleanup completed")
        except subprocess.TimeoutExpired:
            logger.warning(f"Process cleanup timeout after {timeout}s, proceeding anyway")
        except Exception as e:
            logger.warning(f"Error during process cleanup: {e}")
        finally:
            self.server_process = None
            self.server_port = None
            self._server_config = None

    def _finalize_process(self) -> None:
        """Best-effort cleanup used by weakref.finalize/atexit."""
        try:
            self.stop_server()
        except Exception:
            pass

    def _adopt_existing_server(self, *args, **kwargs) -> None:
        # Legacy no-op retained for compatibility.
        return

    def _launch_server_process_colab(
        self,
        command: list,
        port: int,
        *,
        provider_options: Optional[dict] = None,
        config_signature: Optional[dict] = None,
    ) -> None:
        """Launch the server process with Colab-specific settings."""
        logger.info(f"Colab Command: {' '.join(command)}")

        # In Colab, we need to be more careful about process management
        env = os.environ.copy()
        encoded_options = encode_provider_options(provider_options)
        if encoded_options:
            env["LEANN_EMBEDDING_OPTIONS"] = encoded_options

        self.server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        self.server_port = port
        logger.info(f"Colab server process started with PID: {self.server_process.pid}")

        # Register atexit callback (unified)
        if not self._atexit_registered:
            atexit.register(self._finalize_process)
            self._atexit_registered = True
        # Record config for in-process reuse is best-effort in Colab mode
        if config_signature is not None:
            self._server_config = config_signature
        else:
            self._server_config = {
                "model_name": "",
                "passages_file": "",
                "embedding_mode": "sentence-transformers",
                "provider_options": provider_options or {},
            }

    def _wait_for_server_ready_colab(self, port: int) -> tuple[bool, int]:
        """Wait for the server to be ready with Colab-specific timeout."""
        max_wait, wait_interval = 30, 0.5  # Shorter timeout for Colab

        for _ in range(int(max_wait / wait_interval)):
            if _check_port(port):
                logger.info("Colab embedding server is ready!")
                return True, port

            if self.server_process and self.server_process.poll() is not None:
                # Check for error output
                stdout, stderr = self.server_process.communicate()
                logger.error("Colab server terminated during startup.")
                logger.error(f"stdout: {stdout}")
                logger.error(f"stderr: {stderr}")
                return False, port

            time.sleep(wait_interval)

        logger.error(f"Colab server failed to start within {max_wait} seconds.")
        self.stop_server()
        return False, port

    @staticmethod
    def _registry_dir() -> Path:
        return Path.home() / ".leann" / "servers"

    @contextlib.contextmanager
    def _registry_lock(self, config_signature: dict[str, Any]):
        """Best-effort lock around the daemon check-then-start critical section.

        Two layers protect the registry from concurrent mutation:

        1. **threading.Lock** (per registry key, module-global) — serialises
           threads inside the same process.  POSIX ``fcntl.flock`` is
           process-granularity and does *not* block concurrent threads within
           a single process, so this layer is required on every platform.
        2. **File lock** (``fcntl`` on POSIX, ``msvcrt`` on Windows) —
           serialises separate OS processes.
        """
        lock_file = None
        lock_info_path: Optional[Path] = None
        thread_lock: Optional[threading.Lock] = None
        try:
            lock_path = self._registry_dir()
            lock_path.mkdir(parents=True, exist_ok=True)
            lock_key = self._registry_key(config_signature)

            with _REGISTRY_LOCKS_GUARD:
                thread_lock = _REGISTRY_LOCKS.setdefault(lock_key, threading.Lock())
            thread_lock.acquire()

            lock_file = (lock_path / f"{lock_key}.lock").open("a+")
            lock_info_path = lock_path / f"{lock_key}.lockinfo.json"
            self._recover_stale_lock_info(lock_info_path)
            _flock_acquire(lock_file)
            self._write_lock_info(lock_info_path)
            yield
        finally:
            if lock_info_path is not None:
                lock_info_path.unlink(missing_ok=True)
            if lock_file is not None:
                _flock_release(lock_file)
                lock_file.close()
            if thread_lock is not None and thread_lock.locked():
                thread_lock.release()

    def _recover_stale_lock_info(self, lock_info_path: Path) -> None:
        if not lock_info_path.exists():
            return
        try:
            info = json.loads(lock_info_path.read_text(encoding="utf-8"))
            pid = int(info.get("pid") or 0)
            ts = float(info.get("ts") or 0)
        except Exception:
            lock_info_path.unlink(missing_ok=True)
            return

        age = (time.time() - ts) if ts else (_LOCK_STALE_SECONDS + 1)
        if age > _LOCK_STALE_SECONDS or (pid > 0 and not _pid_is_alive(pid)):
            lock_info_path.unlink(missing_ok=True)

    def _write_lock_info(self, lock_info_path: Path) -> None:
        payload = {"pid": os.getpid(), "ts": time.time()}
        tmp_path = lock_info_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp_path, lock_info_path)

    def _registry_key(self, config_signature: dict[str, Any]) -> str:
        payload = {
            "backend_module_name": self.backend_module_name,
            "config_signature": config_signature,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _write_registry_record(
        self,
        *,
        port: int,
        config_signature: dict[str, Any],
        daemon_ttl_seconds: int,
    ) -> Path:
        registry_dir = self._registry_dir()
        registry_dir.mkdir(parents=True, exist_ok=True)
        registry_path = registry_dir / f"{self._registry_key(config_signature)}.json"
        record = {
            "pid": self.server_process.pid if self.server_process else None,
            "port": port,
            "backend_module_name": self.backend_module_name,
            "daemon_ttl_seconds": int(daemon_ttl_seconds),
            "created_at": time.time(),
            "config_signature": config_signature,
        }
        tmp_path = registry_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(tmp_path, registry_path)
        return registry_path

    def _adopt_registered_server(self, config_signature: dict[str, Any]) -> Optional[int]:
        registry_dir = self._registry_dir()
        if not registry_dir.exists():
            return None

        target = registry_dir / f"{self._registry_key(config_signature)}.json"
        if not target.exists():
            return None

        try:
            record = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            target.unlink(missing_ok=True)
            return None

        if record.get("backend_module_name") != self.backend_module_name:
            target.unlink(missing_ok=True)
            return None
        if record.get("config_signature") != config_signature:
            target.unlink(missing_ok=True)
            return None

        pid = int(record.get("pid") or 0)
        port = int(record.get("port") or 0)
        if not _pid_is_alive(pid) or not _check_port(port):
            target.unlink(missing_ok=True)
            return None

        self._registry_path = target
        logger.info("Reusing daemonized embedding server on port %s", port)
        return port

    @classmethod
    def list_daemons(cls) -> list[dict[str, Any]]:
        registry_dir = cls._registry_dir()
        if not registry_dir.exists():
            return []

        records: list[dict[str, Any]] = []
        for record_path in sorted(registry_dir.glob("*.json")):
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except Exception:
                record_path.unlink(missing_ok=True)
                continue

            pid = int(record.get("pid") or 0)
            port = int(record.get("port") or 0)
            alive = _pid_is_alive(pid) and _check_port(port)
            if not alive:
                record_path.unlink(missing_ok=True)
                continue

            record["record_path"] = str(record_path)
            records.append(record)
        return records

    @classmethod
    def stop_daemons(
        cls,
        *,
        backend_module_name: Optional[str] = None,
        passages_file: Optional[str] = None,
    ) -> int:
        resolved_passages_file = _safe_resolve(Path(passages_file)) if passages_file else None
        stopped = 0
        for record in cls.list_daemons():
            if backend_module_name and record.get("backend_module_name") != backend_module_name:
                continue

            record_passages = (
                record.get("config_signature", {}).get("passages_file")
                if isinstance(record.get("config_signature"), dict)
                else None
            )
            if resolved_passages_file and record_passages != resolved_passages_file:
                continue

            pid = int(record.get("pid") or 0)
            try:
                os.kill(pid, 15)
                stopped += 1
            except OSError:
                pass

            record_path = record.get("record_path")
            if record_path:
                Path(record_path).unlink(missing_ok=True)
        return stopped
