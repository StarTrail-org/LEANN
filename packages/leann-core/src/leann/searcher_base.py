import hashlib
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from .embedding_server_manager import EmbeddingServerManager
from .interface import LeannBackendSearcherInterface


class QueryEmbeddingCache:
    """LRU cache for query embeddings to avoid recomputation.

    Args:
        max_size: Maximum number of cached queries per searcher. Values <= 0
            disable the cache.
        embedding_model: Model name used to scope cache keys. An entry is never
            served to a different model.
    """

    def __init__(self, max_size: int = 1000, embedding_model: Optional[str] = None):
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.max_size = max_size
        self.embedding_model = embedding_model or ""

    def _hash_query(self, query: str, query_template: Optional[str] = None) -> str:
        """Create hash key for query."""
        key_data = {
            "query": query,
            "template": query_template or "",
            "model": self.embedding_model,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, query: str, query_template: Optional[str] = None) -> Optional[np.ndarray]:
        """Get cached embedding if exists.

        Returns a copy of the stored vector with shape (D,).
        Callers that need batch shape should reshape.
        """
        key = self._hash_query(query, query_template)
        cached = self.cache.get(key)
        if cached is None:
            return None
        # Mark as most-recently-used; without this, eviction below is FIFO and
        # a hot query can be dropped while a never-reused one survives.
        self.cache.move_to_end(key)
        return cached.copy()

    def put(self, query: str, embedding: np.ndarray, query_template: Optional[str] = None):
        """Cache embedding (stores a 1-D vector of shape (D,))."""
        key = self._hash_query(query, query_template)

        # Normalize to 1-D so cache hits always return a consistent shape
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)

        # max_size <= 0 disables the cache; storing nothing keeps `get` a miss
        # and avoids evicting from an empty mapping.
        if self.max_size <= 0:
            return

        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # evict least-recently-used

        self.cache[key] = vec.copy()

    def clear(self):
        """Clear cache."""
        self.cache.clear()


class ReusableZMQConnection:
    """Persistent ZMQ REQ connection to the embedding server.

    The socket is created once and reused across searches instead of being
    rebuilt per query. It lives on the searcher for the searcher's lifetime,
    and `LeannChat` and the LlamaIndex retrievers hold a searcher for as long
    as they live.

    **A `zmq.Socket` is not thread-safe.** A searcher sharing this connection
    must not have `search` called concurrently from multiple threads; give
    each thread its own searcher. The previous per-query socket did not carry
    this constraint.

    """

    def __init__(self):
        self.context = None
        self.socket = None
        self.port = None

    def connect(self, port: int):
        """Connect to ZMQ server on given port."""
        import zmq

        if self.port == port and self.socket is not None:
            # Already connected to this port
            return

        # Close existing connection
        self.close()

        # Create new connection
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 second timeout
        self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close
        self.socket.connect(f"tcp://127.0.0.1:{port}")
        self.port = port

    def send_recv(self, data: list) -> list:
        """Send data and receive response."""
        import msgpack

        if self.socket is None:
            raise RuntimeError("ZMQ connection not established")

        # Send request
        request_bytes = msgpack.packb(data)
        self.socket.send(request_bytes)

        # Receive response
        response_bytes = self.socket.recv()
        response = msgpack.unpackb(response_bytes)

        return response

    def close(self):
        """Close ZMQ connection safely (tolerates partial/torn-down state)."""
        socket = self.socket
        context = self.context
        self.socket = None
        self.context = None
        self.port = None

        if socket is not None:
            try:
                socket.close(linger=0)
            except Exception:
                pass

        # Deliberately no context.term() here. close() is reached from
        # __del__, and terminating a ZMQ context during interpreter shutdown
        # can block or abort the process with no traceback. Closing the socket
        # with linger=0 releases the fd; the context is freed when its last
        # reference goes.
        del context

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass


class BaseSearcher(LeannBackendSearcherInterface, ABC):
    """
    Abstract base class for Leann searchers, containing common logic for
    loading metadata, managing embedding servers, and handling file paths.
    """

    def __init__(self, index_path: str, backend_module_name: str, **kwargs):
        """
        Initializes the BaseSearcher.

        Args:
            index_path: Path to the Leann index file (e.g., '.../my_index.leann').
            backend_module_name: The specific embedding server module to use
                                 (e.g., 'leann_backend_hnsw.hnsw_embedding_server').
            **kwargs: Additional keyword arguments. `query_cache_size` (default
                1000) caps the per-searcher query embedding cache; pass 0 to
                disable it.
        """
        self.index_path = Path(index_path)
        self.index_dir = self.index_path.parent
        self.meta = kwargs.get("meta", self._load_meta())

        if not self.meta:
            raise ValueError("Searcher requires metadata from .meta.json.")

        self.dimensions = self.meta.get("dimensions")
        if not self.dimensions:
            raise ValueError("Dimensions not found in Leann metadata.")

        self.embedding_model = self.meta.get("embedding_model")
        if not self.embedding_model:
            print("WARNING: embedding_model not found in meta.json. Recompute will fail.")

        self.embedding_mode = self.meta.get("embedding_mode", "sentence-transformers")
        self.embedding_options = self.meta.get("embedding_options", {})
        self.enable_warmup = bool(kwargs.get("enable_warmup", True))
        self.use_daemon = bool(kwargs.get("use_daemon", True))
        self.daemon_ttl_seconds = int(kwargs.get("daemon_ttl_seconds", 900))

        self.embedding_server_manager = EmbeddingServerManager(
            backend_module_name=backend_module_name,
        )

        # Optimization: Query embedding cache
        cache_size = kwargs.get("query_cache_size", 1000)
        self.query_cache = QueryEmbeddingCache(
            max_size=cache_size, embedding_model=self.embedding_model
        )

        # Optimization: Reusable ZMQ connection
        self.zmq_connection = ReusableZMQConnection()
        self._zmq_port: Optional[int] = None

    def _search_provider_options(self) -> dict[str, Any]:
        # compute_query_embedding applies the query template itself; a build template
        # left in here would be prepended to the query by the provider as well.
        return {
            k: v
            for k, v in self.embedding_options.items()
            if k not in ("build_prompt_template", "query_prompt_template", "prompt_template")
        }

    def _load_meta(self) -> dict[str, Any]:
        """Loads the metadata file associated with the index."""
        # This is the corrected logic for finding the meta file.
        meta_path = self.index_dir / f"{self.index_path.name}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Leann metadata file not found at {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    def _ensure_server_running(
        self, passages_source_file: str, port: Optional[int], **kwargs
    ) -> int:
        """
        Ensures the embedding server is running if recompute is needed.
        This is a helper for subclasses.
        """
        if not self.embedding_model:
            raise ValueError("Cannot use recompute mode without 'embedding_model' in meta.json.")

        # Get distance_metric from meta if not provided in kwargs
        distance_metric = (
            kwargs.get("distance_metric")
            or self.meta.get("backend_kwargs", {}).get("distance_metric")
            or "mips"
        )

        search_provider_options = self._search_provider_options()

        server_started, actual_port = self.embedding_server_manager.start_server(
            port=port if port is not None else 5557,
            model_name=self.embedding_model,
            embedding_mode=self.embedding_mode,
            passages_file=passages_source_file,
            distance_metric=distance_metric,
            enable_warmup=kwargs.get("enable_warmup", self.enable_warmup),
            use_daemon=kwargs.get("use_daemon", self.use_daemon),
            daemon_ttl_seconds=kwargs.get("daemon_ttl_seconds", self.daemon_ttl_seconds),
            provider_options=search_provider_options,
        )
        if not server_started:
            # Raised rather than logged: the CLI points fd 2 at /dev/null during a search.
            message = f"Failed to start embedding server on port {actual_port}"
            output = self.embedding_server_manager.daemon_startup_output()
            raise RuntimeError(f"{message}\n{output}" if output else message)

        # Remember port so the reusable ZMQ client can reconnect only when needed.
        # Do not connect here — the server may still be warming; connect on first send.
        self._zmq_port = actual_port

        return actual_port

    def compute_query_embedding(
        self,
        query: str,
        use_server_if_available: bool = True,
        zmq_port: Optional[int] = None,
        query_template: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute embedding for a query string with caching and connection reuse.

        Args:
            query: The query string to embed
            zmq_port: ZMQ port for embedding server
            use_server_if_available: Whether to try using embedding server first
            query_template: Optional prompt template to prepend to query

        Returns:
            Query embedding as numpy array with shape (1, D)
        """
        # Store original query for caching (before template is applied)
        original_query = query

        # Check cache first (before applying template). Cache stores (D,);
        # always return (1, D) to match uncached paths.
        cached = self.query_cache.get(original_query, query_template)
        if cached is not None:
            return np.asarray(cached, dtype=np.float32).reshape(1, -1)

        # Apply query template BEFORE any computation path
        # This ensures template is applied consistently for both server and fallback paths
        if query_template:
            query = f"{query_template}{query}"

        # Try to use embedding server if available and requested
        if use_server_if_available:
            try:
                # Ensure we have a server with passages_file for compatibility
                passages_source_file = self.index_dir / f"{self.index_path.name}.meta.json"
                # Convert to absolute path to ensure server can find it
                zmq_port = self._ensure_server_running(
                    str(passages_source_file.resolve()),
                    zmq_port,
                    enable_warmup=self.enable_warmup,
                    use_daemon=self.use_daemon,
                    daemon_ttl_seconds=self.daemon_ttl_seconds,
                )

                embedding = self._compute_embedding_via_server([query], zmq_port)[
                    0:1
                ]  # Return (1, D) shape

                # Cache the result (use original query before template)
                self.query_cache.put(original_query, embedding[0], query_template)

                return embedding
            except Exception as e:
                print(f"⚠️ Embedding server failed: {e}")
                print("⏭️ Falling back to direct model loading...")

        # Fallback to direct computation
        from .embedding_compute import compute_embeddings

        embedding_mode = self.meta.get("embedding_mode", "sentence-transformers")
        embedding = compute_embeddings(
            [query],
            self.embedding_model,
            embedding_mode,
            provider_options=self._search_provider_options(),
        )

        # Cache the result (use original query before template)
        self.query_cache.put(original_query, embedding[0], query_template)

        return embedding

    def _compute_embedding_via_server(self, chunks: list, zmq_port: int) -> np.ndarray:
        """Compute embeddings using the ZMQ embedding server with connection reuse."""
        # Ensure connection is established (lazy — first request only / port change)
        self.zmq_connection.connect(zmq_port)
        self._zmq_port = zmq_port

        try:
            # Send request and get response using reusable connection
            response = self.zmq_connection.send_recv(chunks)

            # Convert response to numpy array
            if isinstance(response, list) and len(response) > 0:
                return np.array(response, dtype=np.float32)
            else:
                raise RuntimeError("Invalid response from embedding server")

        except Exception as e:
            # Drop broken connection so the next call reconnects cleanly
            try:
                self.zmq_connection.close()
            except Exception:
                pass
            self._zmq_port = None
            raise RuntimeError(f"Failed to compute embeddings via server: {e}")

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        top_k: int,
        complexity: int = 64,
        beam_width: int = 1,
        prune_ratio: float = 0.0,
        recompute_embeddings: bool = False,
        pruning_strategy: Literal["global", "local", "proportional"] = "global",
        zmq_port: Optional[int] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Search for the top_k nearest neighbors of the query vector.

        Args:
            query: Query vectors (B, D) where B is batch size, D is dimension
            top_k: Number of nearest neighbors to return
            complexity: Search complexity/candidate list size, higher = more accurate but slower
            beam_width: Number of parallel search paths/IO requests per iteration
            prune_ratio: Ratio of neighbors to prune via approximate distance (0.0-1.0)
            recompute_embeddings: Whether to fetch fresh embeddings from server vs use stored PQ codes
            pruning_strategy: PQ candidate selection strategy - "global" (default), "local", or "proportional"
            zmq_port: ZMQ port for embedding server communication. Must be provided if recompute_embeddings is True.
            **kwargs: Backend-specific parameters (e.g., batch_size, dedup_node_dis, etc.)

        Returns:
            Dict with 'labels' (list of lists) and 'distances' (ndarray)
        """
        pass

    def __del__(self):
        """Ensures cleanup when the searcher is destroyed."""
        if hasattr(self, "zmq_connection"):
            self.zmq_connection.close()
        if hasattr(self, "embedding_server_manager"):
            self.embedding_server_manager.stop_server()
