import contextlib
import logging
import multiprocessing as mp
import os
import struct
import sys
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import psutil
from leann.interface import (
    LeannBackendBuilderInterface,
    LeannBackendFactoryInterface,
    LeannBackendSearcherInterface,
)
from leann.registry import register_backend
from leann.searcher_base import BaseSearcher

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_cpp_output_if_needed():
    """Suppress C++ stdout/stderr based on LEANN_LOG_LEVEL"""
    # In CI we avoid fiddling with low-level file descriptors to prevent aborts
    if os.getenv("CI") == "true":
        yield
        return

    log_level = os.getenv("LEANN_LOG_LEVEL", "WARNING").upper()

    # Only suppress if log level is WARNING or higher (ERROR, CRITICAL)
    should_suppress = log_level in ["WARNING", "ERROR", "CRITICAL"]

    if not should_suppress:
        # Don't suppress, just yield
        yield
        return

    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    # Save original stdout/stderr
    stdout_dup = os.dup(stdout_fd)
    stderr_dup = os.dup(stderr_fd)

    try:
        # Redirect to /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)

        yield

    finally:
        # Restore original file descriptors
        os.dup2(stdout_dup, stdout_fd)
        os.dup2(stderr_dup, stderr_fd)
        os.close(stdout_dup)
        os.close(stderr_dup)


def _get_diskann_metrics():
    from . import _diskannpy as diskannpy  # type: ignore

    return {
        "mips": diskannpy.Metric.INNER_PRODUCT,
        "l2": diskannpy.Metric.L2,
        "cosine": diskannpy.Metric.COSINE,
    }


@contextlib.contextmanager
def chdir(path):
    original_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_dir)


def _write_vectors_to_bin(data: np.ndarray, file_path: Path):
    num_vectors, dim = data.shape
    with open(file_path, "wb") as f:
        f.write(struct.pack("I", num_vectors))
        f.write(struct.pack("I", dim))
        f.write(data.tobytes())


def _calculate_smart_memory_config(data: np.ndarray) -> tuple[float, float]:
    """
    Calculate smart memory configuration for DiskANN based on data size and system specs.

    Args:
        data: The embedding data array

    Returns:
        tuple: (search_memory_maximum, build_memory_maximum) in GB
    """
    num_vectors, dim = data.shape

    # Calculate embedding storage size
    embedding_size_bytes = num_vectors * dim * 4  # float32 = 4 bytes
    embedding_size_gb = embedding_size_bytes / (1024**3)

    # search_memory_maximum: 1/10 of embedding size for optimal PQ compression
    # This controls Product Quantization size - smaller means more compression
    search_memory_gb = max(0.1, embedding_size_gb / 10)  # At least 100MB

    # build_memory_maximum: Based on available system RAM for sharding control
    # This controls how much memory DiskANN uses during index construction
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    total_memory_gb = psutil.virtual_memory().total / (1024**3)

    # Use 50% of available memory, but at least 2GB and at most 75% of total
    build_memory_gb = max(2.0, min(available_memory_gb * 0.5, total_memory_gb * 0.75))

    logger.info(
        f"Smart memory config - Data: {embedding_size_gb:.2f}GB, "
        f"Search mem: {search_memory_gb:.2f}GB (PQ control), "
        f"Build mem: {build_memory_gb:.2f}GB (sharding control)"
    )

    return search_memory_gb, build_memory_gb


@register_backend("diskann")
class DiskannBackend(LeannBackendFactoryInterface):
    @staticmethod
    def builder(**kwargs) -> LeannBackendBuilderInterface:
        return DiskannBuilder(**kwargs)

    @staticmethod
    def searcher(index_path: str, **kwargs) -> LeannBackendSearcherInterface:
        return DiskannSearcher(index_path, **kwargs)


class DiskannBuilder(LeannBackendBuilderInterface):
    def __init__(self, **kwargs):
        self.build_params = kwargs

    def _safe_cleanup_after_partition(self, index_dir: Path, index_prefix: str):
        """
        Safely cleanup files after partition.
        In partition mode, C++ doesn't read _disk.index content,
        so we can delete it if all derived files exist.
        """
        disk_index_file = index_dir / f"{index_prefix}_disk.index"
        beam_search_file = index_dir / f"{index_prefix}_disk_beam_search.index"

        # Required files that C++ partition mode needs
        # Note: C++ generates these with _disk.index suffix
        disk_suffix = "_disk.index"
        required_files = [
            f"{index_prefix}{disk_suffix}_medoids.bin",  # Critical: assert fails if missing
            # Note: _centroids.bin is not created in single-shot build - C++ handles this automatically
            f"{index_prefix}_pq_pivots.bin",  # PQ table
            f"{index_prefix}_pq_compressed.bin",  # PQ compressed vectors
        ]

        # Check if all required files exist
        missing_files = []
        for filename in required_files:
            file_path = index_dir / filename
            if not file_path.exists():
                missing_files.append(filename)

        if missing_files:
            logger.warning(
                f"Cannot safely delete _disk.index - missing required files: {missing_files}"
            )
            logger.info("Keeping all original files for safety")
            return

        # Calculate space savings
        space_saved = 0
        files_to_delete = []

        if disk_index_file.exists():
            space_saved += disk_index_file.stat().st_size
            files_to_delete.append(disk_index_file)

        if beam_search_file.exists():
            space_saved += beam_search_file.stat().st_size
            files_to_delete.append(beam_search_file)

        # Safe to delete!
        for file_to_delete in files_to_delete:
            try:
                os.remove(file_to_delete)
                logger.info(f"✅ Safely deleted: {file_to_delete.name}")
            except Exception as e:
                logger.warning(f"Failed to delete {file_to_delete.name}: {e}")

        if space_saved > 0:
            space_saved_mb = space_saved / (1024 * 1024)
            logger.info(f"💾 Space saved: {space_saved_mb:.1f} MB")

            # Show what files are kept
            logger.info("📁 Kept essential files for partition mode:")
            for filename in required_files:
                file_path = index_dir / filename
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    logger.info(f"  - {filename} ({size_mb:.1f} MB)")

    def build(self, data: np.ndarray, ids: list[str], index_path: str, **kwargs):
        path = Path(index_path)
        index_dir = path.parent
        index_prefix = path.stem
        index_dir.mkdir(parents=True, exist_ok=True)

        if data.dtype != np.float32:
            logger.warning(f"Converting data to float32, shape: {data.shape}")
            data = data.astype(np.float32)

        # DiskANN uses 256 PQ centroids by default for product quantization.
        # When num_vectors < 256, MKL's cblas_sgemm receives degenerate matrix dimensions
        # and emits non-fatal "Parameter N was incorrect" errors to stderr on Windows.
        # The index still builds correctly, but the noise can be confusing.
        # See: https://github.com/StarTrail-org/LEANN/issues/280
        _NUM_PQ_CENTROIDS = 256
        if data.shape[0] < _NUM_PQ_CENTROIDS:
            logger.warning(
                f"DiskANN dataset has only {data.shape[0]} vector(s), which is fewer than "
                f"the default number of PQ centroids ({_NUM_PQ_CENTROIDS}). "
                f"On Windows with Intel MKL this produces non-fatal 'cblas_sgemm parameter' "
                f"errors in the output — the index will still build correctly. "
                f"Consider using '--backend-name hnsw' for small datasets to avoid these messages."
            )

        data_filename = f"{index_prefix}_data.bin"
        _write_vectors_to_bin(data, index_dir / data_filename)

        build_kwargs = {**self.build_params, **kwargs}

        # Extract is_recompute from nested backend_kwargs if needed
        is_recompute = build_kwargs.get("is_recompute", False)
        if not is_recompute and "backend_kwargs" in build_kwargs:
            is_recompute = build_kwargs["backend_kwargs"].get("is_recompute", False)

        # Flatten all backend_kwargs parameters to top level for compatibility
        if "backend_kwargs" in build_kwargs:
            nested_params = build_kwargs.pop("backend_kwargs")
            build_kwargs.update(nested_params)

        metric_enum = _get_diskann_metrics().get(
            build_kwargs.get("distance_metric", "mips").lower()
        )
        if metric_enum is None:
            raise ValueError(
                f"Unsupported distance_metric '{build_kwargs.get('distance_metric', 'unknown')}'."
            )

        # Calculate smart memory configuration if not explicitly provided
        if (
            "search_memory_maximum" not in build_kwargs
            or "build_memory_maximum" not in build_kwargs
        ):
            smart_search_mem, smart_build_mem = _calculate_smart_memory_config(data)
        else:
            smart_search_mem = build_kwargs.get("search_memory_maximum", 4.0)
            smart_build_mem = build_kwargs.get("build_memory_maximum", 8.0)

        try:
            # Build in a child process so native crashes/exits in diskannpy
            # do not terminate the main CLI process silently.
            def _run_build(
                work_dir: str,
                metric: Any,
                data_file: str,
                prefix: str,
                complexity: int,
                graph_degree: int,
                search_mem: float,
                build_mem: float,
                num_threads: int,
                pq_disk_bytes: int,
                out_q: "mp.Queue[tuple[bool, str]]",
            ) -> None:
                try:
                    from . import _diskannpy as diskannpy  # type: ignore

                    with chdir(work_dir):
                        diskannpy.build_disk_float_index(
                            metric,
                            data_file,
                            prefix,
                            complexity,
                            graph_degree,
                            search_mem,
                            build_mem,
                            num_threads,
                            pq_disk_bytes,
                            "",
                        )
                    out_q.put((True, "ok"))
                except Exception as e:
                    out_q.put((False, str(e)))

            queue: mp.Queue[tuple[bool, str]] = mp.Queue()
            proc = mp.Process(
                target=_run_build,
                args=(
                    str(index_dir),
                    metric_enum,
                    data_filename,
                    index_prefix,
                    int(build_kwargs.get("complexity", 64)),
                    int(build_kwargs.get("graph_degree", 32)),
                    float(build_kwargs.get("search_memory_maximum", smart_search_mem)),
                    float(build_kwargs.get("build_memory_maximum", smart_build_mem)),
                    int(build_kwargs.get("num_threads", 8)),
                    int(build_kwargs.get("pq_disk_bytes", 0)),
                    queue,
                ),
            )
            proc.start()
            proc.join()

            if proc.exitcode != 0:
                raise RuntimeError(
                    f"DiskANN native build process exited unexpectedly with code {proc.exitcode}. "
                    "Re-run with LEANN_LOG_LEVEL=DEBUG and check system logs."
                )
            if queue.empty():
                raise RuntimeError(
                    "DiskANN native build exited without status message. "
                    "This usually indicates a native termination inside diskannpy."
                )
            ok, msg = queue.get_nowait()
            if not ok:
                raise RuntimeError(f"DiskANN build failed: {msg}")

            # Validate expected output to catch silent native failures.
            expected_disk_index = index_dir / f"{index_prefix}_disk.index"
            if not expected_disk_index.exists():
                produced = sorted(str(p.name) for p in index_dir.glob(f"{index_prefix}*"))
                raise RuntimeError(
                    "DiskANN build finished but expected index file is missing: "
                    f"{expected_disk_index.name}. Produced files: {produced}"
                )

            # Auto-partition if is_recompute is enabled
            if build_kwargs.get("is_recompute", False):
                logger.info("is_recompute=True, starting automatic graph partitioning...")
                from .graph_partition import partition_graph

                # Partition the index using absolute paths
                # Convert to absolute paths to avoid issues with working directory changes
                absolute_index_dir = Path(index_dir).resolve()
                absolute_index_prefix_path = str(absolute_index_dir / index_prefix)
                disk_graph_path, partition_bin_path = partition_graph(
                    index_prefix_path=absolute_index_prefix_path,
                    output_dir=str(absolute_index_dir),
                    partition_prefix=index_prefix,
                )

                # Safe cleanup: In partition mode, C++ doesn't read _disk.index content
                # but still needs the derived files (_medoids.bin, _centroids.bin, etc.)
                self._safe_cleanup_after_partition(index_dir, index_prefix)

                logger.info("✅ Graph partitioning completed successfully!")
                logger.info(f"  - Disk graph: {disk_graph_path}")
                logger.info(f"  - Partition file: {partition_bin_path}")

        finally:
            temp_data_file = index_dir / data_filename
            if temp_data_file.exists():
                os.remove(temp_data_file)
                logger.debug(f"Cleaned up temporary data file: {temp_data_file}")


class DiskannSearcher(BaseSearcher):
    def __init__(self, index_path: str, **kwargs):
        super().__init__(
            index_path,
            backend_module_name="leann_backend_diskann.diskann_embedding_server",
            **kwargs,
        )

        # Initialize DiskANN index with suppressed C++ output based on log level
        with suppress_cpp_output_if_needed():
            from . import _diskannpy as diskannpy  # type: ignore

            distance_metric = kwargs.get("distance_metric", "mips").lower()
            metric_enum = _get_diskann_metrics().get(distance_metric)
            if metric_enum is None:
                raise ValueError(f"Unsupported distance_metric '{distance_metric}'.")

            self.num_threads = kwargs.get("num_threads", 8)

            # For DiskANN, we need to reinitialize the index when zmq_port changes
            # Store the initialization parameters for later use
            # Note: C++ load method expects the BASE path (without _disk.index suffix)
            # C++ internally constructs: index_prefix + "_disk.index"
            index_name = self.index_path.stem  # "simple_test.leann" -> "simple_test"
            diskann_index_prefix = str(self.index_dir / index_name)  # /path/to/simple_test
            full_index_prefix = diskann_index_prefix  # /path/to/simple_test (base path)

            # Auto-detect partition files and set partition_prefix
            partition_graph_file = self.index_dir / f"{index_name}_disk_graph.index"
            partition_bin_file = self.index_dir / f"{index_name}_partition.bin"

            partition_prefix = ""
            if partition_graph_file.exists() and partition_bin_file.exists():
                # C++ expects full path prefix, not just filename
                partition_prefix = str(self.index_dir / index_name)  # /path/to/simple_test
                logger.info(
                    f"✅ Detected partition files, using partition_prefix='{partition_prefix}'"
                )
            else:
                logger.debug("No partition files detected, using standard index files")

            self._init_params = {
                "metric_enum": metric_enum,
                "full_index_prefix": full_index_prefix,
                "num_threads": self.num_threads,
                "num_nodes_to_cache": kwargs.get("num_nodes_to_cache", 0),
                # 1 -> initialize cache using sample_data; 2 -> ready cache without init; others disable cache
                "cache_mechanism": kwargs.get("cache_mechanism", 1),
                "pq_prefix": "",
                "partition_prefix": partition_prefix,
            }

            # Log partition configuration for debugging
            if partition_prefix:
                logger.info(
                    f"✅ Detected partition files, using partition_prefix='{partition_prefix}'"
                )
            self._diskannpy = diskannpy
            self._current_zmq_port = None
            self._index = None
            logger.debug("DiskANN searcher initialized (index will be loaded on first search)")

    def close(self):
        """Release the C++ index object and its file handles.

        On Windows, open memory-mapped files prevent temp directory cleanup.
        Call this (or use LeannSearcher as a context manager) before deleting
        the index directory.
        """
        self._index = None
        self._current_zmq_port = None
        import gc

        gc.collect()

    def _ensure_index_loaded(self, zmq_port: int):
        """Ensure the index is loaded with the correct zmq_port."""
        if self._index is None or self._current_zmq_port != zmq_port:
            # Need to (re)load the index with the correct zmq_port
            with suppress_cpp_output_if_needed():
                if self._index is not None:
                    logger.debug(f"Reloading DiskANN index with new zmq_port: {zmq_port}")
                else:
                    logger.debug(f"Loading DiskANN index with zmq_port: {zmq_port}")

                self._index = self._diskannpy.StaticDiskFloatIndex(
                    self._init_params["metric_enum"],
                    self._init_params["full_index_prefix"],
                    self._init_params["num_threads"],
                    self._init_params["num_nodes_to_cache"],
                    self._init_params["cache_mechanism"],
                    zmq_port,
                    self._init_params["pq_prefix"],
                    self._init_params["partition_prefix"],
                )
                self._current_zmq_port = zmq_port

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
        batch_recompute: bool = False,
        dedup_node_dis: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Search for nearest neighbors using DiskANN index.

        Args:
            query: Query vectors (B, D) where B is batch size, D is dimension
            top_k: Number of nearest neighbors to return
            complexity: Search complexity/candidate list size, higher = more accurate but slower
            beam_width: Number of parallel IO requests per iteration
            prune_ratio: Ratio of neighbors to prune via approximate distance (0.0-1.0)
            recompute_embeddings: Whether to fetch fresh embeddings from server
            pruning_strategy: PQ candidate selection strategy:
                - "global": Use global pruning strategy (default)
                - "local": Use local pruning strategy
                - "proportional": Not supported in DiskANN, falls back to global
            zmq_port: ZMQ port for embedding server communication. Must be provided if recompute_embeddings is True.
            batch_recompute: Whether to batch neighbor recomputation (DiskANN-specific)
            dedup_node_dis: Whether to cache and reuse distance computations (DiskANN-specific)
            **kwargs: Additional DiskANN-specific parameters (for legacy compatibility)

        Returns:
            Dict with 'labels' (list of lists) and 'distances' (ndarray)
        """
        # Handle zmq_port compatibility: Ensure index is loaded with correct port
        if recompute_embeddings:
            if zmq_port is None:
                raise ValueError("zmq_port must be provided if recompute_embeddings is True")
            self._ensure_index_loaded(zmq_port)
        else:
            # If not recomputing, we still need an index, use a default port
            if self._index is None:
                self._ensure_index_loaded(6666)  # Default port when not recomputing

        # DiskANN doesn't support "proportional" strategy
        if pruning_strategy == "proportional":
            raise NotImplementedError(
                "DiskANN backend does not support 'proportional' pruning strategy. Use 'global' or 'local' instead."
            )

        if query.dtype != np.float32:
            query = query.astype(np.float32)

        # Map pruning_strategy to DiskANN's global_pruning parameter
        if pruning_strategy == "local":
            use_global_pruning = False
        else:  # "global"
            use_global_pruning = True

        # Strategy:
        # - Traversal always uses PQ distances
        # - If recompute_embeddings=True, do a single final rerank via deferred fetch
        #   (fetch embeddings for the final candidate set only)
        # - Do not recompute neighbor distances along the path
        use_deferred_fetch = True if recompute_embeddings else False
        recompute_neighors = False  # Expected typo. For backward compatibility.

        with suppress_cpp_output_if_needed():
            labels, distances = self._index.batch_search(
                query,
                query.shape[0],
                top_k,
                complexity,
                beam_width,
                self.num_threads,
                use_deferred_fetch,
                kwargs.get("skip_search_reorder", False),
                recompute_neighors,
                dedup_node_dis,
                prune_ratio,
                batch_recompute,
                use_global_pruning,
            )

        string_labels = [[str(int_label) for int_label in batch_labels] for batch_labels in labels]

        return {"labels": string_labels, "distances": distances}
