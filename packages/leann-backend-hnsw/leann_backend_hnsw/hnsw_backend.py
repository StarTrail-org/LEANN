import logging
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from leann.interface import (
    LeannBackendBuilderInterface,
    LeannBackendFactoryInterface,
    LeannBackendSearcherInterface,
)
from leann.registry import register_backend
from leann.searcher_base import BaseSearcher

from .convert_to_csr import convert_hnsw_graph_to_csr, prune_hnsw_embeddings_inplace

logger = logging.getLogger(__name__)


def get_metric_map():
    from . import faiss  # type: ignore

    return {
        "mips": faiss.METRIC_INNER_PRODUCT,
        "l2": faiss.METRIC_L2,
        "cosine": faiss.METRIC_INNER_PRODUCT,
    }


def normalize_l2(data: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return data / norms


@register_backend("hnsw")
class HNSWBackend(LeannBackendFactoryInterface):
    @staticmethod
    def builder(**kwargs) -> LeannBackendBuilderInterface:
        return HNSWBuilder(**kwargs)

    @staticmethod
    def searcher(index_path: str, **kwargs) -> LeannBackendSearcherInterface:
        return HNSWSearcher(index_path, **kwargs)


class HNSWBuilder(LeannBackendBuilderInterface):
    def __init__(self, **kwargs):
        self.build_params = kwargs.copy()
        self.is_compact = self.build_params.setdefault("is_compact", True)
        self.is_recompute = self.build_params.setdefault("is_recompute", True)
        # graph_degree/complexity are the backend-neutral names that `leann build` and the
        # apps pass; FAISS's own M/efConstruction take precedence when given explicitly.
        self.M = self.build_params.setdefault("M", self.build_params.get("graph_degree", 32))
        self.efConstruction = self.build_params.setdefault(
            "efConstruction", self.build_params.get("complexity", 200)
        )
        # Graph construction makes no ZMQ calls, so on Linux it uses every core by default
        # even though leann pins OpenMP to one thread there for recompute searches (#208).
        # Elsewhere the process-wide OpenMP setting applies unless num_threads is given.
        self.num_threads = self.build_params.get("num_threads") or (
            os.cpu_count() if platform.system() == "Linux" else None
        )
        self.distance_metric = self.build_params.setdefault("distance_metric", "mips")
        self.dimensions = self.build_params.get("dimensions")
        if not self.is_recompute and self.is_compact:
            # Auto-correct: non-recompute requires non-compact storage for HNSW
            logger.warning(
                "is_recompute=False requires non-compact HNSW. Forcing is_compact=False."
            )
            self.is_compact = False
            self.build_params["is_compact"] = False

    def build(self, data: np.ndarray, ids: list[str], index_path: str, **kwargs):
        from . import faiss  # type: ignore

        path = Path(index_path)
        index_dir = path.parent
        index_prefix = path.stem
        index_dir.mkdir(parents=True, exist_ok=True)

        if data.dtype != np.float32:
            logger.warning(f"Converting data to float32, shape: {data.shape}")
            data = data.astype(np.float32)

        metric_enum = get_metric_map().get(self.distance_metric.lower())
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{self.distance_metric}'.")

        dim = self.dimensions or data.shape[1]
        index = faiss.IndexHNSWFlat(dim, self.M, metric_enum)
        index.hnsw.efConstruction = self.efConstruction

        if self.distance_metric.lower() == "cosine":
            data = normalize_l2(data)

        omp_threads = faiss.omp_get_max_threads()
        if self.num_threads:
            faiss.omp_set_num_threads(int(self.num_threads))
        try:
            index.add(data.shape[0], faiss.swig_ptr(data))
        finally:
            faiss.omp_set_num_threads(omp_threads)
        index_file = index_dir / f"{index_prefix}.index"
        faiss.write_index(index, str(index_file))

        # Persist ID map so searcher can map FAISS integer labels back to passage IDs
        try:
            idmap_file = index_dir / f"{index_prefix}.ids.txt"
            with open(idmap_file, "w", encoding="utf-8") as f:
                for id_str in ids:
                    f.write(str(id_str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write ID map: {e}")

        if self.is_compact:
            self._convert_to_csr(index_file)
        elif self.is_recompute:
            prune_hnsw_embeddings_inplace(str(index_file))

    def _convert_to_csr(self, index_file: Path):
        """Convert built index to CSR format"""
        mode_str = "CSR-pruned" if self.is_recompute else "CSR-standard"
        logger.info(f"INFO: Converting HNSW index to {mode_str} format...")

        csr_temp_file = index_file.with_suffix(".csr.tmp")

        success = convert_hnsw_graph_to_csr(
            str(index_file), str(csr_temp_file), prune_embeddings=self.is_recompute
        )

        if success:
            logger.info("✅ CSR conversion successful.")
            # index_file_old = index_file.with_suffix(".old")
            # shutil.move(str(index_file), str(index_file_old))
            shutil.move(str(csr_temp_file), str(index_file))
            logger.info(f"INFO: Replaced original index with {mode_str} version at '{index_file}'")
        else:
            # Clean up and fail fast
            if csr_temp_file.exists():
                os.remove(csr_temp_file)
            raise RuntimeError("CSR conversion failed - cannot proceed with compact format")


class HNSWSearcher(BaseSearcher):
    def __init__(self, index_path: str, **kwargs):
        super().__init__(
            index_path,
            backend_module_name="leann_backend_hnsw.hnsw_embedding_server",
            **kwargs,
        )
        from . import faiss  # type: ignore

        self.distance_metric = (
            self.meta.get("backend_kwargs", {}).get("distance_metric", "mips").lower()
        )
        metric_enum = get_metric_map().get(self.distance_metric)
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{self.distance_metric}'.")

        backend_meta_kwargs = self.meta.get("backend_kwargs", {})
        self.is_compact = self.meta.get("is_compact", backend_meta_kwargs.get("is_compact", True))
        default_pruned = backend_meta_kwargs.get("is_recompute", self.is_compact)
        self.is_pruned = bool(self.meta.get("is_pruned", default_pruned))

        index_file = self.index_dir / f"{self.index_path.stem}.index"
        if not index_file.exists():
            raise FileNotFoundError(f"HNSW index file not found at {index_file}")

        hnsw_config = faiss.HNSWIndexConfig()
        hnsw_config.is_compact = self.is_compact
        hnsw_config.is_recompute = (
            self.is_pruned
        )  # In C++ code, it's called is_recompute, but it's only for loading IIUC.

        self._index = faiss.read_index(str(index_file), faiss.IO_FLAG_MMAP, hnsw_config)

        # Load ID map if available
        self._id_map: list[str] = []
        try:
            idmap_file = self.index_dir / f"{self.index_path.stem}.ids.txt"
            if idmap_file.exists():
                with open(idmap_file, encoding="utf-8") as f:
                    self._id_map = [line.rstrip("\n") for line in f]
        except Exception as e:
            logger.warning(f"Failed to load ID map: {e}")

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        zmq_port: Optional[int] = None,
        complexity: int = 64,
        beam_width: int = 1,
        prune_ratio: float = 0.0,
        recompute_embeddings: bool = True,
        pruning_strategy: Literal["global", "local", "proportional"] = "global",
        batch_size: int = 0,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Search for nearest neighbors using HNSW index.

        Args:
            query: Query vectors (B, D) where B is batch size, D is dimension
            top_k: Number of nearest neighbors to return
            complexity: Search complexity/efSearch, higher = more accurate but slower
            beam_width: Number of parallel search paths/beam_size
            prune_ratio: Ratio of neighbors to prune via PQ (0.0-1.0)
            recompute_embeddings: Whether to fetch fresh embeddings from server
            pruning_strategy: PQ candidate selection strategy:
                - "global": Use global PQ queue size for selection (default)
                - "local": Local pruning, sort and select best candidates
                - "proportional": Base selection on new neighbor count ratio
            zmq_port: ZMQ port for embedding server communication. Must be provided if recompute_embeddings is True.
            batch_size: Neighbor processing batch size, 0=disabled (HNSW-specific)
            **kwargs: Additional HNSW-specific parameters (for legacy compatibility)

        Returns:
            Dict with 'labels' (list of lists) and 'distances' (ndarray)
        """
        from . import faiss  # type: ignore

        if not recompute_embeddings and self.is_pruned:
            raise RuntimeError(
                "Recompute is required for pruned/compact HNSW index. "
                "Re-run search with --recompute, or rebuild with --no-recompute and --no-compact."
            )
        if recompute_embeddings:
            if zmq_port is None:
                raise ValueError("zmq_port must be provided if recompute_embeddings is True")
            if hasattr(self._index, "set_zmq_port"):
                self._index.set_zmq_port(zmq_port)

        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if self.distance_metric == "cosine":
            query = normalize_l2(query)

        params = faiss.SearchParametersHNSW()
        if zmq_port is not None:
            params.zmq_port = zmq_port  # C++ code won't use this if recompute_embeddings is False
        params.efSearch = complexity
        params.beam_size = beam_width

        # For OpenAI embeddings with cosine distance, disable relative distance check
        # This prevents early termination when all scores are in a narrow range
        embedding_model = self.meta.get("embedding_model", "").lower()
        if self.distance_metric == "cosine" and any(
            openai_model in embedding_model for openai_model in ["text-embedding", "openai"]
        ):
            params.check_relative_distance = False
        else:
            params.check_relative_distance = True

        # PQ pruning: direct mapping to HNSW's pq_pruning_ratio
        params.pq_pruning_ratio = prune_ratio

        # Map pruning_strategy to HNSW parameters
        if pruning_strategy == "local":
            params.local_prune = True
            params.send_neigh_times_ratio = 0.0
        elif pruning_strategy == "proportional":
            params.local_prune = False
            params.send_neigh_times_ratio = 1.0  # Any value > 1e-6 triggers proportional mode
        else:  # "global"
            params.local_prune = False
            params.send_neigh_times_ratio = 0.0

        # HNSW-specific batch processing parameter
        params.batch_size = batch_size

        batch_size_query = query.shape[0]
        distances = np.empty((batch_size_query, top_k), dtype=np.float32)
        labels = np.empty((batch_size_query, top_k), dtype=np.int64)

        search_time = time.time()
        self._index.search(
            query.shape[0],
            faiss.swig_ptr(query),
            top_k,
            faiss.swig_ptr(distances),
            faiss.swig_ptr(labels),
            params,
        )
        search_time = time.time() - search_time
        logger.info(f"  Search time in HNSWSearcher.search() backend: {search_time} seconds")
        if self._id_map:

            def map_label(x: int) -> str:
                if 0 <= x < len(self._id_map):
                    return self._id_map[x]
                return str(x)

            string_labels = [
                [map_label(int(label)) for label in batch_labels] for batch_labels in labels
            ]
        else:
            string_labels = [
                [str(int_label) for int_label in batch_labels] for batch_labels in labels
            ]

        return {"labels": string_labels, "distances": distances}
