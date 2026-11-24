"""
IVF backend: FAISS IndexIVFFlat with DirectMap.Hashtable for add/remove by passage id.

Uses IndexIVFFlat + DirectMap.Hashtable (no IndexIDMap2) so remove_ids works correctly.
Provides add_vectors() and remove_ids() for incremental updates.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]

from leann.interface import (
    LeannBackendBuilderInterface,
    LeannBackendFactoryInterface,
    LeannBackendSearcherInterface,
)
from leann.registry import register_backend
from leann.searcher_base import BaseSearcher

logger = logging.getLogger(__name__)

ID_MAP_FILENAME = "ivf_id_map.json"


def _check_faiss():
    if faiss is None:
        raise ImportError(
            "faiss-cpu is required for IVF backend. Install with: pip install faiss-cpu"
        )


def _get_metric_map():
    _check_faiss()
    return {
        "mips": faiss.METRIC_INNER_PRODUCT,
        "l2": faiss.METRIC_L2,
        "cosine": faiss.METRIC_INNER_PRODUCT,
    }


def _normalize_l2(data: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return data / norms


def _load_id_map(index_dir: Path, index_prefix: str) -> tuple[dict[int, str], dict[str, int], int]:
    """Load id_map.json. Returns (id_to_passage, passage_to_id, next_id)."""
    path = index_dir / f"{index_prefix}.{ID_MAP_FILENAME}"
    if not path.exists():
        return {}, {}, 0
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    id_to_passage = {int(k): v for k, v in data.get("id_to_passage", {}).items()}
    passage_to_id = data.get("passage_to_id", {})
    next_id = int(data.get("next_id", 0))
    return id_to_passage, passage_to_id, next_id


def _save_id_map(
    index_dir: Path, index_prefix: str, id_to_passage: dict[int, str], next_id: int
) -> None:
    passage_to_id = {v: k for k, v in id_to_passage.items()}
    path = index_dir / f"{index_prefix}.{ID_MAP_FILENAME}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "id_to_passage": {str(k): v for k, v in id_to_passage.items()},
                "passage_to_id": passage_to_id,
                "next_id": next_id,
            },
            f,
            indent=2,
        )


@register_backend("ivf")
class IVFBackend(LeannBackendFactoryInterface):
    @staticmethod
    def builder(**kwargs) -> LeannBackendBuilderInterface:
        return IVFBuilder(**kwargs)

    @staticmethod
    def searcher(index_path: str, **kwargs) -> LeannBackendSearcherInterface:
        return IVFSearcher(index_path, **kwargs)


class IVFBuilder(LeannBackendBuilderInterface):
    def __init__(self, **kwargs):
        _check_faiss()
        self.build_params = kwargs.copy()
        self.nlist = self.build_params.setdefault("nlist", 100)
        self.distance_metric = self.build_params.setdefault("distance_metric", "l2")
        self.dimensions = self.build_params.get("dimensions")

    def build(self, data: np.ndarray, ids: list[str], index_path: str, **kwargs) -> None:
        _check_faiss()
        path = Path(index_path)
        index_dir = path.parent
        index_prefix = path.stem
        index_dir.mkdir(parents=True, exist_ok=True)

        if data.dtype != np.float32:
            data = data.astype(np.float32)
        data = np.ascontiguousarray(data)
        dim = self.dimensions or data.shape[1]
        n = data.shape[0]
        if n == 0:
            raise ValueError("Cannot build an IVF index from zero vectors.")
        metric_enum = _get_metric_map().get(self.distance_metric.lower())
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{self.distance_metric}'.")

        if self.distance_metric.lower() == "cosine":
            data = _normalize_l2(data)

        quantizer = (
            faiss.IndexFlatL2(dim) if metric_enum == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)
        )
        # k-means needs at least one vector per list, and a small folder can chunk into
        # fewer passages than the default 100 lists.
        ivf = faiss.IndexIVFFlat(quantizer, dim, min(self.nlist, n), metric_enum)
        ivf.train(data)
        ivf.set_direct_map_type(faiss.DirectMap.Hashtable)
        faiss_ids = np.arange(n, dtype=np.int64)
        ivf.add_with_ids(data, faiss_ids)

        index_file = index_dir / f"{index_prefix}.index"
        faiss.write_index(ivf, str(index_file))

        id_to_passage = dict(enumerate(ids))
        _save_id_map(index_dir, index_prefix, id_to_passage, next_id=n)


class IVFSearcher(BaseSearcher):
    def __init__(self, index_path: str, **kwargs):
        # Use HNSW embedding server for query embedding if available (same as other backends)
        super().__init__(
            index_path,
            backend_module_name="leann_backend_hnsw.hnsw_embedding_server",
            **kwargs,
        )
        _check_faiss()
        self.distance_metric = (
            self.meta.get("backend_kwargs", {}).get("distance_metric", "l2").lower()
        )
        index_prefix = self.index_path.stem
        index_file = self.index_dir / f"{index_prefix}.index"
        if not index_file.exists():
            raise FileNotFoundError(f"IVF index file not found at {index_file}")

        self._index = faiss.read_index(str(index_file))
        self._id_to_passage: dict[int, str] = {}
        id_to_passage, _, _ = _load_id_map(self.index_dir, index_prefix)
        self._id_to_passage = id_to_passage

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        complexity: int = 64,
        nprobe: Optional[int] = None,
        **kwargs,
    ) -> dict[str, Any]:
        _check_faiss()
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if self.distance_metric == "cosine":
            query = _normalize_l2(query)
        ivf_index = faiss.extract_index_ivf(self._index)
        nprobe = nprobe or min(complexity, ivf_index.nlist)
        ivf_index.nprobe = nprobe
        distances, label_rows = self._index.search(query, top_k)

        def map_label(x: int) -> str:
            return self._id_to_passage.get(int(x), str(x))

        string_labels = [[map_label(int(lab)) for lab in row] for row in label_rows]
        return {"labels": string_labels, "distances": distances}

    def compute_query_embedding(
        self,
        query: str,
        use_server_if_available: bool = True,
        zmq_port: Optional[int] = None,
        query_template: Optional[str] = None,
    ) -> np.ndarray:
        return super().compute_query_embedding(
            query,
            use_server_if_available=use_server_if_available,
            zmq_port=zmq_port,
            query_template=query_template,
        )


def add_vectors(index_path: str, embeddings: np.ndarray, passage_ids: list[str]) -> None:
    """
    Append vectors to an existing IVF index (same role as HNSW update_index add path).

    Args:
        index_path: Path to the .leann index (e.g. .../documents.leann).
        embeddings: (N, D) float32 array.
        passage_ids: List of N passage id strings (must not already exist in index).
    """
    _check_faiss()
    path = Path(index_path)
    index_dir = path.parent
    index_prefix = path.stem
    index_file = index_dir / f"{index_prefix}.index"
    if not index_file.exists():
        raise FileNotFoundError(f"IVF index not found: {index_file}")

    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    id_to_passage, passage_to_id, next_id = _load_id_map(index_dir, index_prefix)
    for pid in passage_ids:
        if pid in passage_to_id:
            raise ValueError(f"Passage id '{pid}' already exists in index.")

    n = embeddings.shape[0]
    if n != len(passage_ids):
        raise ValueError("embeddings.shape[0] must equal len(passage_ids).")

    index = faiss.read_index(str(index_file))
    new_ids = np.arange(next_id, next_id + n, dtype=np.int64)
    index.add_with_ids(embeddings, new_ids)
    faiss.write_index(index, str(index_file))

    for i, pid in enumerate(passage_ids):
        id_to_passage[next_id + i] = pid
    _save_id_map(index_dir, index_prefix, id_to_passage, next_id=next_id + n)
    logger.info("IVF add_vectors: appended %d vectors, next_id=%d", n, next_id + n)


def remove_ids(index_path: str, passage_ids: list[str]) -> int:
    """
    Remove vectors by passage id (for incremental update: delete changed chunks before re-insert).

    Args:
        index_path: Path to the .leann index.
        passage_ids: List of passage id strings to remove.

    Returns:
        Number of vectors actually removed.
    """
    _check_faiss()
    path = Path(index_path)
    index_dir = path.parent
    index_prefix = path.stem
    index_file = index_dir / f"{index_prefix}.index"
    if not index_file.exists():
        raise FileNotFoundError(f"IVF index not found: {index_file}")

    id_to_passage, passage_to_id, next_id = _load_id_map(index_dir, index_prefix)
    to_remove_int: list[int] = []
    for pid in passage_ids:
        if pid in passage_to_id:
            to_remove_int.append(passage_to_id[pid])
    if not to_remove_int:
        return 0

    index = faiss.read_index(str(index_file))
    ntotal_before = index.ntotal
    sel = np.array(to_remove_int, dtype=np.int64)
    nremoved = index.remove_ids(sel)
    faiss.write_index(index, str(index_file))

    for pid in passage_ids:
        if pid in passage_to_id:
            i = passage_to_id[pid]
            id_to_passage.pop(i, None)
    # passage_to_id is rebuilt from id_to_passage when we save; we don't decrease next_id so new adds get new ids
    _save_id_map(index_dir, index_prefix, id_to_passage, next_id=next_id)
    logger.info(
        "IVF remove_ids: ntotal %d -> %d, removed %d vectors (requested %d, found %d in id_map)",
        ntotal_before,
        index.ntotal,
        nremoved,
        len(passage_ids),
        len(to_remove_int),
    )
    if nremoved != len(to_remove_int):
        logger.warning(
            "IVF remove_ids: FAISS removed %d but expected %d. Possible index inconsistency.",
            nremoved,
            len(to_remove_int),
        )
    return nremoved
