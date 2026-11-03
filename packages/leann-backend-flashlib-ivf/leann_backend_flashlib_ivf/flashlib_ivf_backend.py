"""
FlashLib IVF backend: GPU-accelerated IVF-Flat (inverted file) ANN search.

FlashLib (https://github.com/FlashML-org/flashlib) is a GPU library of classical
ML primitives. This backend uses its IVF-Flat index, which runs an *approximate*
nearest-neighbor search entirely on CUDA tensors:

    from flashlib import flash_ivf_flat_build, flash_ivf_flat_search
    index = flash_ivf_flat_build(db_cuda, nlist, nprobe=..., niter=...)   # (M, D) CUDA
    vals, ids = flash_ivf_flat_search(index, queries_cuda, k, nprobe=...)

This is the GPU counterpart of the FAISS ``ivf`` backend (FAISS ``IndexIVFFlat``):
both coarse-quantize the corpus into ``nlist`` cells and, at search time, scan only
the ``nprobe`` nearest cells. At a fixed ``(nlist, nprobe)`` FlashLib probes the same
candidate set as a reference IVF-Flat, so recall is comparable; the difference is GPU
vs CPU kernels.

The built index is a small set of torch tensors (centroids, cell-contiguous data,
row ids, CSR offsets), so we persist it with ``torch.save`` (``<index>.flashlib_ivf.pt``)
plus an id map (``<index>.flashlib_ivf_id_map.json``) and reload it onto the GPU at
searcher start-up (no k-means re-train).

FlashLib ranks by squared L2. For ``mips`` / ``cosine`` we L2-normalize both the
database and query vectors, on which squared-L2 ranking is equivalent to
inner-product / cosine ranking.

Requires a CUDA GPU at both build (k-means training) and search time.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from leann.interface import (
    LeannBackendBuilderInterface,
    LeannBackendFactoryInterface,
    LeannBackendSearcherInterface,
)
from leann.registry import register_backend
from leann.searcher_base import BaseSearcher

logger = logging.getLogger(__name__)

INDEX_SUFFIX = "flashlib_ivf.pt"
ID_MAP_SUFFIX = "flashlib_ivf_id_map.json"

# IvfFlatIndex dataclass fields, split by how they serialize.
_TENSOR_FIELDS = ("centroids", "data", "ids", "list_offsets")
_SCALAR_FIELDS = ("metric", "D", "Dp", "nlist", "nprobe", "max_list_len")


def _import_flashlib():
    try:
        import torch  # noqa: F401
        from flashlib import (
            IvfFlatIndex,
            flash_ivf_flat_build,
            flash_ivf_flat_search,
        )
    except ImportError as e:
        raise ImportError(
            "The FlashLib IVF backend requires 'flashlib' and 'torch' with CUDA. "
            "Install with: pip install flashlib (a CUDA GPU is required at build and search time)."
        ) from e
    return IvfFlatIndex, flash_ivf_flat_build, flash_ivf_flat_search


def _normalize_l2(data: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return data / norms


def _needs_normalize(distance_metric: str) -> bool:
    return distance_metric.lower() in ("mips", "cosine")


def _index_path(index_dir: Path, index_prefix: str) -> Path:
    return index_dir / f"{index_prefix}.{INDEX_SUFFIX}"


def _id_map_path(index_dir: Path, index_prefix: str) -> Path:
    return index_dir / f"{index_prefix}.{ID_MAP_SUFFIX}"


def _save_id_map(index_dir: Path, index_prefix: str, ids: list[str]) -> None:
    with open(_id_map_path(index_dir, index_prefix), "w", encoding="utf-8") as f:
        json.dump({"ids": ids}, f)


def _load_id_map(index_dir: Path, index_prefix: str) -> list[str]:
    path = _id_map_path(index_dir, index_prefix)
    if not path.exists():
        raise FileNotFoundError(f"FlashLib IVF id map not found at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["ids"]


def _save_index(index, path: Path) -> None:
    import torch

    state: dict[str, Any] = {f: getattr(index, f).detach().cpu() for f in _TENSOR_FIELDS}
    for f in _SCALAR_FIELDS:
        state[f] = getattr(index, f)
    torch.save(state, str(path))


def _load_index(path: Path, device: str = "cuda"):
    import torch

    IvfFlatIndex, _, _ = _import_flashlib()
    state = torch.load(str(path), map_location=device, weights_only=True)
    kwargs: dict[str, Any] = {f: state[f].to(device) for f in _TENSOR_FIELDS}
    for f in _SCALAR_FIELDS:
        kwargs[f] = state[f]
    return IvfFlatIndex(**kwargs)


@register_backend("flashlib_ivf")
class FlashlibIVFBackend(LeannBackendFactoryInterface):
    @staticmethod
    def builder(**kwargs) -> LeannBackendBuilderInterface:
        return FlashlibIVFBuilder(**kwargs)

    @staticmethod
    def searcher(index_path: str, **kwargs) -> LeannBackendSearcherInterface:
        return FlashlibIVFSearcher(index_path, **kwargs)


class FlashlibIVFBuilder(LeannBackendBuilderInterface):
    def __init__(self, **kwargs):
        self.build_params = kwargs.copy()
        self.distance_metric = self.build_params.setdefault("distance_metric", "mips")
        self.nlist = self.build_params.setdefault("nlist", 1024)
        self.nprobe = self.build_params.setdefault("nprobe", 16)
        self.niter = self.build_params.setdefault("niter", 20)
        self.seed = self.build_params.setdefault("seed", 0)
        self.dimensions = self.build_params.get("dimensions")

    def build(self, data: np.ndarray, ids: list[str], index_path: str, **kwargs) -> None:
        import torch

        _, flash_ivf_flat_build, _ = _import_flashlib()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FlashLib IVF backend requires a CUDA GPU at build time (k-means training), "
                "but none is available."
            )

        path = Path(index_path)
        index_dir = path.parent
        index_prefix = path.stem
        index_dir.mkdir(parents=True, exist_ok=True)

        if data.dtype != np.float32:
            data = data.astype(np.float32)
        data = np.ascontiguousarray(data)
        if _needs_normalize(self.distance_metric):
            data = _normalize_l2(data)

        n = data.shape[0]
        nlist = int(min(self.nlist, n)) if n > 0 else int(self.nlist)

        db = torch.from_numpy(data).cuda()
        index = flash_ivf_flat_build(
            db,
            nlist,
            metric="l2",
            nprobe=int(self.nprobe),
            niter=int(self.niter),
            seed=int(self.seed),
        )
        _save_index(index, _index_path(index_dir, index_prefix))
        _save_id_map(index_dir, index_prefix, list(ids))
        logger.info(
            "FlashLib IVF build: %d vectors (dim=%d, metric=%s, nlist=%d, nprobe=%d) at %s",
            n,
            data.shape[1],
            self.distance_metric,
            nlist,
            self.nprobe,
            _index_path(index_dir, index_prefix),
        )


class FlashlibIVFSearcher(BaseSearcher):
    def __init__(self, index_path: str, **kwargs):
        # Reuse the HNSW embedding server (if present) to embed queries, exactly like
        # the other non-recompute backends; falls back to direct model loading.
        super().__init__(
            index_path,
            backend_module_name="leann_backend_hnsw.hnsw_embedding_server",
            **kwargs,
        )
        backend_kwargs = self.meta.get("backend_kwargs", {})
        self.distance_metric = backend_kwargs.get("distance_metric", "mips").lower()

        index_prefix = self.index_path.stem
        index_file = _index_path(self.index_dir, index_prefix)
        if not index_file.exists():
            raise FileNotFoundError(f"FlashLib IVF index file not found at {index_file}")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "FlashLib IVF backend requires a CUDA GPU at search time, but none is available."
            )

        self._ids = _load_id_map(self.index_dir, index_prefix)
        self._index = _load_index(index_file, device="cuda")
        self._nlist = int(self._index.nlist)
        self._ntotal = int(self._index.data.shape[0])
        logger.info(
            "FlashLib IVF searcher ready: %d vectors, nlist=%d, metric=%s",
            self._ntotal,
            self._nlist,
            self.distance_metric,
        )

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        complexity: int = 64,
        nprobe: Optional[int] = None,
        **kwargs,
    ) -> dict[str, Any]:
        import torch

        _, _, flash_ivf_flat_search = _import_flashlib()
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if _needs_normalize(self.distance_metric):
            query = _normalize_l2(query)

        # complexity is the recall knob shared with the FAISS ivf backend.
        nprobe = nprobe or min(complexity, self._nlist)
        k = min(top_k, self._ntotal)
        q = torch.from_numpy(np.ascontiguousarray(query)).cuda()
        distances, indices = flash_ivf_flat_search(self._index, q, k, nprobe=int(nprobe))
        distances_np = distances.detach().cpu().numpy().astype(np.float32)
        indices_np = indices.detach().cpu().numpy()

        def map_label(i: int) -> str:
            # flash_ivf_flat_search pads short candidate lists with -1.
            return self._ids[i] if i >= 0 else "-1"

        string_labels = [[map_label(int(i)) for i in row] for row in indices_np]
        return {"labels": string_labels, "distances": distances_np}

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
