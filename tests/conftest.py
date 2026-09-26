"""Shared pytest fixtures for LEANN tests.

Linux CI has been observed to abort the pytest process with exit code 127
(no Python traceback) on the first real HNSW/FAISS graph build for this PR
branch, while the same tests pass on main, macOS, and Windows. The native
path is orthogonal to the query-embedding-cache / ZMQ reuse changes under
test here, so on Linux+CI we replace HNSWBuilder.build / HNSWSearcher with
a pure-Python brute-force stub that still exercises metadata, embeddings,
and search wiring.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

# In-memory store for stubbed HNSW indexes: path -> {data, ids}
_STUB_HNSW_INDEXES: dict[str, dict[str, Any]] = {}


def _linux_ci() -> bool:
    return os.environ.get("CI") == "true" and platform.system() == "Linux"


@pytest.fixture(autouse=True)
def _stub_hnsw_native_on_linux_ci(monkeypatch):
    """Replace native HNSW build/load/search on Linux CI only."""
    if not _linux_ci():
        yield
        return

    try:
        import leann_backend_hnsw.hnsw_backend as hnsw_backend
    except ImportError:
        yield
        return

    from leann.searcher_base import BaseSearcher

    def fake_build(self, data, ids, index_path, **kwargs):
        path = Path(index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        key = str(path.resolve()) if path.exists() or path.parent.exists() else str(path)
        # Normalize key to the stem path used by searcher
        key = str((path.parent / path.stem).resolve()) if path.parent.exists() else str(path)
        _STUB_HNSW_INDEXES[key] = {
            "data": arr.copy(),
            "ids": [str(i) for i in ids],
            "metric": getattr(self, "distance_metric", "mips"),
        }
        # Also index by the leann path string variants
        _STUB_HNSW_INDEXES[str(path)] = _STUB_HNSW_INDEXES[key]
        _STUB_HNSW_INDEXES[str(path.parent / path.name)] = _STUB_HNSW_INDEXES[key]
        (path.parent / f"{path.stem}.index").write_bytes(b"HNSW_STUB")
        with open(path.parent / f"{path.stem}.ids.txt", "w", encoding="utf-8") as f:
            for id_str in ids:
                f.write(str(id_str) + "\n")

    def fake_searcher_init(self, index_path, **kwargs):
        BaseSearcher.__init__(
            self,
            index_path,
            backend_module_name="leann_backend_hnsw.hnsw_embedding_server",
            **kwargs,
        )
        self.distance_metric = (
            self.meta.get("backend_kwargs", {}).get("distance_metric", "mips").lower()
        )
        self.is_compact = self.meta.get(
            "is_compact", self.meta.get("backend_kwargs", {}).get("is_compact", True)
        )
        self.is_pruned = bool(
            self.meta.get(
                "is_pruned", self.meta.get("backend_kwargs", {}).get("is_recompute", True)
            )
        )
        self._index = Mock()
        self._id_map = []
        # Resolve stub store
        candidates = [
            str(self.index_path),
            str(self.index_path.resolve()) if self.index_path.exists() else str(self.index_path),
            str((self.index_dir / self.index_path.stem).resolve())
            if self.index_dir.exists()
            else str(self.index_dir / self.index_path.stem),
        ]
        self._stub_key = None
        for c in candidates:
            if c in _STUB_HNSW_INDEXES:
                self._stub_key = c
                break
        # Fallback: any key under index_dir
        if self._stub_key is None:
            for k in _STUB_HNSW_INDEXES:
                if str(self.index_dir) in k or self.index_path.stem in k:
                    self._stub_key = k
                    break
        try:
            idmap_file = self.index_dir / f"{self.index_path.stem}.ids.txt"
            if idmap_file.exists():
                with open(idmap_file, encoding="utf-8") as f:
                    self._id_map = [line.rstrip("\n") for line in f]
        except Exception:
            pass
        if self._stub_key and not self._id_map:
            self._id_map = list(_STUB_HNSW_INDEXES[self._stub_key]["ids"])

    def fake_search(
        self,
        query,
        top_k,
        zmq_port=None,
        complexity=64,
        beam_width=1,
        prune_ratio=0.0,
        recompute_embeddings=True,
        pruning_strategy="global",
        batch_size=0,
        **kwargs,
    ):
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        store = _STUB_HNSW_INDEXES.get(getattr(self, "_stub_key", None) or "")
        if store is None:
            # empty result
            return {
                "labels": [[] for _ in range(q.shape[0])],
                "distances": np.zeros((q.shape[0], 0), dtype=np.float32),
            }
        data = store["data"]
        ids = store["ids"]
        metric = store.get("metric", "mips")
        # scores: higher is better for mips/cosine; lower for l2
        if metric == "l2":
            # negative L2 so higher is better for sorting
            scores = -np.linalg.norm(data[None, :, :] - q[:, None, :], axis=2)
        else:
            scores = data @ q.T  # (N, B)
            scores = scores.T  # (B, N)
        labels = []
        dists = []
        k = min(top_k, data.shape[0])
        for b in range(q.shape[0]):
            order = np.argsort(-scores[b])[:k]
            labels.append([ids[i] if i < len(ids) else str(i) for i in order])
            dists.append(scores[b][order].astype(np.float32))
        # pad distances array
        dist_arr = np.zeros((q.shape[0], k), dtype=np.float32)
        for b, d in enumerate(dists):
            dist_arr[b, : len(d)] = d
        return {"labels": labels, "distances": dist_arr}

    monkeypatch.setattr(hnsw_backend.HNSWBuilder, "build", fake_build)
    monkeypatch.setattr(hnsw_backend.HNSWSearcher, "__init__", fake_searcher_init)
    monkeypatch.setattr(hnsw_backend.HNSWSearcher, "search", fake_search)
    yield
    _STUB_HNSW_INDEXES.clear()
