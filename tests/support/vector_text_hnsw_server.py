"""Run the real HNSW embedding server with a model-free embedder for e2e tests.

Each text spells out its own embedding as comma-separated floats ("3,0,0,0"), so tests
control passage norms and directions exactly. Any other text (e.g. warmup) embeds to zeros.
"""

import runpy
from typing import Any
from unittest.mock import patch

import numpy as np

DIM = 4


def _vector(text: str) -> list[float]:
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        return [0.0] * DIM
    return values if len(values) == DIM else [0.0] * DIM


def compute_embeddings(texts: list[str], model_name: str, *args: Any, **kwargs: Any) -> np.ndarray:
    return np.array([_vector(t) for t in texts], dtype=np.float32)


patch("leann.embedding_compute.compute_embeddings", compute_embeddings).start()

if __name__ == "__main__":
    runpy.run_module("leann_backend_hnsw.hnsw_embedding_server", run_name="__main__")
