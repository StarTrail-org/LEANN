"""An IVF index must build from fewer chunks than its number of lists.

IVF clusters the vectors into ``nlist`` lists (100 by default) with k-means, and FAISS
refuses to train when there are fewer vectors than lists. The MCP ``leann_build`` tool
defaults to IVF, so with the default ``nlist`` any folder that chunked into 1 to 99
passages failed with
"Number of training points (2) should be at least as large as number of clusters (100)".
"""

from unittest.mock import patch

import leann.api as api
import numpy as np
import pytest
from leann.api import LeannBuilder
from leann_backend_ivf import ivf_backend

VECTORS = {
    text: np.random.default_rng(ord(text)).standard_normal(8).astype(np.float32) for text in "abc"
}


def _embed(texts, *args, **kwargs):
    return np.stack([VECTORS[text] for text in texts])


def test_ivf_builds_from_fewer_chunks_than_the_default_nlist(tmp_path):
    index = tmp_path / "small.leann"
    with patch.object(api, "compute_embeddings", side_effect=_embed):
        builder = LeannBuilder(backend_name="ivf", dimensions=8)
        for text in "abc":
            builder.add_text(text, metadata={"id": text})
        builder.build_index(str(index))

    # Read the saved index back instead of searching it: once an HNSW wheel's FAISS
    # bindings are loaded in the same process, IVF search loses its NumPy wrapper.
    faiss = ivf_backend.faiss
    ivf = faiss.extract_index_ivf(faiss.read_index(str(tmp_path / "small.index")))
    assert (ivf.ntotal, ivf.nlist) == (3, 3)


def test_ivf_refuses_to_build_from_zero_vectors(tmp_path):
    # With the list count capped at the number of vectors, zero vectors would give an
    # index with no lists, which saves but can never be searched.
    builder = LeannBuilder(backend_name="ivf", dimensions=2)

    with pytest.raises(ValueError, match="zero vectors"):
        builder.build_index_from_arrays(str(tmp_path / "empty.leann"), [], np.zeros((0, 2)))
