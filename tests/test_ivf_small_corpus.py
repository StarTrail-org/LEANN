"""An IVF index must build from fewer chunks than its number of lists.

IVF clusters the vectors into ``nlist`` lists (100 by default) with k-means, and FAISS
refuses to train when there are fewer vectors than lists. The MCP ``leann_build`` tool
defaults to IVF, so with the default ``nlist`` any folder that chunked into 1 to 99
passages failed with
"Number of training points (2) should be at least as large as number of clusters (100)".
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
from leann.api import LeannBuilder

# Runs in a fresh interpreter. Once an HNSW wheel's FAISS bindings are loaded, which other
# tests in the same pytest process do, FAISS objects that IVF reads back come through those
# bindings and are unreliable, so this process must load only the IVF backend's FAISS.
BUILD_AND_SEARCH = textwrap.dedent(
    """
    import sys
    from unittest.mock import patch

    import leann.api as api
    import numpy as np
    from leann.api import LeannBuilder
    from leann_backend_ivf.ivf_backend import IVFSearcher

    vectors = {
        t: np.random.default_rng(ord(t)).standard_normal(8).astype(np.float32) for t in "abc"
    }
    embed = lambda texts, *args, **kwargs: np.stack([vectors[t] for t in texts])
    with patch.object(api, "compute_embeddings", side_effect=embed):
        builder = LeannBuilder(backend_name="ivf", dimensions=8)
        for text in "abc":
            builder.add_text(text, metadata={"id": text})
        builder.build_index(sys.argv[1])

    result = IVFSearcher(sys.argv[1]).search(vectors["b"][None, :], top_k=3)
    print("LABELS=" + ",".join(result["labels"][0]))
    """
)


def test_ivf_builds_and_searches_fewer_chunks_than_the_default_nlist(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", BUILD_AND_SEARCH, str(tmp_path / "small.leann")],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    labels = result.stdout.split("LABELS=")[-1].split()[0].split(",")
    assert labels[0] == "b"
    assert sorted(labels) == ["a", "b", "c"]


def test_ivf_refuses_to_build_from_zero_vectors(tmp_path):
    # With the list count capped at the number of vectors, zero vectors would give an
    # index with no lists, which saves but can never be searched.
    builder = LeannBuilder(backend_name="ivf", dimensions=2)

    with pytest.raises(ValueError, match="zero vectors"):
        builder.build_index_from_arrays(str(tmp_path / "empty.leann"), [], np.zeros((0, 2)))
