import json
import pickle
from pathlib import Path
from unittest.mock import Mock

import pytest
from leann import BACKEND_REGISTRY, LeannBuilder, LeannSearcher
from leann.index_offsets import load_offset_map
from leann.sync import FileSynchronizer


class _MarkerPayload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        code = f"from pathlib import Path; Path({str(self.marker)!r}).write_text('executed')"
        return exec, (code,)


def test_load_offset_map_accepts_legacy_mapping(tmp_path: Path) -> None:
    path = tmp_path / "demo.passages.idx"
    with path.open("wb") as handle:
        pickle.dump({"passage-a": 0, 2: 17}, handle)

    assert load_offset_map(path) == {"passage-a": 0, "2": 17}


def test_load_offset_map_rejects_code_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    path = tmp_path / "evil.passages.idx"
    with path.open("wb") as handle:
        pickle.dump(_MarkerPayload(marker), handle)

    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        load_offset_map(path)

    assert not marker.exists()


def test_load_offset_map_rejects_unexpected_shape(tmp_path: Path) -> None:
    path = tmp_path / "invalid.passages.idx"
    with path.open("wb") as handle:
        pickle.dump(["passage-a", 0], handle)

    with pytest.raises(pickle.UnpicklingError, match="dictionary"):
        load_offset_map(path)


def test_searcher_rejects_malicious_offset_map_before_loading_backend(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    index_path = tmp_path / "demo.leann"
    (tmp_path / "demo.leann.passages.jsonl").write_text("", encoding="utf-8")
    with (tmp_path / "demo.leann.passages.idx").open("wb") as handle:
        pickle.dump(_MarkerPayload(marker), handle)
    Path(f"{index_path}.meta.json").write_text(
        json.dumps(
            {
                "backend_name": "unused",
                "embedding_model": "unused",
                "passage_sources": [
                    {
                        "type": "jsonl",
                        "path": "demo.leann.passages.jsonl",
                        "index_path": "demo.leann.passages.idx",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        LeannSearcher(str(index_path), enable_warmup=False, use_daemon=False)

    assert not marker.exists()


def test_sync_snapshot_rejects_pickled_callable(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    snapshot = tmp_path / "crafted.sync_context.pickle"
    with snapshot.open("wb") as handle:
        pickle.dump(_MarkerPayload(marker), handle)

    synchronizer = FileSynchronizer(root_dir=str(tmp_path), snapshot_path=str(snapshot))

    assert synchronizer.tree is None
    assert not marker.exists()


def test_build_from_embeddings_rejects_pickled_callable(tmp_path: Path, monkeypatch) -> None:
    marker = tmp_path / "executed"
    embeddings = tmp_path / "crafted-embeddings.pkl"
    with embeddings.open("wb") as handle:
        pickle.dump(_MarkerPayload(marker), handle)
    monkeypatch.setitem(BACKEND_REGISTRY, "security-test", Mock())
    builder = LeannBuilder(backend_name="security-test", dimensions=2)

    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        builder.build_index_from_embeddings(str(tmp_path / "index.leann"), str(embeddings))

    assert not marker.exists()
