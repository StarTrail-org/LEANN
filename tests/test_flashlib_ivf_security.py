import pickle
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


class _MarkerPayload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return Path.write_text, (self.marker, "executed")


def test_flashlib_index_load_rejects_pickled_callable(tmp_path: Path, monkeypatch) -> None:
    from leann_backend_flashlib_ivf import flashlib_ivf_backend

    marker = tmp_path / "executed"
    archive = tmp_path / "crafted.flashlib_ivf.pt"
    torch.save(_MarkerPayload(marker), archive)
    monkeypatch.setattr(flashlib_ivf_backend, "_import_flashlib", lambda: (dict, None, None))

    with pytest.raises(pickle.UnpicklingError):
        flashlib_ivf_backend._load_index(archive, device="cpu")

    assert not marker.exists()


def test_flashlib_index_load_accepts_state_dict(tmp_path: Path, monkeypatch) -> None:
    from leann_backend_flashlib_ivf import flashlib_ivf_backend

    class StubIndex:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    state = {field: torch.zeros(1) for field in flashlib_ivf_backend._TENSOR_FIELDS}
    state.update(dict.fromkeys(flashlib_ivf_backend._SCALAR_FIELDS, 1))
    archive = tmp_path / "valid.flashlib_ivf.pt"
    torch.save(state, archive)
    monkeypatch.setattr(flashlib_ivf_backend, "_import_flashlib", lambda: (StubIndex, None, None))

    index = flashlib_ivf_backend._load_index(archive, device="cpu")

    assert isinstance(index, StubIndex)
    assert index.nlist == 1
