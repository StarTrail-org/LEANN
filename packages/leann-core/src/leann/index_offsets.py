import pickle
from pathlib import Path


class _OffsetMapUnpickler(pickle.Unpickler):
    """Deserialize only the primitive container used by legacy offset maps."""

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(
            f"Global object {module}.{name} is not allowed in a passage offset map"
        )

    def persistent_load(self, pid):
        raise pickle.UnpicklingError(
            "Persistent references are not allowed in a passage offset map"
        )


def load_offset_map(path: str | Path) -> dict[str, int]:
    with Path(path).open("rb") as handle:
        data = _OffsetMapUnpickler(handle).load()

    if not isinstance(data, dict):
        raise pickle.UnpicklingError("Passage offset map must be a dictionary")

    offsets: dict[str, int] = {}
    for key, offset in data.items():
        if isinstance(key, bool) or not isinstance(key, (str, int)):
            raise pickle.UnpicklingError("Passage offset map keys must be strings or integers")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise pickle.UnpicklingError("Passage offset map values must be non-negative integers")
        normalized_key = str(key)
        if normalized_key in offsets:
            raise pickle.UnpicklingError("Passage offset map contains duplicate passage IDs")
        offsets[normalized_key] = offset
    return offsets
