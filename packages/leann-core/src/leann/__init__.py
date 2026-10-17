# packages/leann-core/src/leann/__init__.py
import os
import platform

# ruff: noqa: E402  (env vars must be set before importing the rest of the package)

# Fix OpenMP/FAISS threading defaults for common platforms
system = platform.system()

if system == "Darwin":
    # macOS ARM64: prevent runaway threading and duplicate lib issues
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["KMP_BLOCKTIME"] = "0"
    # Additional fixes for PyTorch/sentence-transformers on macOS ARM64 only in CI
    if os.environ.get("CI") == "true":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
elif system == "Linux":
    # Linux CPU-only: default to single-thread to avoid FAISS/ZMQ hangs (issue #208)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("FAISS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .api import LeannBuilder, LeannChat, LeannSearcher
    from .react_agent import ReActAgent, create_react_agent
    from .registry import BACKEND_REGISTRY

# The public API is resolved lazily (PEP 562). Importing a lightweight
# submodule such as ``leann.mcp`` or ``leann.settings`` must not pull in the
# whole stack (api -> chat -> embedding server -> faiss backends), which costs
# gigabytes of resident memory for processes that only need a few helpers.
# Backend auto-discovery runs when ``leann.api`` is first imported.
_LAZY_ATTRS = {
    "LeannBuilder": ".api",
    "LeannChat": ".api",
    "LeannSearcher": ".api",
    "ReActAgent": ".react_agent",
    "create_react_agent": ".react_agent",
    "BACKEND_REGISTRY": ".registry",
}

__all__ = [
    "BACKEND_REGISTRY",
    "LeannBuilder",
    "LeannChat",
    "LeannSearcher",
    "ReActAgent",
    "create_react_agent",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    if name == "BACKEND_REGISTRY":
        # Make sure backends are discovered before handing out the registry.
        importlib.import_module(".api", __name__)
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value  # cache so __getattr__ is only hit once per name
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
