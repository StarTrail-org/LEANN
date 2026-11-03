"""Packaging metadata checks for CPU-only installs."""

import ast
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for Python < 3.11
    tomllib = pytest.importorskip("tomli")


def _load_leann_pyproject():
    pyproject_path = Path(__file__).resolve().parents[1] / "packages" / "leann" / "pyproject.toml"
    return tomllib.loads(pyproject_path.read_text())


def _leann_core_source_files():
    package_root = Path(__file__).resolve().parents[1] / "packages" / "leann-core" / "src" / "leann"
    return sorted(package_root.rglob("*.py"))


def _module_scope_imports(source: str):
    """Yield (lineno, module) for every import executed at module scope."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module


def _load_leann_core_pyproject():
    pyproject_path = (
        Path(__file__).resolve().parents[1] / "packages" / "leann-core" / "pyproject.toml"
    )
    return tomllib.loads(pyproject_path.read_text())


def test_leann_base_dependencies_include_diskann():
    data = _load_leann_pyproject()
    version = data["project"]["version"]
    deps = data["project"].get("dependencies", [])

    assert f"leann-core>={version}" in deps
    assert f"leann-backend-hnsw>={version}" in deps
    assert f"leann-backend-diskann>={version}" in deps


def test_leann_core_numpy_is_bounded_below_3():
    data = _load_leann_core_pyproject()
    deps = data["project"].get("dependencies", [])

    assert any(dep.startswith("numpy") and ">=1.20.0" in dep and "<3" in dep for dep in deps)


def test_leann_core_cpu_extra_pins_cpu_torch():
    data = _load_leann_core_pyproject()
    extras = data["project"].get("optional-dependencies", {})

    cpu_deps = extras.get("cpu", [])
    assert cpu_deps, "cpu extra should be defined"
    assert any(
        dep.startswith("torch")
        and "==2.2.2" in dep
        and "platform_system == 'Linux'" in dep
        and "python_version < '3.13'" in dep
        for dep in cpu_deps
    )


def test_leann_cpu_extra_defined():
    data = _load_leann_pyproject()
    version = data["project"]["version"]
    extras = data["project"].get("optional-dependencies", {})

    assert "cpu" in extras
    assert f"leann-core[cpu]>={version}" in extras["cpu"]


def test_leann_core_does_not_import_a_backend_at_module_scope():
    """leann-core declares no leann-backend-* dependency, so importing one eagerly
    turns `import leann` into a ModuleNotFoundError for anyone who installed
    leann-core on its own or paired it with a non-HNSW backend."""
    data = _load_leann_core_pyproject()
    deps = data["project"].get("dependencies", [])
    assert not any(dep.startswith("leann-backend") for dep in deps), (
        "leann-core now depends on a backend; this test needs revisiting"
    )

    offenders = [
        f"{path.name}:{lineno} imports {module}"
        for path in _leann_core_source_files()
        for lineno, module in _module_scope_imports(path.read_text(encoding="utf-8"))
        if module.split(".")[0].startswith("leann_backend")
    ]

    assert not offenders, (
        "leann-core must import backends lazily inside the functions that use them: "
        + "; ".join(offenders)
    )
