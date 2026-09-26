"""
Prepare ContextBench repositories and build LEANN indexes.

For each selected ContextBench instance:
  1. Clone the repo into <WORK_ROOT>/<instance_id>
  2. Checkout base_commit
  3. Build LEANN index under <WORK_ROOT>/<instance_id>/.leann/

Usage:
    cd scripts
    python prepare_repos_with_leann.py
"""

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from git import Repo

WORK_ROOT = os.environ.get("WORK_ROOT", "contextbench_work_dir_claude")
LEANN_BIN = os.environ.get("LEANN_BIN", "leann")
DATASET_NAME = os.environ.get("DATASET_NAME", "Contextbench/ContextBench")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "train")
BENCH_FILTER = os.environ.get("BENCH_FILTER", "").strip()  # Verified | Pro | Poly | Multi
LEANN_SOURCE_EXTENSIONS = os.environ.get(
    "LEANN_SOURCE_EXTENSIONS",
    "py,go,js,jsx,ts,tsx,java,kt,kts,rs,rb,php,cs,c,cc,cpp,h,hpp,m,mm,swift,scala,sh,sql,lua,r",
)

# LEANN index build parameters
LEANN_EMBEDDING_MODEL = os.environ.get(
    "LEANN_EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-code"
)
LEANN_AST_CHUNK_SIZE = os.environ.get("LEANN_AST_CHUNK_SIZE", "600")
LEANN_AST_CHUNK_OVERLAP = os.environ.get("LEANN_AST_CHUNK_OVERLAP", "96")
# Set to "0" to disable vendor/generated exclusion (e.g. for ablation experiments).
LEANN_EXCLUDE_VENDOR = os.environ.get("LEANN_EXCLUDE_VENDOR", "1").strip() != "0"
# Set to "1" to exclude test files (e.g. for ablation experiments). Default off
# to avoid missing bugfix targets that touch test/fixture/spec files.
LEANN_EXCLUDE_TESTS = os.environ.get("LEANN_EXCLUDE_TESTS", "0").strip() != "0"
FAILED_INSTANCES_LOG = os.environ.get(
    "FAILED_INSTANCES_LOG", "prepare_repos_with_leann_failures.jsonl"
).strip()

# Fill in ContextBench instance_ids. Leave empty to prepare all tasks
# (optionally filtered by BENCH_FILTER).
SELECTED_IDS: list[str] = [
    # "SWE-Bench-Pro__python__maintenance__bugfix__19a1fba2",
    # "SWE-Bench-Pro__python__maintenance__bugfix__2464eadb",
    # "SWE-Bench-Pro__python__maintenance__bugfix__38dc8f4e",
    # "SWE-Bench-Pro__javascript__maintenance__bugfix__2bfb5681",
    # "SWE-Bench-Pro__python__maintenance__bugfix__71253eae",
    # "SWE-Bench-Pro__javascript__maintenance__bugfix__93b583ae",
    # "SWE-Bench-Pro__python__maintenance__bugfix__dcc84d4c",
    # "SWE-Bench-Pro__python__maintenance__bugfix__462b957d",
    # "SWE-Bench-Pro__python__maintenance__bugfix__9af74069",
    # "SWE-Bench-Pro__python__maintenance__bugfix__7b688a35",
    # "SWE-Bench-Pro__python__maintenance__bugfix__64fffdfa",
    # "SWE-Bench-Pro__python__maintenance__bugfix__22a1484c",
    # "SWE-Bench-Pro__go__maintenance__bugfix__1177cd53",
    # "SWE-Bench-Pro__python__maintenance__bugfix__a4287775",
    # "SWE-Bench-Pro__python__maintenance__bugfix__ba13492e",
    # "SWE-Bench-Pro__go__maintenance__bugfix__b91d5788",
    # "SWE-Bench-Pro__python__maintenance__bugfix__091dae2f",
    # "SWE-Bench-Pro__python__maintenance__bugfix__b6eff698",
    # "SWE-Bench-Pro__python__maintenance__bugfix__fcb506a5",
    # "SWE-Bench-Pro__python__maintenance__bugfix__3cfd9a02",
    # "SWE-Bench-Pro__python__maintenance__bugfix__4c132bfd",
    # "SWE-Bench-Pro__python__maintenance__bugfix__7c2efe8a",
    # "SWE-Bench-Pro__go__maintenance__bugfix__40a717e5",
    # "SWE-Bench-Pro__go__maintenance__bugfix__52d866b3",
    # "SWE-Bench-Pro__go__maintenance__bugfix__720b4d92",
    # "SWE-Bench-Pro__go__maintenance__bugfix__997c7afd",
    # "SWE-Bench-Pro__javascript__maintenance__bugfix__82518720",
    # "SWE-Bench-Pro__javascript__maintenance__bugfix__e31ec45c",
    # "SWE-Bench-Pro__python__maintenance__bugfix__07bb383a",
    # "SWE-Bench-Pro__python__maintenance__bugfix__0bac5789",
    # "SWE-Bench-Pro__python__maintenance__bugfix__18d7bbbc",
    # "SWE-Bench-Pro__python__maintenance__bugfix__1cf3e889",
    # "SWE-Bench-Pro__python__maintenance__bugfix__20dad82b",
    # "SWE-Bench-Pro__python__maintenance__bugfix__20f502e0",
    # "SWE-Bench-Pro__python__maintenance__bugfix__509a20d9",
    # "SWE-Bench-Pro__python__maintenance__bugfix__53ca6a30",
    # "SWE-Bench-Pro__python__maintenance__bugfix__552343cd",
    # "SWE-Bench-Pro__python__maintenance__bugfix__5b2cf9bb",
    # "SWE-Bench-Pro__python__maintenance__bugfix__66e05eaa",
    # "SWE-Bench-Pro__python__maintenance__bugfix__6ebb54dc",
    # "SWE-Bench-Pro__python__maintenance__bugfix__87bfb374",
    # "SWE-Bench-Pro__python__maintenance__bugfix__89932d58",
    # "SWE-Bench-Pro__python__maintenance__bugfix__942d0b14",
    # "SWE-Bench-Pro__python__maintenance__bugfix__983f2896",
    # "SWE-Bench-Pro__python__maintenance__bugfix__a984b409",
    # "SWE-Bench-Pro__python__maintenance__bugfix__aa07d0c3",
    # "SWE-Bench-Pro__python__maintenance__bugfix__cf01f471",
    # "SWE-Bench-Pro__python__maintenance__bugfix__d2506f10",
    # "SWE-Bench-Pro__python__maintenance__bugfix__e579f2f0",
    # "SWE-Bench-Pro__python__maintenance__bugfix__eafb1f0b",
    # "SWE-Bench-Pro__python__maintenance__bugfix__ef8756b1",
    # "SWE-Bench-Pro__python__maintenance__bugfix__f87209f8",
    # "SWE-Bench-Pro__python__maintenance__bugfix__ff79bafd",
]

if os.environ.get("SELECTED_IDS"):
    SELECTED_IDS = [x.strip() for x in os.environ["SELECTED_IDS"].split(",") if x.strip()]

# Vendor and generated-code directory/file patterns to exclude from the index.
# These are third-party or machine-generated files that are never the target of
# a bugfix, so indexing them only adds noise to search results.
_VENDOR_DIR_PATTERNS = (
    "vendor/",
    "node_modules/",
    "third_party/",
    "thirdparty/",
    "externals/",
    ".cache/",
)
_GENERATED_FILE_PATTERNS = (
    "_pb.go",
    ".pb.go",
    "_gen.go",
    ".pb.cc",
    ".pb.h",
)
# Filenames like `zz_generated.deepcopy.go` end in `.go`, not `zz_generated`;
# match these as path substrings (controller-gen / k8s-style outputs).
_GENERATED_FILE_SUBSTRINGS = ("zz_generated",)

# Test file path/name patterns to exclude from the index.
_TEST_PATH_PATTERNS = (
    "/test/",
    "/tests/",
    "/__tests__/",
    "/spec/",
    "/testdata/",
    "/test_",
    "/fixtures/",
)
_TEST_FILE_PATTERNS = (
    "_test.py",
    "_test.go",
    ".test.js",
    ".test.ts",
    ".test.jsx",
    ".test.tsx",
    ".spec.js",
    ".spec.ts",
    ".spec.jsx",
    ".spec.tsx",
    "_spec.rb",
)


def _run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _read_json_file(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _count_index_chunks(repo_dir: Path, instance_id: str) -> Optional[int]:
    ids_path = repo_dir / ".leann" / "indexes" / instance_id / "documents.ids.txt"
    if not ids_path.exists():
        return None
    with ids_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _print_subprocess_output(label: str, text: str, max_lines: int = 20) -> None:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return
    print(f"   📄 {label}:")
    for line in lines[:max_lines]:
        print(f"      {line}")
    if len(lines) > max_lines:
        print(f"      ... ({len(lines) - max_lines} more lines)")


def _write_failure_report(failures: list[dict]) -> Optional[Path]:
    if not failures:
        return None
    report_path = Path(FAILED_INSTANCES_LOG)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")
    return report_path


def _load_tasks() -> list[dict]:
    print(f"📚 Loading dataset: {DATASET_NAME} ({DATASET_SPLIT})...")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    tasks: list[dict] = list(ds)
    if BENCH_FILTER:
        tasks = [t for t in tasks if t.get("source", "") == BENCH_FILTER]

    if SELECTED_IDS:
        task_lookup = {t["instance_id"]: t for t in tasks}
        selected: list[dict] = []
        for iid in SELECTED_IDS:
            task = task_lookup.get(iid)
            if not task:
                print(f"⚠️ Instance not found in dataset/split/filter: {iid}")
                continue
            selected.append(task)
        return selected

    return tasks


def _is_pytest_style_test_py(normalized_path: str) -> bool:
    """True for pytest-style modules: basename test_*.py (e.g. test_foo.py)."""
    name = Path(normalized_path).name
    return name.startswith("test_") and name.lower().endswith(".py")


def build_leann_index(instance_id: str, repo_dir: Path) -> tuple[bool, Optional[str]]:
    print(f"   🔍 Building LEANN index for {instance_id}...")

    result = _run_command(["git", "ls-files"], cwd=repo_dir)
    if result.returncode != 0:
        error = f"Could not list git files: {result.stderr.strip()}"
        print(f"   ⚠️ {error}")
        return False, error

    tracked_files = [f for f in result.stdout.strip().split("\n") if f]
    allowed_exts = {
        f".{ext.strip().lstrip('.').lower()}"
        for ext in LEANN_SOURCE_EXTENSIONS.split(",")
        if ext.strip()
    }
    source_files = [f for f in tracked_files if Path(f).suffix.lower() in allowed_exts]

    # Exclude vendor/generated files — they are never bugfix targets and add noise.
    if LEANN_EXCLUDE_VENDOR:
        before = len(source_files)
        normalized = [f.replace("\\", "/") for f in source_files]
        source_files = [
            f
            for f, n in zip(source_files, normalized)
            if not any(pat in n for pat in _VENDOR_DIR_PATTERNS)
            and not any(n.endswith(pat) for pat in _GENERATED_FILE_PATTERNS)
            and not any(sub in n for sub in _GENERATED_FILE_SUBSTRINGS)
        ]
        excluded = before - len(source_files)
        if excluded:
            print(
                f"   🚫 Excluded {excluded} vendor/generated files ({before} → {len(source_files)})"
            )

    # Exclude test files — they are rarely bugfix targets and consistently rank
    # high in semantic search due to mirroring production code patterns.
    if LEANN_EXCLUDE_TESTS:
        before = len(source_files)
        normalized = [f.replace("\\", "/") for f in source_files]
        source_files = [
            f
            for f, n in zip(source_files, normalized)
            if not any(pat in n for pat in _TEST_PATH_PATTERNS)
            and not _is_pytest_style_test_py(n)
            and not any(n.endswith(pat) for pat in _TEST_FILE_PATTERNS)
        ]
        excluded = before - len(source_files)
        if excluded:
            print(f"   🚫 Excluded {excluded} test files ({before} → {len(source_files)})")

    if not source_files:
        error = f"No source files found for extensions: {sorted(allowed_exts)}"
        print(f"   ⚠️ {error}")
        return False, error

    # Derive --file-types from the actual extensions present after filtering,
    # so all indexed file types benefit from AST-aware chunking.
    indexed_exts = sorted(
        {Path(f).suffix.lstrip(".").lower() for f in source_files if Path(f).suffix}
    )
    file_types_arg = ",".join(indexed_exts)

    print(f"   📊 Found {len(source_files)} source files (types: {file_types_arg})")
    leann_cmd = [
        LEANN_BIN,
        "build",
        instance_id,
        "--docs",
        *source_files,
        "--embedding-mode",
        "sentence-transformers",
        "--embedding-model",
        LEANN_EMBEDDING_MODEL,
        "--backend",
        "hnsw",
        "--file-types",
        file_types_arg,
        "--force",
        "--ast-chunk-size",
        LEANN_AST_CHUNK_SIZE,
        "--ast-chunk-overlap",
        LEANN_AST_CHUNK_OVERLAP,
        "--use-ast-chunking",
        "--no-recompute",
    ]
    debug_cmd = [
        LEANN_BIN,
        "build",
        instance_id,
        "--docs",
        f"<{len(source_files)} files>",
        *leann_cmd[len(["leann", "build", instance_id, "--docs"]) + len(source_files) :],
    ]
    print(f"   🧪 LEANN command: {shlex.join(debug_cmd)}")

    try:
        proc = subprocess.run(
            leann_cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=1800,
            env={
                **os.environ,
                "LEANN_EMBEDDING_DEVICE": os.environ.get("LEANN_EMBEDDING_DEVICE", "mps"),
                "LEANN_BATCH_SIZE": os.environ.get("LEANN_BATCH_SIZE", "8"),
            },
        )
    except subprocess.TimeoutExpired:
        error = "LEANN build timed out"
        print(f"   ⏰ {error}")
        return False, error
    except Exception as e:
        error = f"LEANN error: {e}"
        print(f"   ❌ {error}")
        return False, error

    _print_subprocess_output("LEANN stdout", proc.stdout)
    _print_subprocess_output("LEANN stderr", proc.stderr)

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        print(f"   ❌ LEANN build failed: {stderr}")
        return False, f"LEANN build failed: {stderr}"

    meta_path = repo_dir / ".leann" / "indexes" / instance_id / "documents.leann.meta.json"
    meta = _read_json_file(meta_path)
    if meta:
        print(f"   🧠 Embedding model in index: {meta.get('embedding_model', 'unknown')}")
    print("   🌲 AST chunking requested: yes (--use-ast-chunking)")
    chunk_count = _count_index_chunks(repo_dir, instance_id)
    if chunk_count is not None:
        print(f"   🧩 Indexed chunks: {chunk_count}")

    print("   ✅ LEANN index built successfully")
    return True, None


def prepare_single_task(task: dict) -> tuple[bool, Optional[str]]:
    instance_id = task["instance_id"]
    repo_url = task["repo_url"]
    base_commit = task["base_commit"]
    target_dir = Path(WORK_ROOT) / instance_id

    print(f"\n{'=' * 72}")
    print(f"📦 Preparing: {instance_id}")
    print(f"   repo: {repo_url}")
    print(f"   commit: {base_commit[:12]}...")
    print(f"{'=' * 72}")

    if not target_dir.exists():
        print(f"   📥 Cloning {repo_url}...")
        try:
            Repo.clone_from(repo_url, target_dir)
        except Exception as e:
            print(f"   ❌ Clone failed: {e}")
            return False, f"Clone failed: {e}"
    else:
        print("   ✓ Repo already exists")

    try:
        repo = Repo(target_dir)
        print(f"   🔀 Checking out {base_commit[:8]}...")
        repo.git.reset("--hard")
        repo.git.checkout(base_commit)
        repo.git.clean("-fdx", "-e", ".leann/")
        (target_dir / "PROBLEM.md").write_text(task.get("problem_statement", ""), encoding="utf-8")
    except Exception as e:
        print(f"   ❌ Checkout/clean failed: {e}")
        return False, f"Checkout/clean failed: {e}"

    return build_leann_index(instance_id, target_dir)


def main():
    print("🚀 ContextBench Repository Preparation with LEANN Indexing")
    print("=" * 72)

    leann_path = shutil.which(LEANN_BIN)
    if not leann_path:
        print("❌ LEANN not found. Install: uv tool install leann-core --with leann")
        return
    print(f"✅ LEANN found: {leann_path}")

    tasks = _load_tasks()
    if not tasks:
        print("⚠️ No tasks selected. Set SELECTED_IDS or adjust BENCH_FILTER.")
        return

    Path(WORK_ROOT).mkdir(parents=True, exist_ok=True)
    print(f"\n📂 Work root: {WORK_ROOT}")
    print(f"🎯 Tasks to prepare: {len(tasks)}")
    if BENCH_FILTER:
        print(f"🔎 Bench filter: {BENCH_FILTER}")

    success_count = 0
    fail_count = 0
    failures: list[dict] = []
    for i, task in enumerate(tasks, start=1):
        print(f"\n[{i}/{len(tasks)}]")
        succeeded, error = prepare_single_task(task)
        if succeeded:
            success_count += 1
        else:
            fail_count += 1
            failures.append(
                {
                    "instance_id": task.get("instance_id", ""),
                    "repo_url": task.get("repo_url", ""),
                    "base_commit": task.get("base_commit", ""),
                    "error": error or "unknown error",
                    "failed_at_unix": int(time.time()),
                }
            )

    print(f"\n{'=' * 72}")
    print(f"🎉 Done! ✅ {success_count} succeeded  ❌ {fail_count} failed")
    if failures:
        print("\nFailed instances:")
        for item in failures:
            print(f"  - {item['instance_id']}: {item['error']}")
        report_path = _write_failure_report(failures)
        if report_path is not None:
            print(f"\n📝 Failure report written to: {report_path}")
    print("\nNext steps:")
    print(f"  1. Verify indexes: ls {WORK_ROOT}/*/.leann")
    print("  2. Run with LEANN MCP: LEANN_MODE=mcp python batch_run_selected.py")
    print("  3. Baseline run:      LEANN_MODE=none python batch_run_selected.py")


if __name__ == "__main__":
    main()
