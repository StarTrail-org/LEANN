"""
Compare two latency models for small incremental updates vs. search:

Scenario A (sequential update then search):
  - Build initial HNSW (is_recompute=True)
  - Start embedding server (ZMQ) for recompute
  - Add N passages one-by-one (each triggers recompute over ZMQ)
  - Then run a search query on the updated index
  - Report total time = sum(add_i) + search_time, with breakdowns

Scenario B (offline embeds + concurrent search; no graph updates):
  - Do NOT insert the N passages into the graph
  - In parallel: (1) compute embeddings for the N passages; (2) compute query
    embedding and run a search on the existing index
  - After both finish, compute similarity between the query embedding and the N
    new passage embeddings, merge with the index search results by score, and
    report time = max(embed_time, search_time) (i.e., no blocking on updates)

This script reuses the model/data loading conventions of
examples/bench_hnsw_rng_recompute.py but focuses on end-to-end latency
comparison for the two execution strategies above.

Example (from the repository root):
  uv run -m benchmarks.update.bench_update_vs_offline_search \
    --index-path .leann/bench/offline_vs_update.leann \
    --max-initial 300 --num-updates 5 --k 10
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil  # type: ignore
from leann.api import LeannBuilder

if os.environ.get("LEANN_FORCE_CPU", "").lower() in ("1", "true", "yes"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from leann.embedding_compute import compute_embeddings
from leann.embedding_server_manager import EmbeddingServerManager
from leann.registry import register_project_directory
from leann_backend_hnsw import faiss  # type: ignore

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def _find_repo_root() -> Path:
    """Locate project root by walking up until pyproject.toml is found."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume repo is two levels up (../..)
    return current.parents[2]


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.chunking import create_text_chunks  # noqa: E402

DEFAULT_INITIAL_FILES = [
    REPO_ROOT / "data" / "2501.14312v1 (1).pdf",
    REPO_ROOT / "data" / "huawei_pangu.md",
]
DEFAULT_UPDATE_FILES = [REPO_ROOT / "data" / "2506.08276v1.pdf"]


def load_chunks_from_files(paths: list[Path], limit: int | None = None) -> list[str]:
    from llama_index.core import SimpleDirectoryReader

    documents = []
    for path in paths:
        p = path.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input path not found: {p}")
        if p.is_dir():
            reader = SimpleDirectoryReader(str(p), recursive=False)
            documents.extend(reader.load_data(show_progress=True))
        else:
            reader = SimpleDirectoryReader(input_files=[str(p)])
            documents.extend(reader.load_data(show_progress=True))

    if not documents:
        return []

    chunks = create_text_chunks(
        documents,
        chunk_size=512,
        chunk_overlap=128,
        use_ast_chunking=False,
    )
    cleaned = [c for c in chunks if isinstance(c, str) and c.strip()]
    if limit is not None:
        cleaned = cleaned[:limit]
    return cleaned


def ensure_index_dir(index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)


def cleanup_index_files(index_path: Path) -> None:
    parent = index_path.parent
    if not parent.exists():
        return
    stem = index_path.stem
    for file in parent.glob(f"{stem}*"):
        if file.is_file():
            file.unlink()


def build_initial_index(
    index_path: Path,
    paragraphs: list[str],
    model_name: str,
    embedding_mode: str,
    distance_metric: str,
    ef_construction: int,
) -> None:
    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model=model_name,
        embedding_mode=embedding_mode,
        is_compact=False,
        is_recompute=True,
        distance_metric=distance_metric,
        backend_kwargs={
            "distance_metric": distance_metric,
            "is_compact": False,
            "is_recompute": True,
            "efConstruction": ef_construction,
        },
    )
    for idx, passage in enumerate(paragraphs):
        builder.add_text(passage, metadata={"id": str(idx)})
    builder.build_index(str(index_path))


def _maybe_norm_cosine(vecs: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs = vecs / norms
    return vecs


def _read_index_for_search(index_path: Path) -> Any:
    index_file = index_path.parent / f"{index_path.stem}.index"
    # Force-disable experimental disk cache when loading the index so that
    # incremental benchmarks don't pick up stale top-degree bitmaps.
    cfg = faiss.HNSWIndexConfig()
    cfg.is_recompute = True
    if hasattr(cfg, "disk_cache_ratio"):
        cfg.disk_cache_ratio = 0.0
    if hasattr(cfg, "external_storage_path"):
        cfg.external_storage_path = None
    io_flags = getattr(faiss, "IO_FLAG_MMAP", 0)
    index = faiss.read_index(str(index_file), io_flags, cfg)
    # ensure recompute mode persists after reload
    try:
        index.is_recompute = True
    except AttributeError:
        pass
    try:
        actual_ntotal = index.hnsw.levels.size()
    except AttributeError:
        actual_ntotal = index.ntotal
    if actual_ntotal != index.ntotal:
        print(
            f"[bench_update_vs_offline_search] Correcting ntotal from {index.ntotal} to {actual_ntotal}",
            flush=True,
        )
        index.ntotal = actual_ntotal
    if getattr(index, "storage", None) is None:
        if index.metric_type == faiss.METRIC_INNER_PRODUCT:
            storage_index = faiss.IndexFlatIP(index.d)
        else:
            storage_index = faiss.IndexFlatL2(index.d)
        index.storage = storage_index
        index.own_fields = True
    return index


def _append_passages_for_updates(
    meta_path: Path,
    start_id: int,
    texts: list[str],
) -> list[str]:
    """Append update passages so the embedding server can serve recompute fetches."""

    if not texts:
        return []

    index_dir = meta_path.parent
    meta_name = meta_path.name
    if not meta_name.endswith(".meta.json"):
        raise ValueError(f"Unexpected meta filename: {meta_path}")
    index_base = meta_name[: -len(".meta.json")]

    passages_file = index_dir / f"{index_base}.passages.jsonl"
    offsets_file = index_dir / f"{index_base}.passages.idx"

    if not passages_file.exists() or not offsets_file.exists():
        raise FileNotFoundError(
            "Passage store missing; cannot register update passages for recompute mode."
        )

    with open(offsets_file, "rb") as f:
        offset_map: dict[str, int] = pickle.load(f)

    assigned_ids: list[str] = []
    with open(passages_file, "a", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            passage_id = str(start_id + i)
            offset = f.tell()
            json.dump({"id": passage_id, "text": text, "metadata": {}}, f, ensure_ascii=False)
            f.write("\n")
            offset_map[passage_id] = offset
            assigned_ids.append(passage_id)

    with open(offsets_file, "wb") as f:
        pickle.dump(offset_map, f)

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except json.JSONDecodeError:
        meta = {}
    meta["total_passages"] = len(offset_map)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return assigned_ids


def _search(index: Any, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    q = np.ascontiguousarray(q, dtype=np.float32)
    distances = np.zeros((1, k), dtype=np.float32)
    indices = np.zeros((1, k), dtype=np.int64)
    index.search(
        1,
        faiss.swig_ptr(q),
        k,
        faiss.swig_ptr(distances),
        faiss.swig_ptr(indices),
    )
    return distances[0], indices[0]


def _score_for_metric(dist: float, metric: str) -> float:
    # Convert FAISS distance to a "higher is better" score
    if metric in ("mips", "cosine"):
        return float(dist)
    # l2 distance (smaller better) -> negative distance as score
    return -float(dist)


def _merge_results(
    index_results: tuple[np.ndarray, np.ndarray],
    offline_scores: list[tuple[int, float]],
    k: int,
    metric: str,
) -> list[tuple[str, float]]:
    distances, indices = index_results
    merged: list[tuple[str, float]] = []
    for distance, idx in zip(distances.tolist(), indices.tolist()):
        merged.append((f"idx:{idx}", _score_for_metric(distance, metric)))
    for j, s in offline_scores:
        merged.append((f"offline:{j}", s))
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:k]


@dataclass
class ScenarioResult:
    name: str
    update_total_s: float
    search_s: float
    overall_s: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path(".leann/bench/offline-vs-update.leann"),
    )
    parser.add_argument(
        "--initial-files",
        nargs="*",
        type=Path,
        default=DEFAULT_INITIAL_FILES,
    )
    parser.add_argument(
        "--update-files",
        nargs="*",
        type=Path,
        default=DEFAULT_UPDATE_FILES,
    )
    parser.add_argument("--max-initial", type=int, default=300)
    parser.add_argument("--num-updates", type=int, default=5)
    parser.add_argument("--k", type=int, default=10, help="Top-k for search/merge")
    parser.add_argument(
        "--query",
        type=str,
        default="neural network",
        help="Query text used for the search benchmark.",
    )
    parser.add_argument("--server-port", type=int, default=5557)
    parser.add_argument("--add-timeout", type=int, default=600)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-mode", default="sentence-transformers")
    parser.add_argument(
        "--distance-metric",
        default="mips",
        choices=["mips", "l2", "cosine"],
    )
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument(
        "--only",
        choices=["A", "B", "both"],
        default="both",
        help="Run only Scenario A, Scenario B, or both",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("benchmarks/update/offline_vs_update.csv"),
        help="Where to append results (CSV).",
    )

    args = parser.parse_args()

    register_project_directory(REPO_ROOT)

    # Load data
    initial_paragraphs = load_chunks_from_files(args.initial_files, args.max_initial)
    update_paragraphs = load_chunks_from_files(args.update_files, None)
    if not update_paragraphs:
        raise ValueError("No update passages loaded from --update-files")
    update_paragraphs = update_paragraphs[: args.num_updates]
    if len(update_paragraphs) < args.num_updates:
        raise ValueError(
            f"Not enough update passages ({len(update_paragraphs)}) for --num-updates={args.num_updates}"
        )

    ensure_index_dir(args.index_path)
    cleanup_index_files(args.index_path)

    # Build initial index
    build_initial_index(
        args.index_path,
        initial_paragraphs,
        args.model_name,
        args.embedding_mode,
        args.distance_metric,
        args.ef_construction,
    )

    # Prepare index object and meta
    meta_path = args.index_path.parent / f"{args.index_path.name}.meta.json"
    index = _read_index_for_search(args.index_path)

    # CSV setup
    run_id = time.strftime("%Y%m%d-%H%M%S")
    if args.csv_path:
        args.csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_fields = [
            "run_id",
            "scenario",
            "max_initial",
            "num_updates",
            "k",
            "total_time_s",
            "add_total_s",
            "search_time_s",
            "emb_time_s",
            "makespan_s",
            "model_name",
            "embedding_mode",
            "distance_metric",
        ]
        if not args.csv_path.exists() or args.csv_path.stat().st_size == 0:
            with args.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writeheader()

    # Debug: list existing HNSW server PIDs before starting
    try:
        existing = [
            p
            for p in psutil.process_iter(attrs=["pid", "cmdline"])
            if any(
                isinstance(arg, str) and "leann_backend_hnsw.hnsw_embedding_server" in arg
                for arg in (p.info.get("cmdline") or [])
            )
        ]
        if existing:
            print("[debug] Found existing hnsw_embedding_server processes before run:")
            for p in existing:
                print(f"[debug]  PID={p.info['pid']} cmd={' '.join(p.info.get('cmdline') or [])}")
    except Exception as _e:
        pass

    add_total = 0.0
    search_after_add = 0.0
    total_seq = 0.0
    port_a = None
    if args.only in ("A", "both"):
        # Scenario A: sequential update then search
        start_id = index.ntotal
        assigned_ids = _append_passages_for_updates(meta_path, start_id, update_paragraphs)
        if assigned_ids:
            logger.debug(
                "Registered %d update passages starting at id %s",
                len(assigned_ids),
                assigned_ids[0],
            )
        server_manager = EmbeddingServerManager(
            backend_module_name="leann_backend_hnsw.hnsw_embedding_server"
        )
        ok, port = server_manager.start_server(
            port=args.server_port,
            model_name=args.model_name,
            embedding_mode=args.embedding_mode,
            passages_file=str(meta_path),
            distance_metric=args.distance_metric,
        )
        if not ok:
            raise RuntimeError("Failed to start embedding server")
        try:
            # Set ZMQ port for recompute mode
            if hasattr(index.hnsw, "set_zmq_port"):
                index.hnsw.set_zmq_port(port)
            elif hasattr(index, "set_zmq_port"):
                index.set_zmq_port(port)

            # Start A overall timer BEFORE computing update embeddings
            t0 = time.time()

            # Compute embeddings for updates (counted into A's overall)
            t_emb0 = time.time()
            upd_embs = compute_embeddings(
                update_paragraphs,
                args.model_name,
                mode=args.embedding_mode,
                is_build=False,
                batch_size=16,
            )
            emb_time_updates = time.time() - t_emb0
            upd_embs = np.asarray(upd_embs, dtype=np.float32)
            upd_embs = _maybe_norm_cosine(upd_embs, args.distance_metric)

            # Perform sequential adds
            for i in range(upd_embs.shape[0]):
                t_add0 = time.time()
                index.add(1, faiss.swig_ptr(upd_embs[i : i + 1]))
                add_total += time.time() - t_add0
            # Don't persist index after adds to avoid contaminating Scenario B
            # index_file = args.index_path.parent / f"{args.index_path.stem}.index"
            # faiss.write_index(index, str(index_file))

            # Search after updates
            q_emb = compute_embeddings(
                [args.query], args.model_name, mode=args.embedding_mode, is_build=False
            )
            q_emb = np.asarray(q_emb, dtype=np.float32)
            q_emb = _maybe_norm_cosine(q_emb, args.distance_metric)

            # Warm up search with a dummy query first
            print("[DEBUG] Warming up search...")
            _ = _search(index, q_emb, 1)

            t_s0 = time.time()
            _D_upd, _I_upd = _search(index, q_emb, args.k)
            search_after_add = time.time() - t_s0
            total_seq = time.time() - t0
        finally:
            server_manager.stop_server()
        port_a = port

        print("\n=== Scenario A: update->search (sequential) ===")
        # emb_time_updates is defined only when A runs
        try:
            _emb_a = emb_time_updates
        except NameError:
            _emb_a = 0.0
        print(
            f"Adds: {args.num_updates} passages; embeds={_emb_a:.3f}s; add_total={add_total:.3f}s; "
            f"search={search_after_add:.3f}s; overall={total_seq:.3f}s"
        )
        # CSV row for A
        if args.csv_path:
            row_a = {
                "run_id": run_id,
                "scenario": "A",
                "max_initial": args.max_initial,
                "num_updates": args.num_updates,
                "k": args.k,
                "total_time_s": round(total_seq, 6),
                "add_total_s": round(add_total, 6),
                "search_time_s": round(search_after_add, 6),
                "emb_time_s": round(_emb_a, 6),
                "makespan_s": 0.0,
                "model_name": args.model_name,
                "embedding_mode": args.embedding_mode,
                "distance_metric": args.distance_metric,
            }
            with args.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writerow(row_a)

        # Verify server cleanup
        try:
            # short sleep to allow signal handling to finish
            time.sleep(0.5)
            leftovers = [
                p
                for p in psutil.process_iter(attrs=["pid", "cmdline"])
                if any(
                    isinstance(arg, str) and "leann_backend_hnsw.hnsw_embedding_server" in arg
                    for arg in (p.info.get("cmdline") or [])
                )
            ]
            if leftovers:
                print("[warn] hnsw_embedding_server process(es) still alive after A-stop:")
                for p in leftovers:
                    print(
                        f"[warn]  PID={p.info['pid']} cmd={' '.join(p.info.get('cmdline') or [])}"
                    )
            else:
                print("[debug] server cleanup confirmed: no hnsw_embedding_server found")
        except Exception:
            pass

    # Scenario B: offline embeds + concurrent search (no graph updates)
    if args.only in ("B", "both"):
        # ensure a server is available for recompute search
        server_manager_b = EmbeddingServerManager(
            backend_module_name="leann_backend_hnsw.hnsw_embedding_server"
        )
        requested_port = args.server_port if port_a is None else port_a
        ok_b, port_b = server_manager_b.start_server(
            port=requested_port,
            model_name=args.model_name,
            embedding_mode=args.embedding_mode,
            passages_file=str(meta_path),
            distance_metric=args.distance_metric,
        )
        if not ok_b:
            raise RuntimeError("Failed to start embedding server for Scenario B")

        # Wait for server to fully initialize
        print("[DEBUG] Waiting 2s for embedding server to fully initialize...")
        time.sleep(2)

        try:
            # Read the index first
            index_no_update = _read_index_for_search(args.index_path)  # unchanged index

            # Then configure ZMQ port on the correct index object
            if hasattr(index_no_update.hnsw, "set_zmq_port"):
                index_no_update.hnsw.set_zmq_port(port_b)
            elif hasattr(index_no_update, "set_zmq_port"):
                index_no_update.set_zmq_port(port_b)

            # Warmup the embedding model before benchmarking (do this for both --only B and --only both)
            # This ensures fair comparison as Scenario A has warmed up the model during update embeddings
            logger.info("Warming up embedding model for Scenario B...")
            _ = compute_embeddings(
                ["warmup text"], args.model_name, mode=args.embedding_mode, is_build=False
            )

            # Prepare worker A: compute embeddings for the same N passages
            emb_time = 0.0
            updates_embs_offline: np.ndarray | None = None

            def _worker_emb():
                nonlocal emb_time, updates_embs_offline
                t = time.time()
                updates_embs_offline = compute_embeddings(
                    update_paragraphs,
                    args.model_name,
                    mode=args.embedding_mode,
                    is_build=False,
                    batch_size=16,
                )
                emb_time = time.time() - t

            # Pre-compute query embedding and warm up search outside of timed section.
            q_vec = compute_embeddings(
                [args.query], args.model_name, mode=args.embedding_mode, is_build=False
            )
            q_vec = np.asarray(q_vec, dtype=np.float32)
            q_vec = _maybe_norm_cosine(q_vec, args.distance_metric)
            print("[DEBUG B] Warming up search...")
            _ = _search(index_no_update, q_vec, 1)

            # Worker B: timed search on the warmed index
            search_time = 0.0
            offline_elapsed = 0.0
            index_results: tuple[np.ndarray, np.ndarray] | None = None

            def _worker_search():
                nonlocal search_time, index_results
                t = time.time()
                distances, indices = _search(index_no_update, q_vec, args.k)
                search_time = time.time() - t
                index_results = (distances, indices)

            # Run two workers concurrently
            t0 = time.time()
            th1 = threading.Thread(target=_worker_emb)
            th2 = threading.Thread(target=_worker_search)
            th1.start()
            th2.start()
            th1.join()
            th2.join()
            offline_elapsed = time.time() - t0

            # For mixing: compute query vs. offline update similarities (pure client-side)
            offline_scores: list[tuple[int, float]] = []
            if updates_embs_offline is not None:
                upd2 = np.asarray(updates_embs_offline, dtype=np.float32)
                upd2 = _maybe_norm_cosine(upd2, args.distance_metric)
                # For mips/cosine, score = dot; for l2, score = -||x-y||^2
                for j in range(upd2.shape[0]):
                    if args.distance_metric in ("mips", "cosine"):
                        s = float(np.dot(q_vec[0], upd2[j]))
                    else:
                        diff = q_vec[0] - upd2[j]
                        s = -float(np.dot(diff, diff))
                    offline_scores.append((j, s))

            merged_topk = (
                _merge_results(index_results, offline_scores, args.k, args.distance_metric)
                if index_results
                else []
            )

            print("\n=== Scenario B: offline embeds + concurrent search (no add) ===")
            print(
                f"embeddings({args.num_updates})={emb_time:.3f}s; search={search_time:.3f}s; makespan≈{offline_elapsed:.3f}s (≈max)"
            )
            if merged_topk:
                preview = ", ".join([f"{lab}:{score:.3f}" for lab, score in merged_topk[:5]])
                print(f"Merged top-5 preview: {preview}")
            # CSV row for B
            if args.csv_path:
                row_b = {
                    "run_id": run_id,
                    "scenario": "B",
                    "max_initial": args.max_initial,
                    "num_updates": args.num_updates,
                    "k": args.k,
                    "total_time_s": 0.0,
                    "add_total_s": 0.0,
                    "search_time_s": round(search_time, 6),
                    "emb_time_s": round(emb_time, 6),
                    "makespan_s": round(offline_elapsed, 6),
                    "model_name": args.model_name,
                    "embedding_mode": args.embedding_mode,
                    "distance_metric": args.distance_metric,
                }
                with args.csv_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fields)
                    writer.writerow(row_b)

        finally:
            server_manager_b.stop_server()

    # Summary
    print("\n=== Summary ===")
    msg_a = (
        f"A: seq-add+search overall={total_seq:.3f}s (adds={add_total:.3f}s, search={search_after_add:.3f}s)"
        if args.only in ("A", "both")
        else "A: skipped"
    )
    msg_b = (
        f"B: offline+concurrent overall≈{offline_elapsed:.3f}s (emb={emb_time:.3f}s, search={search_time:.3f}s)"
        if args.only in ("B", "both")
        else "B: skipped"
    )
    print(msg_a + "\n" + msg_b)


if __name__ == "__main__":
    main()
