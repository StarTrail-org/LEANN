"""Benchmark incremental HNSW add() under different RNG pruning modes with real
embedding recomputation.

This script clones the structure of ``examples/dynamic_update_no_recompute.py``
so that we build a non-compact ``is_recompute=True`` index, spin up the
standard HNSW embedding server, and measure how long incremental ``add`` takes
when RNG pruning is fully enabled vs. partially/fully disabled.

Example usage (run from the repo root; downloads the model on first run)::

    uv run -m benchmarks.update.bench_hnsw_rng_recompute \
        --index-path .leann/bench/leann-demo.leann \
        --runs 1

You can tweak the input documents with ``--initial-files`` / ``--update-files``
if you want a larger or different workload, and change the embedding model via
``--model-name``.
"""

import argparse
import json
import logging
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import zmq
from leann.api import LeannBuilder

if os.environ.get("LEANN_FORCE_CPU", "").lower() in ("1", "true", "yes"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from leann.embedding_compute import compute_embeddings
from leann.embedding_server_manager import EmbeddingServerManager
from leann.registry import register_project_directory
from leann_backend_hnsw import faiss  # type: ignore
from leann_backend_hnsw.convert_to_csr import prune_hnsw_embeddings_inplace

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

DEFAULT_HNSW_LOG = Path(".leann/bench/hnsw_server.log")


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


def prepare_new_chunks(paragraphs: list[str]) -> list[dict[str, Any]]:
    return [{"text": text, "metadata": {}} for text in paragraphs]


def benchmark_update_with_mode(
    index_path: Path,
    new_chunks: list[dict[str, Any]],
    model_name: str,
    embedding_mode: str,
    distance_metric: str,
    disable_forward_rng: bool,
    disable_reverse_rng: bool,
    server_port: int,
    add_timeout: int,
    ef_construction: int,
) -> tuple[float, float]:
    meta_path = index_path.parent / f"{index_path.name}.meta.json"
    passages_file = index_path.parent / f"{index_path.name}.passages.jsonl"
    offset_file = index_path.parent / f"{index_path.name}.passages.idx"
    index_file = index_path.parent / f"{index_path.stem}.index"

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    with open(offset_file, "rb") as f:
        offset_map: dict[str, int] = pickle.load(f)
    existing_ids = set(offset_map.keys())

    valid_chunks: list[dict[str, Any]] = []
    for chunk in new_chunks:
        text = chunk.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        metadata = chunk.setdefault("metadata", {})
        passage_id = chunk.get("id") or metadata.get("id")
        if passage_id and passage_id in existing_ids:
            raise ValueError(f"Passage ID '{passage_id}' already exists in the index.")
        valid_chunks.append(chunk)

    if not valid_chunks:
        raise ValueError("No valid chunks to append.")

    texts_to_embed = [chunk["text"] for chunk in valid_chunks]
    embeddings = compute_embeddings(
        texts_to_embed,
        model_name,
        mode=embedding_mode,
        is_build=False,
        batch_size=16,
    )

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if distance_metric == "cosine":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

    index = faiss.read_index(str(index_file))
    index.is_recompute = True
    if getattr(index, "storage", None) is None:
        if index.metric_type == faiss.METRIC_INNER_PRODUCT:
            storage_index = faiss.IndexFlatIP(index.d)
        else:
            storage_index = faiss.IndexFlatL2(index.d)
        index.storage = storage_index
        index.own_fields = True
        try:
            storage_index.ntotal = index.ntotal
        except AttributeError:
            pass
    try:
        index.hnsw.set_disable_rng_during_add(disable_forward_rng)
        index.hnsw.set_disable_reverse_prune(disable_reverse_rng)
        if ef_construction is not None:
            index.hnsw.efConstruction = ef_construction
    except AttributeError:
        pass

    applied_forward = getattr(index.hnsw, "disable_rng_during_add", None)
    applied_reverse = getattr(index.hnsw, "disable_reverse_prune", None)
    logger.info(
        "HNSW RNG config -> requested forward=%s, reverse=%s | applied forward=%s, reverse=%s",
        disable_forward_rng,
        disable_reverse_rng,
        applied_forward,
        applied_reverse,
    )

    base_id = index.ntotal
    for offset, chunk in enumerate(valid_chunks):
        new_id = str(base_id + offset)
        chunk.setdefault("metadata", {})["id"] = new_id
        chunk["id"] = new_id

    rollback_size = passages_file.stat().st_size if passages_file.exists() else 0
    offset_map_backup = offset_map.copy()

    try:
        with open(passages_file, "a", encoding="utf-8") as f:
            for chunk in valid_chunks:
                offset = f.tell()
                json.dump(
                    {
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "metadata": chunk.get("metadata", {}),
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
                offset_map[chunk["id"]] = offset

        with open(offset_file, "wb") as f:
            pickle.dump(offset_map, f)

        server_manager = EmbeddingServerManager(
            backend_module_name="leann_backend_hnsw.hnsw_embedding_server"
        )
        server_started, actual_port = server_manager.start_server(
            port=server_port,
            model_name=model_name,
            embedding_mode=embedding_mode,
            passages_file=str(meta_path),
            distance_metric=distance_metric,
        )
        if not server_started:
            raise RuntimeError("Failed to start embedding server.")

        if hasattr(index.hnsw, "set_zmq_port"):
            index.hnsw.set_zmq_port(actual_port)
        elif hasattr(index, "set_zmq_port"):
            index.set_zmq_port(actual_port)

        _warmup_embedding_server(actual_port)

        total_start = time.time()
        add_elapsed = 0.0

        try:
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError("incremental add timed out")

            if add_timeout > 0:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(add_timeout)

            add_start = time.time()
            for i in range(embeddings.shape[0]):
                index.add(1, faiss.swig_ptr(embeddings[i : i + 1]))
            add_elapsed = time.time() - add_start
            if add_timeout > 0:
                signal.alarm(0)
            faiss.write_index(index, str(index_file))
        finally:
            server_manager.stop_server()

    except TimeoutError:
        raise
    except Exception:
        if passages_file.exists():
            with open(passages_file, "rb+") as f:
                f.truncate(rollback_size)
        with open(offset_file, "wb") as f:
            pickle.dump(offset_map_backup, f)
        raise

    prune_hnsw_embeddings_inplace(str(index_file))

    meta["total_passages"] = len(offset_map)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Reset toggles so the index on disk returns to baseline behaviour.
    try:
        index.hnsw.set_disable_rng_during_add(False)
        index.hnsw.set_disable_reverse_prune(False)
    except AttributeError:
        pass
    faiss.write_index(index, str(index_file))

    total_elapsed = time.time() - total_start

    return total_elapsed, add_elapsed


def _total_zmq_nodes(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    with log_path.open("r", encoding="utf-8") as log_file:
        text = log_file.read()
    return sum(int(match) for match in re.findall(r"ZMQ received (\d+) node IDs", text))


def _warmup_embedding_server(port: int) -> None:
    """Send a dummy REQ so the embedding server loads its model."""
    ctx = zmq.Context()
    try:
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.setsockopt(zmq.SNDTIMEO, 5000)
        sock.connect(f"tcp://127.0.0.1:{port}")
        payload = msgpack.packb(["__WARMUP__"], use_bin_type=True)
        sock.send(payload)
        try:
            sock.recv()
        except zmq.error.Again:
            pass
    finally:
        sock.close()
        ctx.term()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path(".leann/bench/leann-demo.leann"),
        help="Output index base path (without extension).",
    )
    parser.add_argument(
        "--initial-files",
        nargs="*",
        type=Path,
        default=DEFAULT_INITIAL_FILES,
        help="Files used to build the initial index.",
    )
    parser.add_argument(
        "--update-files",
        nargs="*",
        type=Path,
        default=DEFAULT_UPDATE_FILES,
        help="Files appended during the benchmark.",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="How many times to repeat each scenario."
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model used for build/update.",
    )
    parser.add_argument(
        "--embedding-mode",
        default="sentence-transformers",
        help="Embedding mode passed to LeannBuilder/embedding server.",
    )
    parser.add_argument(
        "--distance-metric",
        default="mips",
        choices=["mips", "l2", "cosine"],
        help="Distance metric for HNSW backend.",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help="efConstruction setting for initial build.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=5557,
        help="Port for the real embedding server.",
    )
    parser.add_argument(
        "--max-initial",
        type=int,
        default=300,
        help="Optional cap on initial passages (after chunking).",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=1,
        help="Optional cap on update passages (after chunking).",
    )
    parser.add_argument(
        "--add-timeout",
        type=int,
        default=900,
        help="Timeout in seconds for the incremental add loop (0 = no timeout).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("bench_latency.png"),
        help="Where to save the latency bar plot.",
    )
    parser.add_argument(
        "--cap-y",
        type=float,
        default=None,
        help="Cap Y-axis (ms). Bars above are hatched and annotated.",
    )
    parser.add_argument(
        "--broken-y",
        action="store_true",
        help="Use broken Y-axis (two stacked axes with gap). Overrides --cap-y unless both provided.",
    )
    parser.add_argument(
        "--lower-cap-y",
        type=float,
        default=None,
        help="Lower axes upper bound for broken Y (ms). Default=1.1x second-highest.",
    )
    parser.add_argument(
        "--upper-start-y",
        type=float,
        default=None,
        help="Upper axes lower bound for broken Y (ms). Default=1.2x second-highest.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("benchmarks/update/bench_results.csv"),
        help="Where to append per-scenario results as CSV.",
    )

    args = parser.parse_args()

    register_project_directory(REPO_ROOT)

    initial_paragraphs = load_chunks_from_files(args.initial_files, args.max_initial)
    update_paragraphs = load_chunks_from_files(args.update_files, args.max_updates)
    if not update_paragraphs:
        raise ValueError("No update passages found; please provide --update-files with content.")

    update_chunks = prepare_new_chunks(update_paragraphs)
    ensure_index_dir(args.index_path)

    scenarios = [
        ("baseline", False, False, True),
        ("no_cache_baseline", False, False, False),
        ("disable_forward_rng", True, False, True),
        ("disable_forward_and_reverse_rng", True, True, True),
    ]

    log_path = Path(os.environ.get("LEANN_HNSW_LOG_PATH", DEFAULT_HNSW_LOG))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["LEANN_HNSW_LOG_PATH"] = str(log_path.resolve())
    os.environ.setdefault("LEANN_LOG_LEVEL", "INFO")

    results_total: dict[str, list[float]] = {name: [] for name, *_ in scenarios}
    results_add: dict[str, list[float]] = {name: [] for name, *_ in scenarios}
    results_zmq: dict[str, list[int]] = {name: [] for name, *_ in scenarios}
    results_stageA: dict[str, list[float]] = {name: [] for name, *_ in scenarios}
    results_stageBC: dict[str, list[float]] = {name: [] for name, *_ in scenarios}
    results_ms_per_passage: dict[str, list[float]] = {name: [] for name, *_ in scenarios}

    # CSV setup
    import csv

    run_id = time.strftime("%Y%m%d-%H%M%S")
    csv_fields = [
        "run_id",
        "scenario",
        "cache_enabled",
        "ef_construction",
        "max_initial",
        "max_updates",
        "total_time_s",
        "add_only_s",
        "latency_ms_per_passage",
        "zmq_nodes",
        "stageA_time_s",
        "stageBC_time_s",
        "model_name",
        "embedding_mode",
        "distance_metric",
    ]
    # Create CSV with header if missing
    if args.csv_path:
        args.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not args.csv_path.exists() or args.csv_path.stat().st_size == 0:
            with args.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writeheader()

    for run in range(args.runs):
        print(f"\n=== Benchmark run {run + 1}/{args.runs} ===")
        for name, disable_forward, disable_reverse, cache_enabled in scenarios:
            print(f"\nScenario: {name}")
            cleanup_index_files(args.index_path)
            if log_path.exists():
                try:
                    log_path.unlink()
                except OSError:
                    pass
            os.environ["LEANN_ZMQ_EMBED_CACHE"] = "1" if cache_enabled else "0"
            build_initial_index(
                args.index_path,
                initial_paragraphs,
                args.model_name,
                args.embedding_mode,
                args.distance_metric,
                args.ef_construction,
            )

            prev_size = log_path.stat().st_size if log_path.exists() else 0

            try:
                total_elapsed, add_elapsed = benchmark_update_with_mode(
                    args.index_path,
                    update_chunks,
                    args.model_name,
                    args.embedding_mode,
                    args.distance_metric,
                    disable_forward,
                    disable_reverse,
                    args.server_port,
                    args.add_timeout,
                    args.ef_construction,
                )
            except TimeoutError as exc:
                print(f"Scenario {name} timed out: {exc}")
                continue

            curr_size = log_path.stat().st_size if log_path.exists() else 0
            if curr_size < prev_size:
                prev_size = 0
            zmq_count = 0
            if log_path.exists():
                with log_path.open("r", encoding="utf-8") as log_file:
                    log_file.seek(prev_size)
                    new_entries = log_file.read()
                zmq_count = sum(
                    int(match) for match in re.findall(r"ZMQ received (\d+) node IDs", new_entries)
                )
                stageA = sum(
                    float(x)
                    for x in re.findall(r"Distance calculation E2E time: ([0-9.]+)s", new_entries)
                )
                stageBC = sum(
                    float(x) for x in re.findall(r"ZMQ E2E time: ([0-9.]+)s", new_entries)
                )
            else:
                stageA = 0.0
                stageBC = 0.0

            per_chunk = add_elapsed / len(update_chunks)
            print(
                f"Total time: {total_elapsed:.3f} s | add-only: {add_elapsed:.3f} s "
                f"for {len(update_chunks)} passages => {per_chunk * 1e3:.3f} ms/passage"
            )
            print(f"ZMQ node fetch total: {zmq_count}")
            results_total[name].append(total_elapsed)
            results_add[name].append(add_elapsed)
            results_zmq[name].append(zmq_count)
            results_ms_per_passage[name].append(per_chunk * 1e3)
            results_stageA[name].append(stageA)
            results_stageBC[name].append(stageBC)

            # Append row to CSV
            if args.csv_path:
                row = {
                    "run_id": run_id,
                    "scenario": name,
                    "cache_enabled": 1 if cache_enabled else 0,
                    "ef_construction": args.ef_construction,
                    "max_initial": args.max_initial,
                    "max_updates": args.max_updates,
                    "total_time_s": round(total_elapsed, 6),
                    "add_only_s": round(add_elapsed, 6),
                    "latency_ms_per_passage": round(per_chunk * 1e3, 6),
                    "zmq_nodes": int(zmq_count),
                    "stageA_time_s": round(stageA, 6),
                    "stageBC_time_s": round(stageBC, 6),
                    "model_name": args.model_name,
                    "embedding_mode": args.embedding_mode,
                    "distance_metric": args.distance_metric,
                }
                with args.csv_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fields)
                    writer.writerow(row)

    print("\n=== Summary ===")
    for name in results_add:
        add_values = results_add[name]
        total_values = results_total[name]
        zmq_values = results_zmq[name]
        latency_values = results_ms_per_passage[name]
        if not add_values:
            print(f"{name}: no successful runs")
            continue
        avg_add = sum(add_values) / len(add_values)
        avg_total = sum(total_values) / len(total_values)
        avg_zmq = sum(zmq_values) / len(zmq_values) if zmq_values else 0.0
        avg_latency = sum(latency_values) / len(latency_values) if latency_values else 0.0
        runs = len(add_values)
        print(
            f"{name}: add-only avg {avg_add:.3f} s | total avg {avg_total:.3f} s "
            f"| ZMQ avg {avg_zmq:.1f} node fetches | latency {avg_latency:.2f} ms/passage over {runs} run(s)"
        )

    if args.plot_path:
        try:
            import matplotlib.pyplot as plt

            labels = [name for name, *_ in scenarios]
            values = [
                sum(results_ms_per_passage[name]) / len(results_ms_per_passage[name])
                if results_ms_per_passage[name]
                else 0.0
                for name in labels
            ]

            def _auto_cap(vals: list[float]) -> float | None:
                s = sorted(vals, reverse=True)
                if len(s) < 2:
                    return None
                if s[1] > 0 and s[0] >= 2.5 * s[1]:
                    return s[1] * 1.1
                return None

            def _fmt_ms(v: float) -> str:
                return f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.1f}"

            colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2"]

            if args.broken_y:
                s = sorted(values, reverse=True)
                second = s[1] if len(s) >= 2 else (s[0] if s else 0.0)
                lower_cap = args.lower_cap_y if args.lower_cap_y is not None else second * 1.1
                upper_start = (
                    args.upper_start_y
                    if args.upper_start_y is not None
                    else max(second * 1.2, lower_cap * 1.02)
                )
                ymax = max(values) * 1.10 if values else 1.0
                _fig, (ax_top, ax_bottom) = plt.subplots(
                    2,
                    1,
                    sharex=True,
                    figsize=(7.4, 5.0),
                    gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05},
                )
                x = list(range(len(labels)))
                ax_bottom.bar(x, values, color=colors[: len(labels)], width=0.8)
                ax_top.bar(x, values, color=colors[: len(labels)], width=0.8)
                ax_bottom.set_ylim(0, lower_cap)
                ax_top.set_ylim(upper_start, ymax)
                for i, v in enumerate(values):
                    if v <= lower_cap:
                        ax_bottom.text(
                            i,
                            v + lower_cap * 0.02,
                            _fmt_ms(v),
                            ha="center",
                            va="bottom",
                            fontsize=9,
                        )
                    else:
                        ax_top.text(i, v, _fmt_ms(v), ha="center", va="bottom", fontsize=9)
                ax_top.spines["bottom"].set_visible(False)
                ax_bottom.spines["top"].set_visible(False)
                ax_top.tick_params(labeltop=False)
                ax_bottom.xaxis.tick_bottom()
                d = 0.015
                kwargs = {"transform": ax_top.transAxes, "color": "k", "clip_on": False}
                ax_top.plot((-d, +d), (-d, +d), **kwargs)
                ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
                kwargs.update({"transform": ax_bottom.transAxes})
                ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
                ax_bottom.set_xticks(range(len(labels)))
                ax_bottom.set_xticklabels(labels)
                ax = ax_bottom
            else:
                cap = args.cap_y or _auto_cap(values)
                plt.figure(figsize=(7.2, 4.2))
                ax = plt.gca()
                if cap is not None:
                    show_vals = [min(v, cap) for v in values]
                    bars = []
                    for i, (v, show) in enumerate(zip(values, show_vals)):
                        b = ax.bar(i, show, color=colors[i], width=0.8)
                        bars.append(b[0])
                        if v > cap:
                            bars[-1].set_hatch("//")
                            ax.text(i, cap * 1.02, _fmt_ms(v), ha="center", va="bottom", fontsize=9)
                        else:
                            ax.text(
                                i,
                                show + max(1.0, 0.01 * (cap or show)),
                                _fmt_ms(v),
                                ha="center",
                                va="bottom",
                                fontsize=9,
                            )
                    ax.set_ylim(0, cap * 1.10)
                    ax.plot(
                        [0.02 - 0.02, 0.02 + 0.02],
                        [0.98 + 0.02, 0.98 - 0.02],
                        transform=ax.transAxes,
                        color="k",
                        lw=1,
                    )
                    ax.plot(
                        [0.98 - 0.02, 0.98 + 0.02],
                        [0.98 + 0.02, 0.98 - 0.02],
                        transform=ax.transAxes,
                        color="k",
                        lw=1,
                    )
                    if any(v > cap for v in values):
                        ax.legend(
                            [bars[0]], ["capped"], fontsize=8, frameon=False, loc="upper right"
                        )
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels)
                else:
                    ax.bar(labels, values, color=colors[: len(labels)])
                    for idx, val in enumerate(values):
                        ax.text(idx, val + 1.0, f"{val:.1f}", ha="center", va="bottom")

            plt.ylabel("Average add latency (ms per passage)")
            plt.title(f"Initial passages {args.max_initial}, updates {args.max_updates}")
            plt.tight_layout()
            plt.savefig(args.plot_path)
            print(f"Saved latency bar plot to {args.plot_path}")
            # ZMQ time split (Stage A vs B/C)
            try:
                plt.figure(figsize=(6, 4))
                a_vals = [sum(results_stageA[n]) / max(1, len(results_stageA[n])) for n in labels]
                bc_vals = [
                    sum(results_stageBC[n]) / max(1, len(results_stageBC[n])) for n in labels
                ]
                ind = range(len(labels))
                plt.bar(ind, a_vals, color="#4e79a7", label="Stage A distance (s)")
                plt.bar(
                    ind, bc_vals, bottom=a_vals, color="#e15759", label="Stage B/C embed-by-id (s)"
                )
                plt.xticks(list(ind), labels, rotation=10)
                plt.ylabel("Server ZMQ time (s)")
                plt.title(
                    f"ZMQ time split (initial {args.max_initial}, updates {args.max_updates})"
                )
                plt.legend()
                out2 = args.plot_path.with_name(
                    args.plot_path.stem + "_zmq_split" + args.plot_path.suffix
                )
                plt.tight_layout()
                plt.savefig(out2)
                print(f"Saved ZMQ time split plot to {out2}")
            except Exception as e:
                print("Failed to plot ZMQ split:", e)
        except ImportError:
            print("matplotlib not available; skipping plot generation")

    # leave the last build on disk for inspection


if __name__ == "__main__":
    main()
