import json
import os
import subprocess
import time
from pathlib import Path

from auto_run import prefetch_task_repositories, run_single_task
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "all_predictions_claude.jsonl")
WORK_ROOT = os.environ.get("WORK_ROOT", "contextbench_work_dir_claude")
MODEL = os.environ.get("MODEL", os.environ.get("CLAUDE_MODEL", "")).strip()
DATASET_NAME = os.environ.get("DATASET_NAME", "Contextbench/ContextBench")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "train")
BENCH_FILTER = os.environ.get("BENCH_FILTER", "").strip()  # e.g. "Verified", "Pro", "Poly", "Multi"
PREFETCH_REPOS = os.environ.get("PREFETCH_REPOS", "1").strip() != "0"
MITM_SCRIPT = ROOT / "mitmproxy_addons" / "trace_recorder.py"
TRACE_DIR = ROOT / "traces" / "raw"

# Instances to run, set instance_ids here
SELECTED_IDS = [
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


def cleanup_residuals():
    print("🧹 Cleaning up residual processes (Claude & Mitm)...")
    try:
        subprocess.run(["pkill", "-f", "claude"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "mitmdump"], stderr=subprocess.DEVNULL)
        time.sleep(2)
    except Exception:
        pass


def main():
    if not SELECTED_IDS:
        print(
            "⚠️ Warning: SELECTED_IDS list is empty. Add instance IDs to the script or set SELECTED_IDS env var."
        )
        return

    Path(WORK_ROOT).mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🧠 Model: {MODEL or '(Claude CLI default)'}")
    if BENCH_FILTER:
        print(f"🔎 Bench filter: {BENCH_FILTER}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        print(f"🔑 Using API key from environment: {api_key[:10]}...")
    else:
        print("🔐 ANTHROPIC_API_KEY not set; using Claude CLI logged-in session.")

    # Load already-completed instance IDs to support resuming.
    existing_ids: set = set()
    output_path = Path(OUTPUT_FILE)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_ids.add(data["instance_id"])
                except Exception:
                    continue
    print(f"✅ Found {len(existing_ids)} already completed tasks.")

    print(f"📚 Loading dataset: {DATASET_NAME} ({DATASET_SPLIT})...")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    # Build a lookup dict for fast access.
    task_lookup = {t["instance_id"]: t for t in ds}

    selected_tasks = []
    for iid in SELECTED_IDS:
        if iid in existing_ids:
            print(f"⏩ Skipping {iid} (already completed)")
            continue
        task = task_lookup.get(iid)
        if task is None:
            print(f"⚠️ Instance {iid} not found in dataset; skipping.")
            continue
        if BENCH_FILTER and task.get("source", "") != BENCH_FILTER:
            print(f"⏩ Skipping {iid} (source={task.get('source')} != {BENCH_FILTER})")
            continue
        selected_tasks.append(task)

    if not selected_tasks:
        print("🎉 No pending selected tasks to run!")
        return

    print(f"🚀 Selected {len(selected_tasks)} tasks to process.")
    if PREFETCH_REPOS:
        prefetch_task_repositories(selected_tasks, Path(WORK_ROOT))
    else:
        print("⏭️ PREFETCH_REPOS=0; skipping prefetch step.")
    success_count = 0
    failure_count = 0

    for i, task in enumerate(selected_tasks):
        instance_id = task["instance_id"]
        repo_url = task["repo_url"]

        print(f"\n{'-' * 60}")
        print(f"📦 [{i + 1}/{len(selected_tasks)}] Running: {instance_id}")
        print(f"   repo: {repo_url}  source: {task.get('source', '?')}")

        try:
            patch, elapsed, agent_seconds, traj_data, usage = run_single_task(
                instance_id=instance_id,
                repo_url=repo_url,
                work_root=WORK_ROOT,
                mitm_script_path=str(MITM_SCRIPT),
                trace_dir=TRACE_DIR,
                model=MODEL,
                task=task,
            )

            result_entry = {
                "instance_id": instance_id,
                "model_patch": patch if patch else "",
                "model_name_or_path": "claude-code-cli",
                "elapsed_seconds": round(elapsed, 1),
                "latency_seconds": round(elapsed, 1),
                "agent_seconds": round(agent_seconds, 1) if agent_seconds is not None else None,
                "traj_data": traj_data,
                "token_usage": usage,
            }

            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_entry) + "\n")
            print(f"✅ Result saved for {instance_id}")
            success_count += 1

        except Exception as e:
            print(f"❌ Error processing {instance_id}: {e}")
            failure_count += 1
        finally:
            cleanup_residuals()
            print("💤 Cooldown...")
            time.sleep(20)

    print(
        f"\n✅ Finished {len(selected_tasks)} selected tasks: "
        f"{success_count} succeeded, {failure_count} failed. "
        f"Results in {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
