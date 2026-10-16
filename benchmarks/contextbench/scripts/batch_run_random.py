import json
import os
import random
import subprocess
import time
from pathlib import Path

from auto_run import prefetch_task_repositories, run_single_task
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "all_predictions_claude.jsonl")
NUM_TASKS = int(os.environ.get("NUM_TASKS", "31"))
WORK_ROOT = os.environ.get("WORK_ROOT", "contextbench_work_dir_claude")
MODEL = os.environ.get("MODEL", os.environ.get("CLAUDE_MODEL", "")).strip()
DATASET_NAME = os.environ.get("DATASET_NAME", "Contextbench/ContextBench")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "train")
# Optionally restrict to one benchmark split: Verified | Pro | Poly | Multi
BENCH_FILTER = os.environ.get("BENCH_FILTER", "Pro").strip()
# Optionally restrict random sampling to one repo (supports partial match),
# e.g. "django/django" or "sympy".
REPO_FILTER = os.environ.get("REPO_FILTER", "").strip().lower()
PREFETCH_REPOS = os.environ.get("PREFETCH_REPOS", "4").strip() != "0"
MITM_SCRIPT = ROOT / "mitmproxy_addons" / "trace_recorder.py"
TRACE_DIR = ROOT / "traces" / "raw"


def cleanup_residuals():
    print("🧹 Cleaning up residual processes (Claude & Mitm)...")
    try:
        subprocess.run(["pkill", "-f", "claude"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "mitmdump"], stderr=subprocess.DEVNULL)
        time.sleep(2)
    except Exception:
        pass


def main():
    Path(WORK_ROOT).mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🧠 Model: {MODEL or '(Claude CLI default)'}")
    print(f"🔎 Bench filter: {BENCH_FILTER or '(all)'}")
    if REPO_FILTER:
        print(f"📁 Repo filter: {REPO_FILTER}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        print(f"🔑 Using API key from environment: {api_key[:10]}...")
    else:
        print("🔐 ANTHROPIC_API_KEY not set; using Claude CLI logged-in session.")

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

    pending_tasks = [
        t
        for t in ds
        if t["instance_id"] not in existing_ids
        and (not BENCH_FILTER or t.get("source", "") == BENCH_FILTER)
        and (
            not REPO_FILTER
            or REPO_FILTER in (t.get("repo", "") or "").lower()
            or REPO_FILTER in (t.get("repo_url", "") or "").lower()
        )
    ]

    if not pending_tasks:
        print("🎉 No pending tasks to run!")
        return

    selected_tasks = random.sample(pending_tasks, min(NUM_TASKS, len(pending_tasks)))
    print(f"🚀 Randomly selected {len(selected_tasks)} tasks to process.")
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

        # If want to run the task, uncomment the following code
    #     try:
    #         patch, elapsed, traj_data, usage = run_single_task(
    #             instance_id=instance_id,
    #             repo_url=repo_url,
    #             work_root=WORK_ROOT,
    #             mitm_script_path=str(MITM_SCRIPT),
    #             trace_dir=TRACE_DIR,
    #             model=MODEL,
    #             task=task,
    #         )

    #         result_entry = {
    #             "instance_id": instance_id,
    #             "model_patch": patch if patch else "",
    #             "model_name_or_path": "claude-code-cli",
    #             "elapsed_seconds": round(elapsed, 1),
    #             "traj_data": traj_data,
    #             "token_usage": usage,
    #         }

    #         with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    #             f.write(json.dumps(result_entry) + "\n")
    #         print(f"✅ Result saved for {instance_id}")
    #         success_count += 1

    #     except Exception as e:
    #         print(f"❌ Error processing {instance_id}: {e}")
    #         failure_count += 1
    #     finally:
    #         cleanup_residuals()
    #         print("💤 Cooldown...")
    #         time.sleep(20)

    # print(
    #     f"\n✅ Finished {len(selected_tasks)} random tasks: "
    #     f"{success_count} succeeded, {failure_count} failed. "
    #     f"Results in {OUTPUT_FILE}"
    # )


if __name__ == "__main__":
    main()
