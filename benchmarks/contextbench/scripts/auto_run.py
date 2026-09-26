import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

import pexpect
from datasets import load_dataset
from git import Repo

DATASET_NAME = os.environ.get("DATASET_NAME", "Contextbench/ContextBench")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "train")


# Parse integer environment variables with validation and defaults.
def _get_int_env(name: str, default: int, min_value: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"⚠️ Invalid {name}={raw!r}; falling back to {default}")
        return default
    if value < min_value:
        print(f"⚠️ {name} must be >= {min_value}; falling back to {default}")
        return default
    return value


LEANN_TOP_K = _get_int_env("LEANN_TOP_K", 5, min_value=1)
USAGE_LOG_FILE = os.environ.get("USAGE_LOG_FILE", "task_session_usage.log").strip()
TASK_METRICS_FILE = os.environ.get("TASK_METRICS_FILE", "task_metrics.jsonl").strip()
PREFETCH_STRICT = os.environ.get("PREFETCH_STRICT", "1").strip() != "0"


def _resolve_leann_mode() -> str:
    mode = os.environ.get("LEANN_MODE", "").strip().lower()
    if not mode:
        return "mcp"
    if mode in {"mcp", "none"}:
        return mode
    print(f"⚠️ Invalid LEANN_MODE={mode!r}; falling back to 'mcp'")
    return "mcp"


# Get uncommitted changes compared to the current HEAD commit.
def get_git_diff(repo_dir: Path) -> str:
    try:
        repo = Repo(repo_dir)
        return repo.git.diff(repo.head.commit)
    except Exception as e:
        print(f"⚠️ Diff Error: {e}")
        return ""


# Prepare repository state for one ContextBench task instance.
def setup_task_environment(
    instance_id: str, repo_url: str, work_root: Path, task: Optional[dict] = None
) -> Path:
    print(f"🚀 [1/4] Preparing environment: {instance_id}")
    if not task:
        ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        task = next((x for x in ds if x["instance_id"] == instance_id), None)
        if not task:
            raise RuntimeError(f"Instance {instance_id} not found in dataset.")

    target_dir = work_root / instance_id
    if not target_dir.exists():
        Repo.clone_from(repo_url, target_dir)

    repo = Repo(target_dir)
    repo.git.reset("--hard")
    # Clear leftovers before checkout so untracked files cannot block it.
    repo.git.clean("-fdx", "-e", ".leann/")
    repo.git.checkout(task["base_commit"])
    repo.git.reset("--hard")
    # After checkout, ensure the base commit worktree is clean while preserving LEANN indexes.
    repo.git.clean("-fdx", "-e", ".leann/")
    (target_dir / "PROBLEM.md").write_text(task["problem_statement"], encoding="utf-8")
    return target_dir


# Pre-download all task repositories before model execution starts.
def prefetch_task_repositories(tasks: list[dict], work_root: Path) -> None:
    if not tasks:
        print("📦 [Prefetch] No tasks to download.")
        return

    print(f"📦 [Prefetch] Downloading repositories for {len(tasks)} tasks...")
    failures: list[str] = []

    for i, task in enumerate(tasks, start=1):
        instance_id = task.get("instance_id", "").strip()
        repo_url = task.get("repo_url", "").strip()
        target_dir = work_root / instance_id

        if not instance_id or not repo_url:
            failures.append(f"task[{i}] missing instance_id/repo_url")
            print(f"   [{i}/{len(tasks)}] ❌ Invalid task metadata; skipping.")
            continue

        if target_dir.exists():
            # A previous interrupted clone can leave a broken .git dir.
            # Treat only repos with a valid HEAD as reusable.
            try:
                repo = Repo(target_dir)
                repo.git.rev_parse("HEAD")
                print(f"   [{i}/{len(tasks)}] ✅ Already present: {instance_id}")
                continue
            except Exception:
                print(f"   [{i}/{len(tasks)}] ♻️ Found incomplete repo, re-cloning: {instance_id}")
                shutil.rmtree(target_dir, ignore_errors=True)

        cloned = False
        last_error: Optional[Exception] = None
        for attempt in range(1, 3 + 1):
            try:
                print(f"   [{i}/{len(tasks)}] ⬇️  Cloning: {instance_id} (attempt {attempt}/3)")
                Repo.clone_from(repo_url, target_dir)
                print(f"   [{i}/{len(tasks)}] ✅ Cloned: {instance_id}")
                cloned = True
                break
            except Exception as e:
                last_error = e
                print(f"   [{i}/{len(tasks)}] ❌ Clone attempt failed: {instance_id} -> {e}")
                shutil.rmtree(target_dir, ignore_errors=True)
                if attempt < 3:
                    sleep_s = 4 * attempt
                    print(f"   [{i}/{len(tasks)}] ⏳ Retrying in {sleep_s}s...")
                    time.sleep(sleep_s)

        if not cloned:
            failures.append(f"{instance_id} ({last_error})")

    if failures:
        sample = ", ".join(failures[:3])
        extra = f" (+{len(failures) - 3} more)" if len(failures) > 3 else ""
        message = f"Repository prefetch failed for {len(failures)} task(s): {sample}{extra}"
        if PREFETCH_STRICT:
            raise RuntimeError(message)
        print(f"⚠️ {message}")
        print("⚠️ PREFETCH_STRICT=0; continuing with available repositories.")
        print("⚠️ [Prefetch] Completed with partial failures.")
        return

    print("✅ [Prefetch] All repositories are ready.")


# Wait until a local TCP port starts accepting connections.
def wait_for_port(port: int, timeout: int = 10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    print(f"⚠️ Warning: Proxy port {port} did not open in {timeout}s")


# Resolve mitmproxy CA cert path for Python HTTPS clients behind proxy.
def _resolve_mitm_ca_cert() -> Optional[Path]:
    env_cert = os.environ.get("MITM_CA_CERT_PATH", "").strip()
    candidates = []
    if env_cert:
        candidates.append(Path(env_cert).expanduser())
    candidates.extend(
        [
            Path.home() / ".mitmproxy" / "mitmproxy-ca-cert.pem",
            Path.home() / ".mitmproxy" / "mitmproxy-ca-cert.cer",
        ]
    )
    for cert_path in candidates:
        if cert_path.exists():
            return cert_path
    return None


def _resolve_mitmdump_bin() -> str:
    env_bin = os.environ.get("MITMDUMP_BIN", "").strip()
    if env_bin:
        return env_bin

    # Prefer a dedicated local mitmproxy venv to avoid dependency conflicts
    # with the main project environment.
    project_root = Path(__file__).resolve().parents[1]
    local_bin = project_root / ".mitmproxy-venv" / "bin" / "mitmdump"
    if local_bin.exists():
        return str(local_bin)

    return "mitmdump"


# Start mitmproxy traffic recorder for the current task.
def start_mitmproxy(instance_id: str, mitm_script_path: str) -> subprocess.Popen:
    print("🕵️ [2/4] Starting Traffic Recorder...")
    env = os.environ.copy()
    env["TASK_INSTANCE"] = instance_id
    mitmdump_bin = _resolve_mitmdump_bin()
    cmd = [mitmdump_bin, "--no-http2", "-s", mitm_script_path, "-q"]
    try:
        process = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "mitmdump not found. Install mitmproxy in a dedicated env with:\n"
            "  python3 -m venv .mitmproxy-venv\n"
            "  .mitmproxy-venv/bin/python -m pip install mitmproxy\n"
            "or set MITMDUMP_BIN to an existing mitmdump path."
        ) from exc
    wait_for_port(8080)
    return process


# Check whether a server name exists in dict/list MCP server containers.
def _server_in_container(servers_obj, server_name: str) -> bool:
    if isinstance(servers_obj, dict):
        return server_name in servers_obj
    if isinstance(servers_obj, list):
        for item in servers_obj:
            if isinstance(item, dict) and item.get("name") == server_name:
                return True
    return False


# Recursively scan a config tree for a specific MCP server entry.
def _has_mcp_server_config(node, server_name: str, parent_key: str = "") -> bool:
    if isinstance(node, dict):
        for key, value in node.items():
            key_l = key.lower()
            if key_l in {"mcpservers", "mcp_servers"} and _server_in_container(value, server_name):
                return True
            if (
                parent_key.lower() == "mcp"
                and key_l == "servers"
                and _server_in_container(value, server_name)
            ):
                return True
            if _has_mcp_server_config(value, server_name, key):
                return True
    elif isinstance(node, list):
        for item in node:
            if _has_mcp_server_config(item, server_name, parent_key):
                return True
    return False


# Resolve Claude settings path that defines the required MCP server.
def resolve_claude_mcp_config(server_name: str) -> Optional[Path]:
    env_path = os.environ.get("CLAUDE_MCP_CONFIG_PATH", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            Path.home() / ".claude" / "settings.json",
            Path.home() / ".config" / "claude" / "settings.json",
            Path.home() / ".config" / "claude-code" / "settings.json",
            Path.home() / ".claude.json",
        ]
    )

    seen = set()
    for cfg_path in candidates:
        cfg_path = cfg_path.resolve()
        if str(cfg_path) in seen:
            continue
        seen.add(str(cfg_path))
        if not cfg_path.exists():
            continue
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _has_mcp_server_config(data, server_name):
            return cfg_path
    return None


def _normalize_mcp_servers(servers_obj) -> dict[str, dict]:
    servers: dict[str, dict] = {}
    if isinstance(servers_obj, dict):
        for name, config in servers_obj.items():
            if isinstance(name, str) and isinstance(config, dict):
                servers[name] = config
    elif isinstance(servers_obj, list):
        for item in servers_obj:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue
            config = dict(item)
            config.pop("name", None)
            servers[name] = config
    return servers


def _extract_mcp_servers(node, parent_key: str = "") -> dict[str, dict]:
    servers: dict[str, dict] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            key_l = key.lower()
            if key_l in {"mcpservers", "mcp_servers"}:
                servers.update(_normalize_mcp_servers(value))
                continue
            if parent_key.lower() == "mcp" and key_l == "servers":
                servers.update(_normalize_mcp_servers(value))
                continue
            servers.update(_extract_mcp_servers(value, key))
    elif isinstance(node, list):
        for item in node:
            servers.update(_extract_mcp_servers(item, parent_key))
    return servers


def build_strict_mcp_config_without_server(server_name: str) -> Optional[str]:
    cfg_path = resolve_claude_mcp_config(server_name)
    if not cfg_path:
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    servers = _extract_mcp_servers(data)
    servers.pop(server_name, None)
    return json.dumps({"mcpServers": servers}, separators=(",", ":"))


# Decide whether LEANN/MCP integration is available for this task repo.
def resolve_leann_integration(target_dir: Path) -> dict[str, str]:
    requested_mode = _resolve_leann_mode()
    mcp_server_name = os.environ.get("LEANN_MCP_SERVER", "leann-server")
    leann_index_exists = (target_dir / ".leann").exists()

    mode = "none"
    if requested_mode == "mcp" and leann_index_exists:
        mode = "mcp"
        print(
            f"   -> 🔍 LEANN MCP enabled (server: {mcp_server_name}, "
            f"forced top_k={LEANN_TOP_K})"
        )
    elif requested_mode == "mcp":
        print("   -> ⚠️ LEANN enabled but no .leann index found; continuing without LEANN")

    print(mode)
    return {
        "mode": mode,
        "mcp_server_name": mcp_server_name,
    }


# Build the initial Claude prompt from PROBLEM.md plus optional LEANN hints.
def build_initial_prompt(
    target_dir: Path, leann_info: dict[str, str], instance_id: str = ""
) -> str:
    problem_file = target_dir / "PROBLEM.md"
    try:
        problem_text = problem_file.read_text(encoding="utf-8").strip()
        if leann_info.get("mode") == "mcp":
            leann_info.get("mcp_server_name", "leann-server")
            mcp_hint = (
                f"Your LEANN index name is: {instance_id}\n"
                "Use LEANN MCP as a cost-aware semantic entry-point router. "
                "Before broad exploration, call leann_search once. "
                "Use top_k=5 by default, top_k=10 only for clearly multi-hop tasks spanning multiple subsystems; never use top_k>10. "
                "Use show_metadata=false.\n"
                "After LEANN, open the top 1-3 likely implementation/source files first, preferring source files over tests/docs/examples/generated files. "
                "Do not open many retrieved files just because they were returned. "
                "After identifying the likely fix location, use targeted Grep to find existing callers, API/server entry points, tests, and config files that may need to be updated. "
                "This targeted Grep is preferred over additional LEANN once concrete symbols, functions, file paths, config keys, or error strings are known.\n"
                "Run at most one additional leann_search only if no plausible implementation entry point is found or a required subsystem is clearly missing. "
                "The second query must be more specific, using concrete identifiers/literals found so far. "
                "Use show_metadata=false unless metadata is necessary to disambiguate files. "
                "Never run more than two leann_search calls total.\n"
            )
            return f"{mcp_hint}\n\n{problem_text}"
        return problem_text
    except Exception as e:
        raise RuntimeError(f"Failed to read problem statement from {problem_file}: {e}")


# Run Claude CLI autonomously in the prepared repository.
def run_claude_autonomous(
    target_dir: Path,
    model: str,
    leann_info: Optional[dict[str, str]] = None,
    instance_id: str = "",
):
    print("🤖 [3/4] Launching Claude Code")
    if (target_dir / ".claude").exists():
        shutil.rmtree(target_dir / ".claude")

    if leann_info is None:
        leann_info = resolve_leann_integration(target_dir)
    server_name = leann_info.get("mcp_server_name", "leann-server")
    if leann_info.get("mode") == "mcp":
        cfg_path = resolve_claude_mcp_config(server_name)
        if not cfg_path:
            raise RuntimeError(
                f"LEANN_MODE=mcp but MCP server '{server_name}' was not found in Claude config. "
                "Set CLAUDE_MCP_CONFIG_PATH or configure this server in Claude settings."
            )
        print(f"   -> ✅ Claude MCP config found for '{server_name}': {cfg_path}")
    initial_prompt = build_initial_prompt(target_dir, leann_info, instance_id=instance_id)

    env = os.environ.copy()
    env.update(
        {
            "HTTP_PROXY": "http://127.0.0.1:8080",
            "HTTPS_PROXY": "http://127.0.0.1:8080",
            "NODE_TLS_REJECT_UNAUTHORIZED": "0",
        }
    )

    claude_args = ["-p", initial_prompt, "--dangerously-skip-permissions"]
    if model:
        claude_args.extend(["--model", model])
    if leann_info.get("mode") != "mcp":
        filtered_mcp_config = build_strict_mcp_config_without_server(server_name)
        if filtered_mcp_config is not None:
            claude_args.extend(
                [
                    "--strict-mcp-config",
                    "--mcp-config",
                    filtered_mcp_config,
                ]
            )
            print(
                f"   -> 🚫 Non-LEANN mode; removed MCP server '{server_name}' from strict MCP config"
            )

    child = pexpect.spawn(
        "claude", claude_args, cwd=target_dir, env=env, encoding="utf-8", timeout=1200
    )

    try:
        while True:
            index = child.expect(
                [r"\[y/n\]", r"Allow execution", pexpect.EOF, pexpect.TIMEOUT], timeout=5
            )
            if index in [0, 1]:
                child.sendline("y")
            elif index == 2:
                break
    except Exception as e:
        print(f"❌ Interaction error: {e}")
    finally:
        if child.isalive():
            child.terminate(force=True)


# ---------------------------------------------------------------------------
# Trajectory extraction: parse mitmproxy trace → ContextBench traj_data
# ---------------------------------------------------------------------------


def _parse_sse_tool_uses(text: str) -> list[dict]:
    """Reconstruct tool_use calls from a streamed SSE response body."""
    tool_blocks: dict[int, dict] = {}  # index -> {name, partial_json}
    completed: list[dict] = []

    for line in text.split("\n"):
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        try:
            event = json.loads(data_str)
        except Exception:
            continue

        event_type = event.get("type", "")

        if event_type == "content_block_start":
            idx = event.get("index", 0)
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                tool_blocks[idx] = {"name": block.get("name", ""), "partial_json": ""}

        elif event_type == "content_block_delta":
            idx = event.get("index", 0)
            delta = event.get("delta", {})
            if idx in tool_blocks and delta.get("type") == "input_json_delta":
                tool_blocks[idx]["partial_json"] += delta.get("partial_json", "")

        elif event_type == "content_block_stop":
            idx = event.get("index", 0)
            if idx in tool_blocks:
                tool = tool_blocks.pop(idx)
                try:
                    input_data = json.loads(tool["partial_json"]) if tool["partial_json"] else {}
                except Exception:
                    input_data = {}
                completed.append({"name": tool["name"], "input": input_data})

    return completed


def _parse_json_tool_uses(text: str) -> list[dict]:
    """Extract tool_use calls from a non-streamed JSON response body."""
    try:
        resp = json.loads(text)
    except Exception:
        return []
    completed = []
    for block in resp.get("content", []):
        if block.get("type") == "tool_use":
            completed.append({"name": block.get("name", ""), "input": block.get("input", {})})
    return completed


def _make_relative(file_path: str, target_dir: Path) -> str:
    """Convert an absolute file path to one relative to target_dir."""
    try:
        return str(Path(file_path).relative_to(target_dir.resolve()))
    except ValueError:
        return file_path


def _is_leann_search_tool(name: str) -> bool:
    return bool(name) and (
        name == "leann_search" or name.endswith("__leann_search") or "leann_search" in name
    )


def _extract_trace_artifacts(
    trace_path: Path,
    target_dir: Path,
    since_timestamp: Optional[float] = None,
) -> dict:
    """
    Parse the mitmproxy JSONL trace once and derive both ContextBench trajectory data
    and LEANN usage statistics.

    Each Anthropic /messages response that contains tool_use calls becomes one pred_step.
    Read calls with line offsets are mapped to line spans; other file-touching tool calls
    (Glob, Grep, Bash cat/head) contribute to pred_files only.
    """
    pred_steps: list[dict] = []
    leann_search_calls = 0

    if not trace_path.exists():
        print(f"⚠️ Trace file not found: {trace_path}")
        return {
            "traj_data": _empty_traj_data(),
            "leann_tool_used": False,
            "leann_search_calls": 0,
        }

    with open(trace_path, encoding="utf-8") as f:
        for raw_line in f:
            try:
                entry = json.loads(raw_line)
            except Exception:
                continue

            ts = entry.get("timestamp")
            if (
                since_timestamp is not None
                and isinstance(ts, (int, float))
                and ts < since_timestamp
            ):
                continue

            url = entry.get("request", {}).get("url", "")
            if "anthropic.com" not in url or "/messages" not in url:
                continue

            response_text = entry.get("response", {}).get("text", "") or ""
            if not response_text:
                continue

            # Support both streaming (SSE) and non-streaming responses.
            if "data: " in response_text:
                tool_uses = _parse_sse_tool_uses(response_text)
            else:
                tool_uses = _parse_json_tool_uses(response_text)

            if not tool_uses:
                continue

            step_files: list[str] = []
            step_spans: dict[str, list[dict]] = {}

            for tool in tool_uses:
                name = tool.get("name", "")
                inp = tool.get("input", {}) or {}

                if _is_leann_search_tool(name):
                    leann_search_calls += 1

                if name == "Read":
                    raw_path = inp.get("file_path", "")
                    if not raw_path:
                        continue
                    rel = _make_relative(raw_path, target_dir)
                    if rel not in step_files:
                        step_files.append(rel)
                    offset = inp.get("offset")
                    limit = inp.get("limit")
                    if offset is not None:
                        start = int(offset)
                        end = start + int(limit) if limit is not None else start + 2000
                        step_spans.setdefault(rel, []).append({"start": start, "end": end})

                elif name in ("Grep", "Glob"):
                    # These don't return exact spans; record any explicit path argument.
                    raw_path = inp.get("path", "")
                    if raw_path and not raw_path.endswith(("*", "/")):
                        rel = _make_relative(raw_path, target_dir)
                        if rel not in step_files:
                            step_files.append(rel)

                elif name == "Bash":
                    # Heuristic: detect `cat`, `head`, `tail` calls on files.
                    cmd = inp.get("command", "") or ""
                    for m in re.finditer(
                        r"(?:cat|head|tail|sed|awk)\s+(?:-[^\s]+\s+)*([^\s|><;]+)", cmd
                    ):
                        candidate = m.group(1)
                        if "/" in candidate or candidate.endswith(
                            (".py", ".go", ".ts", ".js", ".java", ".c", ".cpp", ".rs")
                        ):
                            rel = _make_relative(candidate, target_dir)
                            if rel not in step_files:
                                step_files.append(rel)

            if step_files:
                pred_steps.append(
                    {
                        "files": step_files,
                        "spans": step_spans,
                        "symbols": {},
                    }
                )

    # Aggregate across all steps.
    all_files: list[str] = []
    all_spans: dict[str, list[dict]] = {}
    for step in pred_steps:
        for f in step["files"]:
            if f not in all_files:
                all_files.append(f)
        for f, spans in step["spans"].items():
            all_spans.setdefault(f, []).extend(spans)

    return {
        "traj_data": {
            "pred_steps": pred_steps,
            "pred_files": all_files,
            "pred_spans": all_spans,
            "pred_symbols": {},
        },
        "leann_tool_used": leann_search_calls > 0,
        "leann_search_calls": leann_search_calls,
    }


def extract_trajectory_from_traces(
    trace_path: Path,
    target_dir: Path,
    since_timestamp: Optional[float] = None,
) -> dict:
    return _extract_trace_artifacts(
        trace_path,
        target_dir,
        since_timestamp=since_timestamp,
    )["traj_data"]


def detect_messages_api_error(
    trace_path: Path,
    since_timestamp: Optional[float] = None,
) -> Optional[str]:
    """Return a concise API error message if /v1/messages requests failed."""
    if not trace_path.exists():
        return None
    try:
        with open(trace_path, encoding="utf-8") as f:
            for raw_line in f:
                try:
                    entry = json.loads(raw_line)
                except Exception:
                    continue
                ts = entry.get("timestamp")
                if (
                    since_timestamp is not None
                    and isinstance(ts, (int, float))
                    and ts < since_timestamp
                ):
                    continue

                url = entry.get("request", {}).get("url", "")
                if "/v1/messages" not in url:
                    continue
                response = entry.get("response", {}) or {}
                status_code = response.get("status_code")
                if not isinstance(status_code, int) or status_code < 400:
                    continue
                text = response.get("text", "") or ""
                err_type = ""
                err_message = ""
                try:
                    payload = json.loads(text)
                    error_obj = payload.get("error", {}) if isinstance(payload, dict) else {}
                    if isinstance(error_obj, dict):
                        err_type = str(error_obj.get("type", "") or "")
                        err_message = str(error_obj.get("message", "") or "")
                except Exception:
                    pass
                detail = f"/v1/messages returned HTTP {status_code}"
                if err_type:
                    detail += f" ({err_type})"
                if err_message:
                    detail += f": {err_message}"
                return detail
    except Exception:
        return None
    return None


def _empty_traj_data() -> dict:
    return {"pred_steps": [], "pred_files": [], "pred_spans": {}, "pred_symbols": {}}


# ---------------------------------------------------------------------------
# Usage tracking (mirrored from swebench runner)
# ---------------------------------------------------------------------------


def _blank_usage() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }


def _normalize_usage(raw: Optional[dict]) -> dict:
    base = _blank_usage()
    if not isinstance(raw, dict):
        return base
    base["input_tokens"] = int(raw.get("inputTokens", 0) or 0)
    base["output_tokens"] = int(raw.get("outputTokens", 0) or 0)
    base["cache_creation_tokens"] = int(raw.get("cacheCreationTokens", 0) or 0)
    base["cache_read_tokens"] = int(raw.get("cacheReadTokens", 0) or 0)
    base["total_tokens"] = int(raw.get("totalTokens", 0) or 0)
    base["cost_usd"] = float(raw.get("totalCost", 0) or 0.0)
    return base


def _usage_delta(before: dict, after: dict) -> dict:
    return {
        "input_tokens": after["input_tokens"] - before["input_tokens"],
        "output_tokens": after["output_tokens"] - before["output_tokens"],
        "cache_creation_tokens": after["cache_creation_tokens"] - before["cache_creation_tokens"],
        "cache_read_tokens": after["cache_read_tokens"] - before["cache_read_tokens"],
        "total_tokens": after["total_tokens"] - before["total_tokens"],
        "cost_usd": round(after["cost_usd"] - before["cost_usd"], 6),
    }


def _has_positive_delta(usage: dict) -> bool:
    return (
        usage["total_tokens"] > 0
        or usage["input_tokens"] > 0
        or usage["output_tokens"] > 0
        or usage["cache_creation_tokens"] > 0
        or usage["cache_read_tokens"] > 0
        or usage["cost_usd"] > 0
    )


def _format_usage(usage: dict) -> str:
    return (
        f"input: {usage['input_tokens']:,}, output: {usage['output_tokens']:,}, "
        f"cache_read: {usage['cache_read_tokens']:,}, cache_create: {usage['cache_creation_tokens']:,} "
        f"total: {usage['total_tokens']:,}, cost: ${usage['cost_usd']:.4f}"
    )


def append_usage_log(line: str) -> None:
    if not USAGE_LOG_FILE:
        return
    log_path = Path(USAGE_LOG_FILE).expanduser()
    try:
        if log_path.parent and str(log_path.parent) != ".":
            log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{line}\n")
    except Exception as e:
        print(f"⚠️ Failed to write usage log {log_path}: {e}")


def append_task_metrics(entry: dict) -> None:
    if not TASK_METRICS_FILE:
        return
    metrics_path = Path(TASK_METRICS_FILE).expanduser()
    try:
        if metrics_path.parent and str(metrics_path.parent) != ".":
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"⚠️ Failed to write task metrics {metrics_path}: {e}")


def get_ccusage_sessions() -> Optional[dict[str, dict]]:
    try:
        result = subprocess.run(
            ["npx", "ccusage", "session", "--json", "--offline"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(stderr or "ccusage returned non-zero exit code")
        data = json.loads(result.stdout or "{}")
        sessions = data.get("sessions", [])
        if not isinstance(sessions, list):
            return {}
        by_session: dict[str, dict] = {}
        for row in sessions:
            if not isinstance(row, dict):
                continue
            session_id = str(row.get("sessionId", "")).strip()
            if not session_id:
                continue
            by_session[session_id] = _normalize_usage(row)
        return by_session
    except Exception as e:
        print(f"⚠️ Failed to get ccusage sessions: {e}")
        return None


def compute_usage_diff(
    before_sessions: Optional[dict[str, dict]],
    after_sessions: Optional[dict[str, dict]],
    instance_id: str,
) -> Optional[dict]:
    if before_sessions is None or after_sessions is None:
        return None

    changed_sessions: dict[str, dict] = {}
    all_session_ids = set(before_sessions.keys()) | set(after_sessions.keys())
    for session_id in all_session_ids:
        before = before_sessions.get(session_id, _blank_usage())
        after = after_sessions.get(session_id, _blank_usage())
        delta = _usage_delta(before, after)
        if _has_positive_delta(delta):
            changed_sessions[session_id] = delta

    if not changed_sessions:
        return None

    instance_lower = instance_id.lower()
    subagent_ids = [
        sid for sid in changed_sessions if sid.lower() == "subagents" or "subagent" in sid.lower()
    ]
    non_subagent_items = [
        (sid, usage) for sid, usage in changed_sessions.items() if sid not in subagent_ids
    ]
    matched_non_subagent = [
        (sid, usage)
        for sid, usage in non_subagent_items
        if sid.lower() in instance_lower or instance_lower in sid.lower()
    ]
    primary_candidates = matched_non_subagent or non_subagent_items
    if not primary_candidates:
        return None

    primary_session_id, session_usage = max(
        primary_candidates, key=lambda x: (x[1]["total_tokens"], x[1]["cost_usd"])
    )
    return {
        "session_id": primary_session_id,
        "input_tokens": session_usage["input_tokens"],
        "output_tokens": session_usage["output_tokens"],
        "cache_creation_tokens": session_usage["cache_creation_tokens"],
        "cache_read_tokens": session_usage["cache_read_tokens"],
        "total_tokens": session_usage["total_tokens"],
        "cost_usd": session_usage["cost_usd"],
        "session_usage": session_usage,
    }


# ---------------------------------------------------------------------------
# Main task runner
# ---------------------------------------------------------------------------


def run_single_task(
    instance_id: str,
    repo_url: str,
    work_root: Path,
    mitm_script_path: str,
    trace_dir: Path,
    model: str = "",
    task: Optional[dict] = None,
) -> tuple:
    """
    Run one ContextBench instance end-to-end.

    Returns: (patch, latency_seconds, agent_seconds, traj_data, usage)
      - patch      : git diff string (the model's code changes)
      - latency_seconds : end-to-end wall-clock seconds
      - agent_seconds   : Claude agent runtime only
      - traj_data  : ContextBench trajectory dict (pred_steps, pred_files, …)
      - usage      : token/cost usage dict, or None
    """
    print(f"\n🌊 Starting Claude Pipeline for {instance_id}")
    work_root = Path(work_root)
    trace_dir = Path(trace_dir)
    mitm_proc = None
    start_time = time.time()
    usage_before: Optional[dict[str, dict]] = None
    usage: Optional[dict] = None
    status = "failed"
    error_message: Optional[str] = None
    claude_start_ts: Optional[float] = None
    claude_end_ts: Optional[float] = None
    leann_mode: Optional[str] = None
    leann_tool_used = False
    leann_search_calls = 0

    try:
        target_dir = setup_task_environment(instance_id, repo_url, work_root, task=task)
        leann_info = resolve_leann_integration(target_dir)
        leann_mode = leann_info.get("mode")
        mitm_proc = start_mitmproxy(instance_id, str(mitm_script_path))

        usage_before = get_ccusage_sessions()
        claude_start_ts = time.time()
        run_claude_autonomous(target_dir, model, leann_info=leann_info, instance_id=instance_id)
        claude_end_ts = time.time()
        usage_after = get_ccusage_sessions()

        trace_path = trace_dir / f"{instance_id}_trace.jsonl"
        api_error = detect_messages_api_error(trace_path, since_timestamp=claude_start_ts)
        if api_error:
            raise RuntimeError(
                f"Claude API request failed for {instance_id}: {api_error}. "
                "Set MODEL/CLAUDE_MODEL to an available model for your account."
            )

        usage = compute_usage_diff(usage_before, usage_after, instance_id=instance_id)
        if usage:
            task_usage_line = (
                f"💰 Task session usage ({usage['session_id']}) — "
                f"{_format_usage(usage['session_usage'])}"
            )
            print(task_usage_line)

        print(f"📍 [4/4] Extracting trajectory from trace: {trace_path.name}")
        trace_artifacts = _extract_trace_artifacts(
            trace_path,
            target_dir,
            since_timestamp=claude_start_ts,
        )
        traj_data = trace_artifacts["traj_data"]
        leann_tool_used = bool(trace_artifacts["leann_tool_used"])
        leann_search_calls = int(trace_artifacts["leann_search_calls"])
        print(
            f"   -> {len(traj_data['pred_steps'])} steps, "
            f"{len(traj_data['pred_files'])} unique files, "
            f"leann_calls={leann_search_calls}"
        )

        patch = get_git_diff(target_dir)
        elapsed = time.time() - start_time
        agent_elapsed = (
            max(0.0, claude_end_ts - claude_start_ts)
            if claude_start_ts is not None and claude_end_ts is not None
            else None
        )
        print(f"⏱️  Task completed in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
        status = "succeeded"
        return patch, elapsed, agent_elapsed, traj_data, usage
    except Exception as e:
        error_message = str(e)
        raise

    finally:
        elapsed = time.time() - start_time
        agent_elapsed = None
        if claude_start_ts is not None:
            agent_end = claude_end_ts if claude_end_ts is not None else time.time()
            agent_elapsed = max(0.0, agent_end - claude_start_ts)
        if usage is None and usage_before is not None:
            usage_after = get_ccusage_sessions()
            usage = compute_usage_diff(usage_before, usage_after, instance_id=instance_id)

        metrics_entry: dict = {
            "instance_id": instance_id,
            "repo_url": repo_url,
            "model": model or "cli-default",
            "status": status,
            "latency_seconds": round(elapsed, 1),
            "timestamp_unix": int(time.time()),
            "token_usage": usage,
            "leann_mode": leann_mode,
            "leann_tool_used": leann_tool_used,
            "leann_search_calls": leann_search_calls,
        }
        if agent_elapsed is not None:
            metrics_entry["agent_seconds"] = round(agent_elapsed, 1)
        if error_message:
            metrics_entry["error"] = error_message
        append_task_metrics(metrics_entry)

        metric_line = (
            f"📊 Task metrics ({instance_id}) — status={status}, "
            f"agent={agent_elapsed:.1f}s, latency={elapsed:.1f}s"
            if agent_elapsed is not None
            else f"📊 Task metrics ({instance_id}) — status={status}, latency={elapsed:.1f}s"
        )
        if usage:
            metric_line += (
                f", session={usage['session_id']}, {_format_usage(usage['session_usage'])}"
            )
        else:
            metric_line += ", token usage unavailable"
        metric_line += (
            f", leann_mode={leann_mode}, "
            f"leann_tool_used={str(leann_tool_used).lower()}, "
            f"leann_search_calls={leann_search_calls}"
        )
        if error_message:
            metric_line += f", error={error_message}"
        print(metric_line)
        append_usage_log(metric_line)

        if mitm_proc:
            os.killpg(os.getpgid(mitm_proc.pid), signal.SIGTERM)
