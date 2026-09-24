#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path

_base_dir: str | None = None


def _working_dir() -> Path:
    return Path(_base_dir) if _base_dir else Path.cwd()


def _leann_cmd() -> list[str]:
    """Build the base command for invoking ``leann`` CLI.

    Using ``sys.executable -m leann`` guarantees we find the CLI even when
    the ``leann`` console-script is not on PATH (common on Windows when
    launched by mcp-proxy or other service wrappers).
    """
    return [sys.executable, "-m", "leann"]


def _run_leann(*args, timeout=120):
    """Run a leann CLI command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [*_leann_cmd(), *args],
        capture_output=True,
        text=True,
        # The leann CLI prints UTF-8 (emoji, CJK, ...). Decode explicitly:
        # text=True alone falls back to the locale encoding (e.g. GBK on
        # Chinese Windows) and crashes the subprocess reader thread.
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=_working_dir(),
    )
    return result.returncode, result.stdout, result.stderr


def _make_result(request_id, content_text):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"content": [{"type": "text", "text": content_text}]},
    }


def _make_error(request_id, message):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -1, "message": message},
    }


TOOLS = [
    {
        "name": "leann_search",
        "description": (
            "Semantic code search across an indexed codebase. Returns matching code "
            "chunks with file paths, scores, and surrounding context.\n\n"
            "Use this to find relevant code before making changes — understand existing "
            "patterns, locate implementations, and discover related files.\n\n"
            "Examples: 'authentication middleware', 'database connection pooling', "
            "'error handling in API routes', 'how are embeddings computed'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "index_name": {
                    "type": "string",
                    "description": "Name of the LEANN index to search. Use leann_list to see available indexes.",
                },
                "query": {
                    "type": "string",
                    "description": "Natural language or technical search query.",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Number of results to return (default 5).",
                },
                "complexity": {
                    "type": "integer",
                    "default": 32,
                    "minimum": 16,
                    "maximum": 128,
                    "description": "Search precision level (default 32, use 64+ for thorough search).",
                },
            },
            "required": ["index_name", "query"],
        },
    },
    {
        "name": "leann_list",
        "description": "List all available LEANN indexes across projects. Shows index names, status, size, and location.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "leann_build",
        "description": (
            "Build or incrementally update a LEANN index for a codebase. "
            "If the index already exists, only new/modified/deleted files are processed "
            "(incremental update). Use this to keep the index current after code changes.\n\n"
            "Provide file paths or directories to index. For git repos, pass the output "
            "of 'git ls-files' as individual paths."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "index_name": {
                    "type": "string",
                    "description": "Name for the index (e.g., 'my-project'). Defaults to current directory name if omitted.",
                },
                "docs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths or directories to index.",
                },
                "backend_name": {
                    "type": "string",
                    "enum": ["hnsw", "ivf"],
                    "default": "ivf",
                    "description": "Index backend. 'ivf' supports incremental updates (recommended). 'hnsw' is faster for search but limited incremental support.",
                },
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "Force full rebuild instead of incremental update.",
                },
            },
            "required": ["docs"],
        },
    },
    {
        "name": "leann_status",
        "description": (
            "Show detailed status of a LEANN index: backend type, embedding model, "
            "number of chunks, file count, index size, and whether the index is up to date."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "index_name": {
                    "type": "string",
                    "description": "Name of the index to inspect.",
                },
            },
            "required": ["index_name"],
        },
    },
]


def handle_search(request_id, args):
    index_name = args.get("index_name", "")
    query = args.get("query", "")
    if not index_name or not query:
        return _make_result(request_id, "Error: Both index_name and query are required.")

    top_k = args.get("top_k", 5)
    complexity = args.get("complexity", 32)

    rc, stdout, stderr = _run_leann(
        "search",
        index_name,
        query,
        f"--top-k={top_k}",
        f"--complexity={complexity}",
        "--json",
        "--show-metadata",
        "--non-interactive",
    )

    if rc != 0:
        return _make_result(request_id, f"Search failed: {stderr.strip()}")

    # Parse JSON results and format for code context
    try:
        results = json.loads(stdout)
    except json.JSONDecodeError:
        # Fallback to raw output if --json isn't available
        return _make_result(
            request_id, stdout if stdout.strip() else f"Search failed: {stderr.strip()}"
        )

    if not results:
        return _make_result(request_id, f"No results found for '{query}'.")

    formatted = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        file_path = meta.get("file_path") or meta.get("source", "unknown")
        score = r.get("score", 0)
        text = r.get("text", "").strip()
        formatted.append(f"### Result {i} — {file_path} (score: {score:.3f})\n```\n{text}\n```")

    header = f"Found {len(results)} results for '{query}':\n"
    return _make_result(request_id, header + "\n\n".join(formatted))


def handle_list(request_id):
    rc, stdout, stderr = _run_leann("list")
    if rc != 0:
        return _make_result(request_id, f"Error listing indexes: {stderr.strip()}")
    return _make_result(request_id, stdout)


def handle_build(request_id, args):
    docs = args.get("docs", [])
    if not docs:
        return _make_result(
            request_id, "Error: 'docs' parameter is required (list of file paths or directories)."
        )

    cmd = ["build"]

    index_name = args.get("index_name")
    if index_name:
        cmd.append(index_name)

    cmd.extend(["--docs", *docs])

    backend = args.get("backend_name", "ivf")
    cmd.extend([f"--backend-name={backend}"])

    if args.get("force", False):
        cmd.append("--force")

    # Auto-detect embedding model from existing index to ensure incremental
    # updates use the same model (avoids mismatch with CLI default).
    if index_name:
        import json as _json

        meta_path = _working_dir() / ".leann" / "indexes" / index_name / "documents.leann.meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = _json.load(f)
                existing_model = meta.get("embedding_model")
                existing_mode = meta.get("embedding_mode")
                if existing_model:
                    cmd.extend([f"--embedding-model={existing_model}"])
                if existing_mode:
                    cmd.extend([f"--embedding-mode={existing_mode}"])
            except (OSError, ValueError):
                pass

    rc, stdout, stderr = _run_leann(*cmd, timeout=600)

    if rc != 0:
        return _make_result(request_id, f"Build failed:\n{stderr.strip()}\n{stdout.strip()}")

    return _make_result(request_id, stdout if stdout.strip() else "Build completed successfully.")


def handle_status(request_id, args):
    index_name = args.get("index_name", "")
    if not index_name:
        return _make_result(request_id, "Error: index_name is required.")

    # Check standard location
    leann_dir = _working_dir() / ".leann" / "indexes" / index_name
    meta_path = leann_dir / "documents.leann.meta.json"
    passages_path = leann_dir / "documents.leann.passages.jsonl"

    if not meta_path.exists():
        return _make_result(request_id, f"Index '{index_name}' not found at {leann_dir}")

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        return _make_result(request_id, f"Error reading index metadata: {e}")

    # Count passages
    num_chunks = 0
    file_paths = set()
    if passages_path.exists():
        with open(passages_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                num_chunks += 1
                try:
                    passage = json.loads(line)
                    meta = passage.get("metadata", {})
                    fp = meta.get("file_path") or meta.get("source", "")
                    if fp:
                        file_paths.add(fp)
                except json.JSONDecodeError:
                    pass

    # Calculate total index size
    total_size = 0
    if leann_dir.exists():
        for f in leann_dir.iterdir():
            if f.is_file():
                total_size += f.stat().st_size

    size_mb = total_size / (1024 * 1024)

    backend = meta.get("backend_name", "unknown")
    embedding_model = meta.get("embedding_model", "unknown")
    embedding_mode = meta.get("embedding_mode", "unknown")
    dimensions = meta.get("dimensions", "unknown")

    status_lines = [
        f"Index: {index_name}",
        f"Backend: {backend}",
        f"Embedding: {embedding_model} ({embedding_mode})",
        f"Dimensions: {dimensions}",
        f"Chunks: {num_chunks}",
        f"Files indexed: {len(file_paths)}",
        f"Size: {size_mb:.1f} MB",
        f"Location: {leann_dir}",
    ]

    return _make_result(request_id, "\n".join(status_lines))


def handle_request(request):
    method = request.get("method")
    request_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "capabilities": {"tools": {}},
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "leann-mcp", "version": "2.0.0"},
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "ping":
        # MCP requires a ping request to be answered with an empty result.
        # Without this it falls through to the -32601 branch below, which tells
        # a client its liveness probe is unsupported — a different wrong answer
        # from the silence #384 fixed, but still one that breaks keepalives
        # (issue #403).
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {},
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        tool_name = request["params"]["name"]
        args = request["params"].get("arguments", {})

        try:
            if tool_name == "leann_search":
                return handle_search(request_id, args)
            elif tool_name == "leann_list":
                return handle_list(request_id)
            elif tool_name == "leann_build":
                return handle_build(request_id, args)
            elif tool_name == "leann_status":
                return handle_status(request_id, args)
            else:
                return _make_error(request_id, f"Unknown tool: {tool_name}")
        except subprocess.TimeoutExpired:
            return _make_result(request_id, "Error: Command timed out.")
        except Exception as e:
            return _make_error(request_id, str(e))

    # Unknown method: a request (with an id) must get a -32601 error response
    # so strict clients can fall back — e.g. Google Antigravity CLI probes
    # with a server/discover request before initialize and hangs forever
    # ("initializing...") if the server stays silent. Notifications (no id)
    # must get no reply at all (JSON-RPC 2.0).
    if request_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "Method not found"},
        }
    return None


def main():
    global _base_dir

    parser = argparse.ArgumentParser(description="LEANN MCP server (stdio)")
    parser.add_argument(
        "--base-dir",
        help="Base directory where LEANN indexes are located. "
        "The leann CLI will run with this as its working directory.",
    )
    cli_args = parser.parse_args()
    _base_dir = cli_args.base_dir

    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = handle_request(request)
            if response:
                print(json.dumps(response))
                sys.stdout.flush()
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -1, "message": str(e)},
            }
            print(json.dumps(error_response))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
