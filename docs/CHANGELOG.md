# Changelog

Append-only log of major changes to LEANN (new features, breaking changes, important
fixes). Newest entries at the bottom.

## 2026-03-05: IVF backend incremental update support

- Added `leann-backend-ivf` with FAISS IndexIVFFlat + DirectMap.Hashtable.
- IVF supports in-place `add_vectors` and `remove_ids` without full rebuild.
- `leann build` is now idempotent: re-running on an existing index does incremental update (add new, remove deleted, re-index modified files).
- Fixed incremental build chunking inconsistency and shared metadata dict bug.
- Fixed IVF incremental update duplicate chunks from stale `passages.jsonl`.

## 2026-03-05: MCP server v2 — build, status, and structured search

- Added `leann_build` MCP tool: build or incrementally update indexes directly from Claude Code.
- Added `leann_status` MCP tool: inspect index details (backend, embedding model, chunk/file count, size).
- `leann_search` now uses `--json` output with file paths always included, formatted as markdown code blocks.
- Fixed `float32` JSON serialization bug in `leann search --json`.
- Cleaned up MCP tool descriptions (concise, no emoji).

## 2026-03-05: Documentation — roadmap, vision, and dev guidelines

- Rewrote `docs/roadmap.md` with current P0/P1 priorities from GitHub issue #237.
- Added `docs/ultimate_goal.md` — long-term vision (personal data platform, best code retrieval MCP, multimodal, local-first).
- Added self-contained documentation principle and dev doc maintenance rules to `CLAUDE.md`.

## 2026-06-02: GPU FlashLib IVF backend (`flashlib_ivf`)

- Add `leann-backend-flashlib-ivf`, a GPU IVF-Flat (inverted file) approximate-NN
  backend built on FlashLib (`flash_ivf_flat`, Triton/CuteDSL) — the GPU counterpart
  of the FAISS `ivf` backend. Registered as backend name `flashlib_ivf`; install via
  `uv sync --extra flashlib-ivf` or `pip install leann-backend-flashlib-ivf`. Shares
  the `nlist`/`nprobe` recall knobs with the `ivf` backend, so the two are drop-in
  comparable. Requires a CUDA GPU at build (k-means) and search.
- Add `benchmarks/flashlib_ivf_vs_faiss_ivf.py`: head-to-head `flashlib_ivf` (GPU) vs
  `ivf` (FAISS, CPU) at matched `nlist` across an `nprobe` sweep (build time,
  single-query latency, batched throughput, recall@k vs exact ground truth). On an
  NVIDIA H200 at 1M x 768 vectors (nlist=4096, 8 CPU threads): ~13x faster build and,
  at nprobe=32, ~6.5x lower single-query latency / ~75x higher batched throughput at
  comparable recall (GPU latency stays ~flat while CPU grows linearly with nprobe).
- Docs: `docs/flashlib_backend_guide.md` gains a `flashlib_ivf` section.

## 2026-07-27: Bounded `leann list` index discovery

- Bound App-format metadata discovery to a configurable directory depth; `leann list` defaults to depth 5 and accepts `--max-depth` for deeper projects.
- Prune common dependency, virtual-environment, cache, and system directories during discovery.
- Preserve discovery of registered CLI-format and App-format projects without recursively scanning an entire home directory.

## 2026-07-29: CI build matrix — Python 3.14 wheels for macOS/Linux (#385)

- `leann-backend-hnsw` and `leann-backend-diskann` 0.3.7 only shipped a `cp314`
  wheel for `win_amd64` (the CI build matrix had a `windows-2022` / Python 3.14
  row but no equivalent for macOS or Linux), and neither package set
  `requires-python`. Resolvers on Python 3.14 (macOS/Linux) selected the
  release anyway and then failed with a confusing "only has wheels for
  `win_amd64`" error instead of a clear incompatibility message.
- Extended `.github/workflows/build-reusable.yml`'s build matrix with Python
  3.14 rows for `ubuntu-22.04`, `ubuntu-22.04-arm`, `macos-14`, `macos-15`, and
  `macos-26`, matching the Windows coverage. `macos-15-intel` (x86_64) is
  intentionally excluded, consistent with its existing Python 3.13 exclusion —
  PyTorch (a `leann-core` dependency) publishes no `macosx x86_64` wheel for
  either version.
- A `requires-python` upper bound was considered instead but rejected: the
  bound applies uniformly across platforms, so it would have also blocked the
  already-working Windows `cp314` installs. Completing the wheel matrix is the
  correct fix; the next 0.3.8 patch release will carry full `cp314` coverage
  once CI confirms the new platforms build cleanly.

## 2026-09-06: Restore CLI parsing after duplicate index command registration

- Remove the seven duplicate index command parsers and handlers introduced by #269 after #285 had already supplied them. This fixes `argparse.ArgumentError: conflicting subparser: index-browser`, which prevented every CLI command, including help, from running.
- Retain the #285 command interface (`--max-count`, `--index-name`, `--no-recompute`), application readers, and shared builder; native indexing and standalone wheel reader packaging are unchanged.

## 2026-09-06: Preserve mixed-script words in BM25 search

- Keep Latin words and numbers separate from adjacent Chinese, Japanese, and Korean n-grams when building SQLite FTS5 indexes and parsing keyword queries. For example, `Python数据库SQL` can be retrieved by `Python`, `SQL`, or a mixed-script query.
- Existing BM25 artifacts remain readable. Rebuild indexes containing mixed-script text to regenerate tokens that were previously joined at script boundaries.
