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

## 2026-10-12: Parallel Search MCP web reader (#433)

- Added `apps/parallel_data`: anonymous Parallel Search MCP search/fetch (no API key) exported as Markdown named `<url-slug>-<hash>.md` for `leann build`.
- Added `apps/parallel_rag.py`: one-command web search → chunk → index → ask, keeping each chunk's title and source URL.

## 2026-10-12: Fit chunk sizes to the embedding model's sequence limit

- `leann build` now caps `--doc-chunk-size` / `--code-chunk-size` at 70% of the embedding model's sequence limit. Previously the macOS default model `all-MiniLM-L6-v2` (reads 256 tokens) got 512-token code chunks, so ~43% of indexed code text was never embedded. On `leann-core` sources, code retrieval MRR@10 rose from 0.30 to 0.54 (18 queries, real `leann build` indexes).
- `get_model_token_limit()` reads `max_seq_length` from Hugging Face sentence-transformers repos instead of assuming 2048 tokens for them.
- `create_text_chunks(max_tokens_per_chunk=...)` now only shrinks `chunk_size`; it previously replaced it, growing 256-token chunks to ~1700 for unknown models.
- Existing indexes are unchanged; rebuild with `--force` to re-chunk with the new sizes.

## 2026-10-22: Harden passage offset loading

- Restricted passage offset-map unpickling to the primitive dictionary shape LEANN writes.
- Rejected crafted `.passages.idx` files before a searcher or incremental update can continue.
- Added regression coverage for the searcher entry point, primitive compatibility, and malformed input.
- Restricted sync snapshots and embeddings pickles to their expected object graphs.
- Loaded FlashLib IVF state dictionaries with PyTorch's safe weights-only loader.

## 2026-10-23: Refuse to load a custom encoder as a vanilla BertModel

- `jinaai/jina-embeddings-v2-base-code` (and any encoder that declares its
  architecture through `config.auto_map` while keeping `model_type="bert"`) was
  being loaded without `trust_remote_code`. Transformers silently built a plain
  `BertModel` instead, discarded every weight it could not map, randomly
  initialised ~70 tensors (all 12 layers' feed-forward and output LayerNorm
  weights, plus position embeddings), and returned one constant vector for every
  input. Nothing raised: the index built, search returned results, and the
  ranking was noise. Measured on an 8-chunk retrieval probe, pairwise chunk
  cosine similarity was 1.0000 (mean and max) and recall@1 was 0.000; with the
  model's own code loaded, cosine fell to 0.064 mean / 0.239 max and recall@1
  reached 1.000.
- `embedding_compute.py` now checks `config.auto_map` before loading and raises
  with the model name and the fix, rather than producing a degenerate index.
  Opt in with `LEANN_TRUST_REMOTE_CODE=1`; it stays off by default because
  `trust_remote_code` executes arbitrary Python from the model repository, which
  matches the existing default in `chat.py`.
- The flag is now threaded through all five `SentenceTransformer(...)` call
  sites in the loader cascade and included in the model cache key.
- `benchmarks/contextbench/scripts/batch_run_random.py` executes tasks again.
  The block had been commented out, and the commented code unpacked four values
  from `run_single_task`, which has since returned five
  (`patch, latency_seconds, agent_seconds, traj_data, usage`) — so uncommenting
  it would have failed on the unpack. The restored block matches
  `batch_run_selected.py`, which was already on the current contract, and
  records `agent_seconds` and `latency_seconds` in each result row.

## 2026-11-01: OCR for scanned pages and images, with an olmOCR-Bench benchmark

- `apps/document_rag.py` now finds image-only pages inside PDFs that also have a text
  layer, and OCRs just those pages. Before, those PDFs skipped OCR and lost the pages.
- Added `--ocr-tier`, and OCR for `.png`, `.jpg`, and `.jpeg` files.
- Files with no text are now skipped and named, instead of being indexed as empty chunks.
- Added `benchmarks/ocr_benchmark/`, which scores OCR with olmOCR's own tests, the way
  olmOCR's `benchmark.py` does. On a sample of 50 PDFs, MinerU 4.0.12 scored 85.5% with
  the `advanced` tier, 82.5% with `standard`, 73.2% with `basic`, and 64.8% with `flash`.

## 2026-11-03: Embedding daemons no longer hold the caller's stdout/stderr

- A daemon started through `LeannSearcher`, `LeannChat` or `leann daemon start` inherited the
  stdio of the process that started it. Anything waiting for EOF on those pipes, such as
  `python app.py | tee out.log` or `subprocess.run(capture_output=True)`, blocked until the idle
  TTL expired (900 s by default) even though the program had already finished.
- Daemons now run in their own session with stdin closed and write to
  `~/.leann/servers/<registry key>.log`, truncated at each launch. `leann daemon status` prints the
  path, and the error raised when a daemon fails to start includes the end of that log.
- Servers started with `use_daemon=False` still share the caller's stdio.
- `_pid_is_alive` uses `psutil.pid_exists`. `os.kill(pid, 0)` is not an existence check on Windows,
  where signal 0 is `CTRL_C_EVENT`.

## 2026-11-10: Stop forced OCR from silently indexing nothing

- `--use-ocr` without MinerU installed produced an index with no PDF content
  and printed nothing. Forced OCR skips text extraction entirely, so
  `extract_pdf_with_ocr_fallback` returned `""` for every file, `OcrPDFReader`
  turned that into zero documents, and a PDF with a perfectly good text layer
  contributed nothing. Verified through `SimpleDirectoryReader` rather than
  the extractor alone: `--use-ocr` on a text-layer PDF yielded 0 documents.
  `OcrPDFReader.__init__` now refuses `use_ocr=True` when MinerU is missing,
  which covers both the CLI flag and direct construction with one check.
  Auto-detection stays opportunistic and still tolerates an empty OCR result.
- Auto-detection discarded text it already had. A page whose text layer is
  usable but under the 100-character scanned-PDF threshold had `text`
  overwritten with the OCR result unconditionally, so a failed or unavailable
  OCR pass turned those characters into no document at all. The OCR result is
  now only taken when it is non-empty: a 33-character text layer went from 0
  documents to 1.
- Both are from the OCR work merged in #190, in the half #440 did not cover.

## 2026-11-17: Cosine recompute, HNSW build options, and four silent-corruption fixes

Triage pass over the open PR queue. Entries for merges that did not carry their own.

- **Cosine HNSW indexes ranked recompute results by raw inner product (#449).** `cosine`
  is implemented as `faiss.METRIC_INNER_PRODUCT` (`hnsw_backend.py:28`), so correctness
  depends on every vector being unit length. `HNSWBuilder` normalized at build time and
  `HNSWSearcher` normalized the query, but the embedding server scored recomputed passages
  with their raw embeddings, so long vectors beat close ones. Recall@10 on 3,000 BEIR
  Natural Questions passages with `facebook/contriever` (whose passage norms span ~1.1-2.2)
  went from 0.480 to 0.990; stored-vector search was unaffected at 0.987. Only models that
  do not L2-normalize their own output were affected; the default `mips` metric was not.
- **`--graph-degree`, `--complexity` and `--num-threads` did nothing on the HNSW backend
  (#450).** `leann build` passed the backend-neutral names but `HNSWBuilder` read only
  `M`/`efConstruction`, so `--graph-degree 8` still built an M=32 graph. Two behavior
  changes follow from the flags now being honored: the bundled apps pass
  `--build-complexity 64` (`apps/base_rag_example.py:267`), so app-built HNSW indexes now
  use efConstruction=64 instead of the 200 they silently got, and `leann watch` replay of a
  pre-existing index rebuilds at that index's saved defaults. Graph construction also now
  uses all cores on Linux instead of inheriting the `OMP_NUM_THREADS=1` that `__init__.py`
  sets for the recompute/ZMQ hang in #208 — construction makes no ZMQ calls, and the thread
  count is restored in a `finally` so later searches keep the guard. 100k x 768-d build on
  an i9-13900K: 125.9s to 21.4s.
- **AST chunking segfaulted on a fresh install (#453).** py-tree-sitter 0.26.0 returns
  borrowed references from `Point.row`/`Point.column`, corrupting memory once astchunk walks
  a large enough file. `tree-sitter` is now `>=0.20.0,!=0.26.0` rather than `<0.26`, so the
  fixed release is allowed in; 0.26.1 has since shipped. `uv sync` users were never exposed
  because `uv.lock` pins 0.25.1; the pin covers `uv pip install -e "."` as in the AST
  chunking guide, and the next `uv lock --upgrade`.
- **A literal `<|endoftext|>` in a document crashed chunking (#461).** tiktoken's `encode()`
  defaults to `disallowed_special="all"` and raises `ValueError` on such text, and both call
  sites in `chunking_utils.py` caught only `ImportError`, so the error propagated out of
  `estimate_token_count` and `validate_chunk_token_limits`. Any corpus quoting a chat
  template or tokenizer documentation failed to index. Both paths now pass
  `disallowed_special=()` so the spelling is counted as ordinary text.
- **`WeChatHistoryReader` truncated any message containing a colon (#465, fixes #464).** A
  broad `^[^:]+:\s*` strip, applied after the legitimate `wxid_<id>:` sender strip, treated
  any leading `word:` as a sender prefix: "Reminder: buy milk tomorrow" indexed as "buy milk
  tomorrow" and "10:30 meeting at noon" as "30 meeting at noon". Worse, "Note:" became the
  empty string, so `_is_text_message` returned `False` and the message was dropped from the
  index entirely. The broad strip was also redundant — the specific `wxid_` strip already
  handles the one format it was written for.
- **The Anthropic factory default was a retired model (#454).** `get_llm({"type":
  "anthropic"})` with no `model` built `claude-3-5-sonnet-20241022`, so the first question
  with the Anthropic provider and no explicit model returned a 404, while `AnthropicChat`'s
  own signature default had already moved to `claude-haiku-4-5`. The factory's duplicate
  default was removed rather than synced, leaving the class signature as the only place the
  default is named. Note for a follow-up: `claude-haiku-4-5` is now previous-generation —
  `claude-haiku-5-5` has a 1M context against 200K and costs $0.10/$0.50 per MTok against
  $1.00/$5.00. The bump is a pure string change here, because the Anthropic request is built
  with only `model`, `messages`, `max_tokens` and an optional `temperature`.
- **Opt-in Outcrawl web document reader (#458).** Search-and-scrape reader under
  `apps/outcrawl_data/` plus `apps/outcrawl_rag.py` and `docs/outcrawl.md`, following the
  layout of the Parallel MCP reader (#433). Self-contained: no core, dependency or existing
  app changes. A page that fails to scrape keeps its search snippet rather than disappearing.
- **Test and docs hygiene.** `tests/test_cli_prompt_template.py` now runs each test in its
  own temporary working directory (#451) — `LeannCLI.__init__` does
  `Path.cwd() / ".leann" / "indexes"` and `mkdir`s it, so the suite left
  `.leann/indexes/test-index` in the checkout and every later local run failed five tests.
  `tests/test_document_rag.py` reads subprocess output as UTF-8 (#460), which on Windows
  cp1252 had made `result.stdout + result.stderr` raise. FAQ §6 documents the nltk failure
  in group-writable environments (#452): llama-index keeps its nltk data inside the
  environment at `site-packages/llama_index/core/_static/nltk_cache`, which nltk's hardened
  loader refuses under `umask 002`; `UV_LINK_MODE=copy` does not help because the copy is
  still group-writable, so point `NLTK_DATA` at a private directory.

### Known CI issue

`Build macos-14 Python 3.10`-`3.14` is cancelled on every pull request, because GitHub has
retired the `macos-14` runner image. The jobs are cancelled rather than failed — three of
them one second after starting — so `gh pr checks` reports them as failures on PRs whose
code is fine. macos-14 last passed on 3 November (#460). Resolved below on 2026-11-19;
moving the target to another runner turned out not to be possible, so the runner was
dropped and the wheel floor moved with it.

## 2026-11-19: Prebuilt macOS wheels now require macOS 15 (breaking)

- **Releases after `0.3.9` do not ship a `macosx_14_0_arm64` wheel.** The prebuilt Apple
  Silicon floor is macOS 15. On macOS 14, pin `leann==0.3.9` — which does ship that wheel for
  Python 3.10 through 3.14 — or build from source.
- Why: GitHub retired the `macos-14` runner image, so those five build jobs stopped being
  scheduled. They were *cancelled* rather than failed, which `gh pr checks` reports as a
  failure, so every pull request showed five red jobs whose code was fine.
- Why the target could not simply move to another runner: a 14.0 deployment target is not
  reachable from a `macos-15` runner, because the Homebrew bottles installed there carry a
  15.0 minimum themselves. `delocate` refuses the wheel rather than mislabel it:

  ```
  DelocationError: Library dependencies do not satisfy target MacOS version 14.0:
    libomp.dylib has a minimum target of 15.0
    libzmq.5.dylib has a minimum target of 15.0
    libsodium.26.dylib has a minimum target of 15.0
  ```

  This is the same constraint the matrix already recorded for `macos-15-intel` ("system
  libraries require it"). Preserving the 14.0 floor would mean building those three
  dependencies from source on every macOS job.
- Coverage is otherwise unchanged: `macos-15` and `macos-26` for Apple Silicon,
  `macos-15-intel` for x86_64. The build matrix goes from 32 jobs to 27.

## 2026-11-20: Deleting files no longer leaves them searchable in HNSW indexes

- HNSW cannot delete vectors in place, so `leann build` falls back to a full rebuild when
  files are modified or removed. When the only changes were deletions, the incremental path
  loaded the changed files, got no text back, and returned with "No documents found" before
  reaching that fallback. The sync snapshot was not committed either, so every later run
  printed the same message, and the deleted files' chunks stayed in the index until
  another file was added or modified, or `--force` was used. The early return now applies
  only when nothing was modified or removed; IVF still removes the chunks in place.

## 2026-11-24: Grep search finds the index's own passages

- `search(..., use_grep=True)` looked for a hardcoded `documents.leann.passages.jsonl`, so
  it raised `FileNotFoundError: No .jsonl passages file found for grep search` for every
  index named differently — including the repo's own `examples/grep_search_example.py`,
  which opens `my-documents.leann`.
- Grep search now reads the passage shards that `PassageManager` resolves from the index's
  own `meta.json` `passage_sources`, and searches every declared shard instead of a single
  hardcoded file, so multi-shard indexes no longer drop matches.
- Nothing that worked before stops working: an index named `documents` resolves the same
  file as before, and a genuinely missing passages file still raises the same error.

## 2026-11-26: Deleting every indexed file now removes the index

- The 2026-11-20 fix covered the delta path, but the same "No documents found" guard still
  existed in the full rebuild that HNSW falls back to when files are only deleted. When every
  indexed file had been removed, the rebuild loaded nothing, printed "No documents found", and
  returned with the old index intact and the sync snapshot uncommitted, so `leann search` kept
  returning the deleted passages and every later build repeated the message.
- The rebuild now asks the synchronizers what they currently track and reads the corpus
  definition the index was built from (the `--file-types` and `--include-hidden` scope recorded
  in the index's sync roots). If the index exists, the build compared a previous snapshot against
  a corpus it actually observed, and nothing is tracked anymore under that same corpus
  definition, the whole corpus was deleted: the index directory is removed (the same one
  `leann remove` deletes), the empty state is committed so the next build starts clean, and the
  command says so explicitly.
- An empty tracked set is only treated as a deletion when this build runs with the same corpus
  definition the index was built from. A build that never observed the corpus — a `--docs` path
  that does not exist, or a directory the index has no snapshot for — never deletes anything; a
  `--file-types` or `--include-hidden` value that matches nothing empties the tracked set while
  every indexed file is still on disk and keeps the index, and an index with no recorded corpus
  definition is kept too. All of these report "No documents found". Files that still exist but
  yield no text keep the previous behaviour — "No documents found" with the index untouched,
  since the index is still valid. A file that merely cannot be read (permissions) is still
  present: what the synchronizer tracks is decided without reading file contents, so an
  unreadable file stays a tracked path and never triggers the removal — only a tracked set
  that is genuinely empty does.
