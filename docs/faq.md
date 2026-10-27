# FAQ

## 1. My building time seems long

You can speed up the process by using a lightweight embedding model. Add this to your arguments:

```bash
--embedding-model sentence-transformers/all-MiniLM-L6-v2
```
**Model sizes:** `all-MiniLM-L6-v2` (30M parameters), `facebook/contriever` (~100M parameters), `Qwen3-0.6B` (600M parameters)

## 2. When should I use prompt templates?

**Use prompt templates ONLY with task-specific embedding models** like Google's EmbeddingGemma. These models are specially trained to use different prompts for documents vs queries.

**DO NOT use with regular models** like `nomic-embed-text`, `text-embedding-3-small`, or `bge-base-en-v1.5` - adding prompts to these models will corrupt the embeddings.

**Example usage with EmbeddingGemma:**
```bash
# Build with document prompt
leann build my-docs --embedding-prompt-template "title: none | text: "

# Search with query prompt
leann search my-docs --query "your question" --embedding-prompt-template "task: search result | query: "
```

See the [Configuration Guide: Task-Specific Prompt Templates](configuration-guide.md#task-specific-prompt-templates) for detailed usage.

## 3. Why is LM Studio loading multiple copies of my model?

This was fixed in recent versions. LEANN now properly unloads models after querying metadata, respecting your LM Studio JIT auto-evict settings.

**If you still see duplicates:**
- Update to the latest LEANN version
- Restart LM Studio to clear loaded models
- Check that you have JIT auto-evict enabled in LM Studio settings

**How it works now:**
1. LEANN loads model temporarily to get context length
2. Immediately unloads after query
3. LM Studio JIT loads model on-demand for actual embeddings
4. Auto-evicts per your settings

## 4. Do I need Node.js and @lmstudio/sdk?

**No, it's completely optional.** LEANN works perfectly fine without them using a built-in token limit registry.

**Benefits if you install it:**
- Automatic context length detection for LM Studio models
- No manual registry maintenance
- Always gets accurate token limits from the model itself

**To install (optional):**
```bash
npm install -g @lmstudio/sdk
```

See [Configuration Guide: LM Studio Auto-Detection](configuration-guide.md#lm-studio-auto-detection-optional) for details.

## 5. `leann build` fails with `Security Violation [pathsec.open]: refusing multiply-linked file` (Linux)

uv installs packages by **hardlinking** files from its cache into the environment by default. Recent nltk releases ship a hardened file loader (`nltk/pathsec.py`) that refuses to open multiply-linked files (`st_nlink > 1`, CWE-59 guard) - and llama-index, which LEANN uses for document parsing, loads its bundled nltk stopwords through exactly that path. The result on an uv-installed LEANN:

```text
PermissionError: Security Violation [pathsec.open]: refusing multiply-linked file
'.../site-packages/llama_index/core/_static/nltk_cache/corpora/stopwords/english' (st_nlink=2);
a hardlink can point at an outside-root inode (CWE-59)
```

**Fix:** reinstall with copy mode so no hardlinks are created:

```bash
UV_LINK_MODE=copy uv pip install --reinstall leann
# or, for the global CLI/MCP install:
UV_LINK_MODE=copy uv tool install --reinstall leann-core --with leann
```

`--reinstall` matters: without it uv leaves the already-hardlinked packages in place (satisfied requirements are skipped, and `uv tool install` refuses to overwrite an existing tool), so the error persists. pip installs are unaffected (pip copies by default).

## 6. `leann build` fails with `Resource 'stopwords' not found` after `NLTK will not read data from ...` warnings (Linux)

The same hardened nltk loader also refuses data directories that other accounts could write to. llama-index keeps its nltk data inside the environment (`site-packages/llama_index/core/_static/nltk_cache`), so an environment created under a group-writable umask (`umask 002`, the default on many Linux distributions) or inside a group-writable directory trips it when documents are chunked:

```text
RuntimeWarning: NLTK will not read data from '.../site-packages/llama_index/core/_static/nltk_cache':
it is writable by, or owned by, another account, so a planted or rewritten file would be loaded as
trusted data. Make it private to its owner (chmod go-w).
[nltk_data] Refusing to install into '.../site-packages/llama_index/core/_static/nltk_cache': ...
LookupError:
  Resource 'stopwords' not found.
```

**Fix:** point llama-index at a private nltk data directory. It downloads `stopwords` and `punkt_tab` there on the first build (network needed once):

```bash
mkdir -m 700 -p ~/.cache/nltk_data
export NLTK_DATA=~/.cache/nltk_data   # add to your shell profile
```

Unlike the hardlink case above, `UV_LINK_MODE=copy` does not help here: the copied directory is still group-writable.
