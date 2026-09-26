# LEANN Memory Search for OpenClaw

97% storage-compressed semantic memory search with free local embeddings.

## Why

Every OpenClaw memory solution stores full embedding vectors. On a 256 GB Mac
Mini, heavy users accumulate 500 MB - 6 GB+ of embedding indexes over time.
LEANN compresses this to ~2% of the original size through graph-based selective
recomputation, while using high-quality local embeddings (zero API cost).

| Feature | Default memory | LEANN |
|---|---|---|
| Storage (50K chunks) | ~75 MB | **~2 MB** |
| Embedding cost | Remote API ($) | **$0 (local)** |
| Scale | ~100K chunks | **60M+ passages** |

## Install

```bash
# Install LEANN
pip install leann-core

# Install the skill
clawhub install leann-team/leann-memory

# Or manually: copy this directory to ~/.openclaw/workspace/skills/leann-memory/
```

## Quick Start

```bash
# Build index on your memory files
leann build openclaw-memory \
  --docs ~/.openclaw/workspace/MEMORY.md ~/.openclaw/workspace/memory/ \
  --embedding-model all-MiniLM-L6-v2

# Test a search
leann search openclaw-memory "what did we decide about the database" --json
```

Then ask your OpenClaw agent: "search my memories for database decisions"

## Auto-Sync

Keep the index updated as new memories are added:

```bash
# One-shot check and rebuild
leann watch openclaw-memory --once

# Continuous monitoring (runs in background)
leann watch openclaw-memory --interval 30
```

## How It Works

LEANN stores a pruned neighbor graph instead of full embedding vectors. During
search, embeddings are recomputed on-demand via a local daemon. OpenClaw's async
"sleep time compute" model makes recomputation latency invisible to users.

## Links

- [LEANN Repository](https://github.com/StarTrail-org/LEANN)
- [Integration Plan](../../docs/openclaw-integration-plan.md)
