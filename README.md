<p align="center">
  <img src="assets/logo-text.png" alt="LEANN Logo" width="400">
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/15049" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/15049" alt="StarTrail-org/LEANN | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg" alt="Python Versions">
  <img src="https://github.com/StarTrail-org/LEANN/actions/workflows/build-and-publish.yml/badge.svg" alt="CI Status">
  <img src="https://img.shields.io/badge/Platform-Ubuntu%20%26%20Arch%20%26%20WSL%20%7C%20macOS%20(ARM64%2FIntel)%20%7C%20Windows-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/MCP-Native%20Integration-blue" alt="MCP Integration">
  <a href="https://join.slack.com/t/leann-e2u9779/shared_invite/zt-3ol2ww9ic-Eg_kB8omwe6xmYVd0epr4Q">
    <img src="https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&logoColor=white" alt="Join Slack">
  </a>

</p>

<div align="center">
  <a href="https://forms.gle/rDbZf864gMNxhpTq8">
    <img src="https://img.shields.io/badge/📣_Community_Survey-Help_Shape_v0.4-007ec6?style=for-the-badge&logo=google-forms&logoColor=white" alt="Take Survey">
  </a>
  <p>
    We track <b>zero telemetry</b>. This survey is the ONLY way to tell us if you want <br>
    <b>GPU Acceleration</b> or <b>More Integrations</b> next.<br>
    👉 <a href="https://forms.gle/rDbZf864gMNxhpTq8"><b>Click here to cast your vote (2 mins)</b></a>
  </p>
</div>

<div align="center">
  <h3>💬 Join our Slack community!</h3>
  <p>
    We'd love for you to be part of the LEANN community!<br>
    👉 <a href="https://join.slack.com/t/leann-e2u9779/shared_invite/zt-3ol2ww9ic-Eg_kB8omwe6xmYVd0epr4Q"><b>Join LEANN Slack</b></a><br>
    If the invite link has expired or you have trouble joining, please <a href="https://github.com/StarTrail-org/LEANN/issues">open an issue</a> and we'll help you get in!
  </p>
</div>

<h2 align="center" tabindex="-1" class="heading-element" dir="auto">
    The smallest vector index in the world. RAG Everything with LEANN!
</h2>

LEANN is an innovative vector database that democratizes personal AI. Transform your laptop into a powerful RAG system that can index and search through millions of documents while using **97% less storage** than traditional solutions **without accuracy loss**.


LEANN achieves this through *graph-based selective recomputation* with *high-degree preserving pruning*, computing embeddings on-demand instead of storing them all. [Illustration Fig →](#️-architecture--how-it-works) | [Paper →](https://arxiv.org/abs/2506.08276)

**Ready to RAG Everything?** Transform your laptop into a personal AI assistant that can semantic search your **[file system](#-personal-data-manager-process-any-documents-pdf-txt-md)**, **[emails](#-your-personal-email-secretary-rag-on-apple-mail)**, **[browser history](#-time-machine-for-the-web-rag-your-entire-browser-history)**, **[chat history](#-wechat-detective-unlock-your-golden-memories)** ([WeChat](#-wechat-detective-unlock-your-golden-memories), [iMessage](#-imessage-history-your-personal-conversation-archive)), **[agent memory](#-chatgpt-chat-history-your-personal-ai-conversation-archive)** ([ChatGPT](#-chatgpt-chat-history-your-personal-ai-conversation-archive), [Claude](#-claude-chat-history-your-personal-ai-conversation-archive)), **[live data](#mcp-integration-rag-on-live-data-from-any-platform)** ([Slack](#slack-messages-search-your-team-conversations), [Twitter](#-twitter-bookmarks-your-personal-tweet-library)), **[codebase](#-claude-code-integration-transform-your-development-workflow)**\*, or external knowledge bases (i.e., 60M documents) - all on your laptop, with zero cloud costs and complete privacy.


\* Claude Code only supports basic `grep`-style keyword search. **LEANN** is a drop-in **semantic search MCP service fully compatible with Claude Code**, unlocking intelligent retrieval without changing your workflow. 🔥 Check out [the easy setup →](packages/leann-mcp/README.md)



## Why LEANN?

<p align="center">
  <img src="assets/effects.png" alt="LEANN vs Traditional Vector DB Storage Comparison" width="70%">
</p>

> **The numbers speak for themselves:** Index 60 million text chunks in just 6GB instead of 201GB. From emails to browser history, everything fits on your laptop. [See detailed benchmarks for different applications below ↓](#-storage-comparison)


🔒 **Privacy:** Your data never leaves your laptop. No OpenAI, no cloud, no "terms of service".

🪶 **Lightweight:** Graph-based recomputation eliminates heavy embedding storage, while smart graph pruning and CSR format minimize graph storage overhead. Always less storage, less memory usage!

📦 **Portable:** Transfer your entire knowledge base between devices (even with others) with minimal cost - your personal AI memory travels with you.

📈 **Scalability:** Handle messy personal data that would crash traditional vector DBs, easily managing your growing personalized data and agent generated memory!

✨ **No Accuracy Loss:** Maintain the same search quality as heavyweight solutions while using 97% less storage.

## Making Your Coding Agent Smarter

We plugged LEANN into Claude Code and compared it with BM25 on 30 SWE-Bench Pro tasks from ContextBench, keeping the model, agent, tools, and 8,192-token retrieval budget fixed.

<p align="center">
  <img src="assets/contextbench_leann_vs_bm25.png" alt="LEANN vs BM25 on ContextBench relevant-code recall, exploration coverage, and agent token usage" width="100%">
</p>

With LEANN, the agent achieved **2.1× the initial relevant-code recall** (24.2% vs. 11.4%), reached **12.6 percentage points higher relevant-code coverage after exploration** (38.4% vs. 25.8%), and used **8.4% fewer tokens** (3.22M vs. 3.51M).

LEANN surfaces relevant code earlier, so coding agents can spend more of their context window reasoning about the right files instead of repeatedly scanning the repository. **[Reproduce the benchmark →](benchmarks/contextbench/README.md)**

*Note: Results are macro-averaged annotated gold-line metrics over 30 SWE-Bench Pro tasks. Better context access does not guarantee issue resolution.*

## Installation

### 📦 Prerequisites: Install uv

[Install uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) first if you don't have it. Typically, you can install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 🚀 Quick Install

Clone the repository to access all examples and try amazing applications,

```bash
git clone https://github.com/StarTrail-org/LEANN.git leann
cd leann
```

and install LEANN from [PyPI](https://pypi.org/project/leann/) to run them immediately:

```bash
uv venv
source .venv/bin/activate
uv pip install leann

# CPU-only (Linux): use the `cpu` extra (e.g. `leann[cpu]`)
```

> **Linux note:** if a later `leann build` fails with `Security Violation [pathsec.open]: refusing multiply-linked file`, reinstall with `UV_LINK_MODE=copy uv pip install --reinstall ...` (or `UV_LINK_MODE=copy uv tool install --reinstall ...`) - uv's default hardlink installs trip nltk's hardened file loader bundled with llama-index. Details in the [FAQ](docs/faq.md).

<!--
> Low-resource? See "Low-resource setups" in the [Configuration Guide](docs/configuration-guide.md#low-resource-setups). -->

<details>
<summary>
<strong>🔧 Build from Source (Recommended for development)</strong>
</summary>



```bash
git clone https://github.com/StarTrail-org/LEANN.git leann
cd leann
git submodule update --init --recursive
```

**macOS:**

Note: DiskANN requires MacOS 13.3 or later.

```bash
brew install libomp boost protobuf zeromq pkgconf
uv sync --extra diskann
```

**Linux (Ubuntu/Debian):**

Note: On Ubuntu 20.04, you may need to build a newer Abseil and pin Protobuf (e.g., v3.20.x) for building DiskANN. See [Issue #30](https://github.com/StarTrail-org/LEANN/issues/30) for a step-by-step note.

You can manually install [Intel oneAPI MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) instead of `libmkl-full-dev` for DiskANN. You can also use `libopenblas-dev` for building HNSW only, by removing `--extra diskann` in the command below.

```bash
sudo apt-get update && sudo apt-get install -y \
  libomp-dev libboost-all-dev protobuf-compiler libzmq3-dev \
  pkg-config libabsl-dev libaio-dev libprotobuf-dev \
  libmkl-full-dev

uv sync --extra diskann
```

**Linux (Arch Linux):**

```bash
sudo pacman -Syu && sudo pacman -S --needed base-devel cmake pkgconf git gcc \
  boost boost-libs protobuf abseil-cpp libaio zeromq

# For MKL in DiskANN
sudo pacman -S --needed base-devel git
git clone https://aur.archlinux.org/paru-bin.git
cd paru-bin && makepkg -si
paru -S intel-oneapi-mkl intel-oneapi-compiler
source /opt/intel/oneapi/setvars.sh

uv sync --extra diskann
```

**Linux (RHEL / CentOS Stream / Oracle / Rocky / AlmaLinux):**

See [Issue #50](https://github.com/StarTrail-org/LEANN/issues/50) for more details.

```bash
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y libomp-devel boost-devel protobuf-compiler protobuf-devel \
  abseil-cpp-devel libaio-devel zeromq-devel pkgconf-pkg-config

# For MKL in DiskANN
sudo dnf install -y intel-oneapi-mkl intel-oneapi-mkl-devel \
  intel-oneapi-openmp || sudo dnf install -y intel-oneapi-compiler
source /opt/intel/oneapi/setvars.sh

uv sync --extra diskann
```

**Windows:**

Requires [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) with the **C++ desktop development** workload, and [vcpkg](https://github.com/microsoft/vcpkg).

```powershell
# Install toolchain (if not already present)
choco install cmake swig pkgconfiglite nuget.commandline -y

# Install C++ dependencies via vcpkg
vcpkg install zeromq:x64-windows openblas:x64-windows lapack:x64-windows `
  boost-program-options:x64-windows protobuf:x64-windows

# Set environment variables (adjust VCPKG_ROOT to your vcpkg path)
$env:CMAKE_PREFIX_PATH = "$env:VCPKG_ROOT\installed\x64-windows"
$env:PKG_CONFIG_PATH = "$env:VCPKG_ROOT\installed\x64-windows\lib\pkgconfig"
$env:PKG_CONFIG_EXECUTABLE = "C:\ProgramData\chocolatey\bin\pkg-config.exe"
$env:OPENBLAS_LIB = "$env:VCPKG_ROOT\installed\x64-windows\lib\openblas.lib"
$env:PATH += ";$env:VCPKG_ROOT\installed\x64-windows\bin"
$env:PATH += ";$env:VCPKG_ROOT\installed\x64-windows\tools\protobuf"

uv sync --extra diskann
```

</details>


## Quick Start

Our declarative API makes RAG as easy as writing a config file.

Check out [demo.ipynb](demo.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StarTrail-org/LEANN/blob/main/demo.ipynb)

```python
from leann import LeannBuilder, LeannSearcher, LeannChat
from pathlib import Path
INDEX_PATH = str(Path("./").resolve() / "demo.leann")

# Build an index
builder = LeannBuilder(backend_name="hnsw")
builder.add_text("LEANN saves 97% storage compared to traditional vector databases.")
builder.add_text("Tung Tung Tung Sahur called—they need their banana‑crocodile hybrid back")
builder.build_index(INDEX_PATH)

# Search
searcher = LeannSearcher(INDEX_PATH)
results = searcher.search("fantastical AI-generated creatures", top_k=1)

# Chat with your data
chat = LeannChat(INDEX_PATH, llm_config={"type": "hf", "model": "Qwen/Qwen3-0.6B"})
response = chat.ask("How much storage does LEANN save?", top_k=1)
```

## RAG on Everything!

LEANN supports RAG on various data sources including documents (`.pdf`, `.txt`, `.md`), Apple Mail, Google Search History, WeChat, ChatGPT conversations, Claude conversations, iMessage conversations, and **live data from any platform through MCP (Model Context Protocol) servers** - including Slack, Twitter, and more.



### Generation Model Setup

#### LLM Backend

LEANN supports many LLM providers for text generation (HuggingFace, Ollama, Anthropic, and Any OpenAI compatible API).


<details>
<summary><strong>🔑 OpenAI API Setup (Default)</strong></summary>

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Make sure to use `--llm openai` flag when using the CLI.
You can also specify the model name with `--llm-model <model-name>` flag.

</details>

<details>
<summary><strong>🛠️ Supported LLM & Embedding Providers (via OpenAI Compatibility)</strong></summary>

Thanks to the widespread adoption of the OpenAI API format, LEANN is compatible out-of-the-box with a vast array of LLM and embedding providers. Simply set the `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variables to connect to your preferred service.

```sh
export OPENAI_API_KEY="xxx"
export OPENAI_BASE_URL="http://localhost:1234/v1" # base url of the provider
```

To use OpenAI compatible endpoint with the CLI interface:

If you are using it for text generation, make sure to use `--llm openai` flag and specify the model name with `--llm-model <model-name>` flag.

If you are using it for embedding, set the `--embedding-mode openai` flag and specify the model name with `--embedding-model <MODEL>`.

-----


Below is a list of base URLs for common providers to get you started.


### 🖥️ Local Inference Engines (Recommended for full privacy)

| Provider         | Sample Base URL             |
| ---------------- | --------------------------- |
| **Ollama** | `http://localhost:11434/v1` |
| **LM Studio** | `http://localhost:1234/v1`  |
| **vLLM** | `http://localhost:8000/v1`  |
| **llama.cpp** | `http://localhost:8080/v1`  |
| **SGLang** | `http://localhost:30000/v1` |
| **LiteLLM** | `http://localhost:4000`     |

-----

### ☁️ Cloud Providers

> **🚨 A Note on Privacy:** Before choosing a cloud provider, carefully review their privacy and data retention policies. Depending on their terms, your data may be used for their own purposes, including but not limited to human reviews and model training, which can lead to serious consequences if not handled properly.


| Provider         | Base URL                                                   |
| ---------------- | ---------------------------------------------------------- |
| **OpenAI** | `https://api.openai.com/v1`                                |
| **OpenRouter** | `https://openrouter.ai/api/v1`                             |
| **Gemini** | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| **x.AI (Grok)** | `https://api.x.ai/v1`                                      |
| **Groq AI** | `https://api.groq.com/openai/v1`                           |
| **DeepSeek** | `https://api.deepseek.com/v1`                              |
| **SiliconFlow** | `https://api.siliconflow.cn/v1`                            |
| **Zhipu (BigModel)** | `https://open.bigmodel.cn/api/paas/v4/`                |
| **Mistral AI** | `https://api.mistral.ai/v1`                                |
| **Anthropic** | `https://api.anthropic.com/v1`                             |
| **Jina AI** (Embeddings) | `https://api.jina.ai/v1`                         |

> **💡 Tip: Separate Embedding Provider**
>
> To use a different provider for embeddings (e.g., Jina AI) while using another for LLM, use `--embedding-api-base` and `--embedding-api-key`:
> ```bash
> leann build my-index --docs ./docs \
>   --embedding-mode openai \
>   --embedding-model jina-embeddings-v3 \
>   --embedding-api-base https://api.jina.ai/v1 \
>   --embedding-api-key $JINA_API_KEY
> ```

If your provider isn't on this list, don't worry! Check their documentation for an OpenAI-compatible endpoint—chances are, it's OpenAI Compatible too!

</details>

<details>
<summary><strong>🔧 Ollama Setup (Recommended for full privacy)</strong></summary>

**macOS:**

First, [download Ollama for macOS](https://ollama.com/download/mac).

```bash
# Pull a lightweight model (recommended for consumer hardware)
ollama pull llama3.2:1b
```

**Linux:**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service manually
ollama serve &

# Pull a lightweight model (recommended for consumer hardware)
ollama pull llama3.2:1b
```

</details>


## ⭐ Flexible Configuration

LEANN provides flexible parameters for embedding models, search strategies, and data processing to fit your specific needs.

📚 **Need configuration best practices?** Check our [Configuration Guide](docs/configuration-guide.md) for detailed optimization tips, model selection advice, and solutions to common issues like slow embeddings or poor search quality.

<details>
<summary><strong>📋 Click to expand: Common Parameters (Available in All Examples)</strong></summary>

All RAG examples share these common parameters. **Interactive mode** is available in all examples - simply run without `--query` to start a continuous Q&A session where you can ask multiple questions. Type 'quit' to exit.

```bash
# Environment Variables (GPU Device Selection)
LEANN_EMBEDDING_DEVICE       # GPU for embedding model (e.g., cuda:0, cuda:1, cpu)
LEANN_LLM_DEVICE             # GPU for HFChat LLM (e.g., cuda:1, or "cuda" for multi-GPU auto)

# Core Parameters (General preprocessing for all examples)
--index-dir DIR              # Directory to store the index (default: current directory)
--query "YOUR QUESTION"      # Single query mode. Omit for interactive chat (type 'quit' to exit), and now you can play with your index interactively
--max-items N                # Limit data preprocessing (default: -1, process all data)
--force-rebuild              # Force rebuild index even if it exists

# Embedding Parameters
--embedding-model MODEL      # e.g., facebook/contriever, text-embedding-3-small, mlx-community/Qwen3-Embedding-0.6B-8bit or nomic-embed-text
--embedding-mode MODE        # sentence-transformers, openai, mlx, or ollama

# LLM Parameters (Text generation models)
--llm TYPE                   # LLM backend: openai, ollama, hf, or anthropic (default: openai)
--llm-model MODEL            # Model name (default: gpt-4o) e.g., gpt-4o-mini, llama3.2:1b, Qwen/Qwen2.5-1.5B-Instruct
--thinking-budget LEVEL      # Thinking budget for reasoning models: low/medium/high (supported by o3, o3-mini, GPT-Oss:20b, and other reasoning models)

# Search Parameters
--top-k N                    # Number of results to retrieve (default: 20)
--search-complexity N        # Search complexity for graph traversal (default: 32)

# Chunking Parameters
--chunk-size N               # Size of text chunks (default varies by source: 256 for most, 192 for WeChat)
--chunk-overlap N            # Overlap between chunks (default varies: 25-128 depending on source)

# Index Building Parameters
--backend-name NAME          # Backend to use: hnsw or diskann (default: hnsw)
--graph-degree N             # Graph degree for index construction (default: 32)
--build-complexity N         # Build complexity for index construction (default: 64)
--compact / --no-compact     # Use compact storage (default: true). Must be `no-compact` for `no-recompute` build.
--recompute / --no-recompute # Enable/disable embedding recomputation (default: enabled). Should not do a `no-recompute` search in a `recompute` build.
```

</details>

### 📄 Personal Data Manager: Process Any Documents (`.pdf`, `.txt`, `.md`)!

Ask questions directly about your personal PDFs, documents, and any directory containing your files!

<p align="center">
  <img src="videos/paper_clear.gif" alt="LEANN Document Search Demo" width="600">
</p>

The example below asks a question about summarizing our paper (uses default data in `data/`, which is a directory with diverse data sources: two papers, Pride and Prejudice, and a Technical report about LLM in Huawei in Chinese), and this is the **easiest example** to run here:

```bash
source .venv/bin/activate # Don't forget to activate the virtual environment
python -m apps.document_rag --query "What are the main techniques LEANN explores?"
```

<details>
<summary><strong>📋 Click to expand: Document-Specific Arguments</strong></summary>

#### Parameters
```bash
--data-dir DIR           # Directory containing documents to process (default: data)
--file-types .ext .ext   # Filter by specific file types (optional - all LlamaIndex supported types if omitted)
```

#### Example Commands
```bash
# Process all documents with larger chunks for academic papers
python -m apps.document_rag --data-dir "~/Documents/Papers" --chunk-size 1024

# Filter only markdown and Python files with smaller chunks
python -m apps.document_rag --data-dir "./docs" --chunk-size 256 --file-types .md .py

# Enable AST-aware chunking for code files
python -m apps.document_rag --enable-code-chunking --data-dir "./my_project"

# Or use the specialized code RAG for better code understanding
python -m apps.code_rag --repo-dir "./my_codebase" --query "How does authentication work?"
```

</details>

### 🎨 ColQwen: Multimodal PDF Retrieval with Vision-Language Models

Search through PDFs using both text and visual understanding with ColQwen2/ColPali models. Perfect for research papers, technical documents, and any PDFs with complex layouts, figures, or diagrams.

> **🍎 Mac Users**: ColQwen is optimized for Apple Silicon with MPS acceleration for faster inference!

```bash
# Build index from PDFs
python -m apps.colqwen_rag build --pdfs ./my_papers/ --index research_papers

# Search with text queries
python -m apps.colqwen_rag search research_papers "How does attention mechanism work?"

# Interactive Q&A
python -m apps.colqwen_rag ask research_papers --interactive
```

<details>
<summary><strong>📋 Click to expand: ColQwen Setup & Usage</strong></summary>

#### Prerequisites
```bash
# Install dependencies
uv pip install colpali_engine pdf2image pillow matplotlib qwen_vl_utils einops seaborn
brew install poppler  # macOS only, for PDF processing
```

#### Build Index
```bash
python -m apps.colqwen_rag build \
  --pdfs ./pdf_directory/ \
  --index my_index \
  --model colqwen2  # or colpali
```

#### Search
```bash
python -m apps.colqwen_rag search my_index "your question here" --top-k 5
```

#### Models
- **ColQwen2** (`colqwen2`): Latest vision-language model with improved performance
- **ColPali** (`colpali`): Proven multimodal retriever

For detailed usage, see the [ColQwen Guide](docs/COLQWEN_GUIDE.md).

</details>

### 📧 Your Personal Email Secretary: RAG on Apple Mail!

> **Note:** The examples below currently support macOS only. Windows support coming soon.


<p align="center">
  <img src="videos/mail_clear.gif" alt="LEANN Email Search Demo" width="600">
</p>

Before running the example below, you need to grant full disk access to your terminal/VS Code in System Preferences → Privacy & Security → Full Disk Access.

```bash
python -m apps.email_rag --query "What's the food I ordered by DoorDash or Uber Eats mostly?"
```
**780K email chunks → 78MB storage.** Finally, search your email like you search Google.

<details>
<summary><strong>📋 Click to expand: Email-Specific Arguments</strong></summary>

#### Parameters
```bash
--mail-path PATH         # Path to specific mail directory (auto-detects if omitted)
--include-html          # Include HTML content in processing (useful for newsletters)
```

#### Example Commands
```bash
# Search work emails from a specific account
python -m apps.email_rag --mail-path "~/Library/Mail/V10/WORK_ACCOUNT"

# Find all receipts and order confirmations (includes HTML)
python -m apps.email_rag --query "receipt order confirmation invoice" --include-html
```

</details>

<details>
<summary><strong>📋 Click to expand: Example queries you can try</strong></summary>

Once the index is built, you can ask questions like:
- "Find emails from my boss about deadlines"
- "What did John say about the project timeline?"
- "Show me emails about travel expenses"
</details>

### 🔍 Time Machine for the Web: RAG Your Entire Chrome Browser History!

<p align="center">
  <img src="videos/google_clear.gif" alt="LEANN Browser History Search Demo" width="600">
</p>

```bash
python -m apps.browser_rag --query "Tell me my browser history about machine learning?"
```
**38K browser entries → 6MB storage.** Your browser history becomes your personal search engine.

<details>
<summary><strong>📋 Click to expand: Browser-Specific Arguments</strong></summary>

#### Parameters
```bash
--chrome-profile PATH    # Path to Chrome profile directory (auto-detects if omitted)
```

#### Example Commands
```bash
# Search academic research from your browsing history
python -m apps.browser_rag --query "arxiv papers machine learning transformer architecture"

# Track competitor analysis across work profile
python -m apps.browser_rag --chrome-profile "~/Library/Application Support/Google/Chrome/Work Profile" --max-items 5000
```

</details>

<details>
<summary><strong>📋 Click to expand: How to find your Chrome profile</strong></summary>

The default Chrome profile path is configured for a typical macOS setup. If you need to find your specific Chrome profile:

1. Open Terminal
2. Run: `ls ~/Library/Application\ Support/Google/Chrome/`
3. Look for folders like "Default", "Profile 1", "Profile 2", etc.
4. Use the full path as your `--chrome-profile` argument

**Common Chrome profile locations:**
- macOS: `~/Library/Application Support/Google/Chrome/Default`
- Linux: `~/.config/google-chrome/Default`

</details>

<details>
<summary><strong>💬 Click to expand: Example queries you can try</strong></summary>

Once the index is built, you can ask questions like:

- "What websites did I visit about machine learning?"
- "Find my search history about programming"
- "What YouTube videos did I watch recently?"
- "Show me websites I visited about travel planning"

</details>

### 💬 WeChat Detective: Unlock Your Golden Memories!

<p align="center">
  <img src="videos/wechat_clear.gif" alt="LEANN WeChat Search Demo" width="600">
</p>

```bash
python -m apps.wechat_rag --query "Show me all group chats about weekend plans"
```
**400K messages → 64MB storage** Search years of chat history in any language.


<details>
<summary><strong>🔧 Click to expand: Installation Requirements</strong></summary>

First, you need to install the [WeChat exporter](https://github.com/sunnyyoung/WeChatTweak-CLI),

```bash
brew install sunnyyoung/repo/wechattweak-cli
```

or install it manually (if you have issues with Homebrew):

```bash
sudo packages/wechat-exporter/wechattweak-cli install
```

**Troubleshooting:**
- **Installation issues**: Check the [WeChatTweak-CLI issues page](https://github.com/sunnyyoung/WeChatTweak-CLI/issues/41)
- **Export errors**: If you encounter the error below, try restarting WeChat
  ```bash
  Failed to export WeChat data. Please ensure WeChat is running and WeChatTweak is installed.
  Failed to find or export WeChat data. Exiting.
  ```
</details>

<details>
<summary><strong>📋 Click to expand: WeChat-Specific Arguments</strong></summary>

#### Parameters
```bash
--export-dir DIR         # Directory to store exported WeChat data (default: wechat_export_direct)
--force-export          # Force re-export even if data exists
```

#### Example Commands
```bash
# Search for travel plans discussed in group chats
python -m apps.wechat_rag --query "travel plans" --max-items 10000

# Re-export and search recent chats (useful after new messages)
python -m apps.wechat_rag --force-export --query "work schedule"
```

</details>

<details>
<summary><strong>💬 Click to expand: Example queries you can try</strong></summary>

Once the index is built, you can ask questions like:

- "我想买魔术师约翰逊的球衣，给我一些对应聊天记录?" (Chinese: Show me chat records about buying Magic Johnson's jersey)

</details>

### 🤖 ChatGPT Chat History: Your Personal AI Conversation Archive!

Transform your ChatGPT conversations into a searchable knowledge base! Search through all your ChatGPT discussions about coding, research, brainstorming, and more.

```bash
python -m apps.chatgpt_rag --export-path chatgpt_export.html --query "How do I create a list in Python?"
```

**Unlock your AI conversation history.** Never lose track of valuable insights from your ChatGPT discussions again.

<details>
<summary><strong>📋 Click to expand: How to Export ChatGPT Data</strong></summary>

**Step-by-step export process:**

1. **Sign in to ChatGPT**
2. **Click your profile icon** in the top right corner
3. **Navigate to Settings** → **Data Controls**
4. **Click "Export"** under Export Data
5. **Confirm the export** request
6. **Download the ZIP file** from the email link (expires in 24 hours)
7. **Extract or use directly** with LEANN

**Supported formats:**
- `.html` files from ChatGPT exports
- `.zip` archives from ChatGPT
- Directories with multiple export files

</details>

<details>
<summary><strong>📋 Click to expand: ChatGPT-Specific Arguments</strong></summary>

#### Parameters
```bash
--export-path PATH           # Path to ChatGPT export file (.html/.zip) or directory (default: ./chatgpt_export)
--separate-messages         # Process each message separately instead of concatenated conversations
--chunk-size N              # Text chunk size (default: 512)
--chunk-overlap N           # Overlap between chunks (default: 128)
```

#### Example Commands
```bash
# Basic usage with HTML export
python -m apps.chatgpt_rag --export-path conversations.html

# Process ZIP archive from ChatGPT
python -m apps.chatgpt_rag --export-path chatgpt_export.zip

# Search with specific query
python -m apps.chatgpt_rag --export-path chatgpt_data.html --query "Python programming help"

# Process individual messages for fine-grained search
python -m apps.chatgpt_rag --separate-messages --export-path chatgpt_export.html

# Process directory containing multiple exports
python -m apps.chatgpt_rag --export-path ./chatgpt_exports/ --max-items 1000
```

</details>

<details>
<summary><strong>💡 Click to expand: Example queries you can try</strong></summary>

Once your ChatGPT conversations are indexed, you can search with queries like:
- "What did I ask ChatGPT about Python programming?"
- "Show me conversations about machine learning algorithms"
- "Find discussions about web development frameworks"
- "What coding advice did ChatGPT give me?"
- "Search for conversations about debugging techniques"
- "Find ChatGPT's recommendations for learning resources"

</details>

### 🤖 Claude Chat History: Your Personal AI Conversation Archive!

Transform your Claude conversations into a searchable knowledge base! Search through all your Claude discussions about coding, research, brainstorming, and more.

```bash
python -m apps.claude_rag --export-path claude_export.json --query "What did I ask about Python dictionaries?"
```

**Unlock your AI conversation history.** Never lose track of valuable insights from your Claude discussions again.

<details>
<summary><strong>📋 Click to expand: How to Export Claude Data</strong></summary>

**Step-by-step export process:**

1. **Open Claude** in your browser
2. **Navigate to Settings** (look for gear icon or settings menu)
3. **Find Export/Download** options in your account settings
4. **Download conversation data** (usually in JSON format)
5. **Place the file** in your project directory

*Note: Claude export methods may vary depending on the interface you're using. Check Claude's help documentation for the most current export instructions.*

**Supported formats:**
- `.json` files (recommended)
- `.zip` archives containing JSON data
- Directories with multiple export files

</details>

<details>
<summary><strong>📋 Click to expand: Claude-Specific Arguments</strong></summary>

#### Parameters
```bash
--export-path PATH           # Path to Claude export file (.json/.zip) or directory (default: ./claude_export)
--separate-messages         # Process each message separately instead of concatenated conversations
--chunk-size N              # Text chunk size (default: 512)
--chunk-overlap N           # Overlap between chunks (default: 128)
```

#### Example Commands
```bash
# Basic usage with JSON export
python -m apps.claude_rag --export-path my_claude_conversations.json

# Process ZIP archive from Claude
python -m apps.claude_rag --export-path claude_export.zip

# Search with specific query
python -m apps.claude_rag --export-path claude_data.json --query "machine learning advice"

# Process individual messages for fine-grained search
python -m apps.claude_rag --separate-messages --export-path claude_export.json

# Process directory containing multiple exports
python -m apps.claude_rag --export-path ./claude_exports/ --max-items 1000
```

</details>

<details>
<summary><strong>💡 Click to expand: Example queries you can try</strong></summary>

Once your Claude conversations are indexed, you can search with queries like:
- "What did I ask Claude about Python programming?"
- "Show me conversations about machine learning algorithms"
- "Find discussions about software architecture patterns"
- "What debugging advice did Claude give me?"
- "Search for conversations about data structures"
- "Find Claude's recommendations for learning resources"

</details>

### 💬 iMessage History: Your Personal Conversation Archive!

Transform your iMessage conversations into a searchable knowledge base! Search through all your text messages, group chats, and conversations with friends, family, and colleagues.

```bash
python -m apps.imessage_rag --query "What did we discuss about the weekend plans?"
```

**Unlock your message history.** Never lose track of important conversations, shared links, or memorable moments from your iMessage history.

<details>
<summary><strong>📋 Click to expand: How to Access iMessage Data</strong></summary>

**iMessage data location:**

iMessage conversations are stored in a SQLite database on your Mac at:
```
~/Library/Messages/chat.db
```

**Important setup requirements:**

1. **Grant Full Disk Access** to your terminal or IDE:
   - Open **System Preferences** → **Security & Privacy** → **Privacy**
   - Select **Full Disk Access** from the left sidebar
   - Click the **+** button and add your terminal app (Terminal, iTerm2) or IDE (VS Code, etc.)
   - Restart your terminal/IDE after granting access

2. **Alternative: Use a backup database**
   - If you have Time Machine backups or manual copies of the database
   - Use `--db-path` to specify a custom location

**Supported formats:**
- Direct access to `~/Library/Messages/chat.db` (default)
- Custom database path with `--db-path`
- Works with backup copies of the database

</details>

<details>
<summary><strong>📋 Click to expand: iMessage-Specific Arguments</strong></summary>

#### Parameters
```bash
--db-path PATH                    # Path to chat.db file (default: ~/Library/Messages/chat.db)
--concatenate-conversations       # Group messages by conversation (default: True)
--no-concatenate-conversations    # Process each message individually
--chunk-size N                    # Text chunk size (default: 1000)
--chunk-overlap N                 # Overlap between chunks (default: 200)
```

#### Example Commands
```bash
# Basic usage (requires Full Disk Access)
python -m apps.imessage_rag

# Search with specific query
python -m apps.imessage_rag --query "family dinner plans"

# Use custom database path
python -m apps.imessage_rag --db-path /path/to/backup/chat.db

# Process individual messages instead of conversations
python -m apps.imessage_rag --no-concatenate-conversations

# Limit processing for testing
python -m apps.imessage_rag --max-items 100 --query "weekend"
```

</details>

<details>
<summary><strong>💡 Click to expand: Example queries you can try</strong></summary>

Once your iMessage conversations are indexed, you can search with queries like:
- "What did we discuss about vacation plans?"
- "Find messages about restaurant recommendations"
- "Show me conversations with John about the project"
- "Search for shared links about technology"
- "Find group chat discussions about weekend events"
- "What did mom say about the family gathering?"

</details>

### MCP Integration: RAG on Live Data from Any Platform

Connect to live data sources through the Model Context Protocol (MCP). LEANN now supports real-time RAG on platforms like Slack, Twitter, and more through standardized MCP servers.

**Key Benefits:**
- **Live Data Access**: Fetch real-time data without manual exports
- **Standardized Protocol**: Use any MCP-compatible server
- **Easy Extension**: Add new platforms with minimal code
- **Secure Access**: MCP servers handle authentication

#### 💬 Slack Messages: Search Your Team Conversations

Transform your Slack workspace into a searchable knowledge base! Find discussions, decisions, and shared knowledge across all your channels.

```bash
# Test MCP server connection
python -m apps.slack_rag --mcp-server "slack-mcp-server" --test-connection

# Index and search Slack messages
python -m apps.slack_rag \
  --mcp-server "slack-mcp-server" \
  --workspace-name "my-team" \
  --channels general dev-team random \
  --query "What did we decide about the product launch?"
```

**📖 Comprehensive Setup Guide**: For detailed setup instructions, troubleshooting common issues (like "users cache is not ready yet"), and advanced configuration options, see our [**Slack Setup Guide**](docs/slack-setup-guide.md).

**Quick Setup:**
1. Install a Slack MCP server (e.g., `npm install -g slack-mcp-server`)
2. Create a Slack App and get API credentials (see detailed guide above)
3. Set environment variables:
   ```bash
   export SLACK_BOT_TOKEN="xoxb-your-bot-token"
   export SLACK_APP_TOKEN="xapp-your-app-token"  # Optional
   ```
4. Test connection with `--test-connection` flag

**Arguments:**
- `--mcp-server`: Command to start the Slack MCP server
- `--workspace-name`: Slack workspace name for organization
- `--channels`: Specific channels to index (optional)
- `--concatenate-conversations`: Group messages by channel (default: true)
- `--max-messages-per-channel`: Limit messages per channel (default: 100)
- `--max-retries`: Maximum retries for cache sync issues (default: 5)
- `--retry-delay`: Initial delay between retries in seconds (default: 2.0)

#### 🐦 Twitter Bookmarks: Your Personal Tweet Library

Search through your Twitter bookmarks! Find that perfect article, thread, or insight you saved for later.

```bash
# Test MCP server connection
python -m apps.twitter_rag --mcp-server "twitter-mcp-server" --test-connection

# Index and search Twitter bookmarks
python -m apps.twitter_rag \
  --mcp-server "twitter-mcp-server" \
  --max-bookmarks 1000 \
  --query "What AI articles did I bookmark about machine learning?"
```

**Setup Requirements:**
1. Install a Twitter MCP server (e.g., `npm install -g twitter-mcp-server`)
2. Get Twitter API credentials:
   - Apply for a Twitter Developer Account at [developer.twitter.com](https://developer.twitter.com)
   - Create a new app in the Twitter Developer Portal
   - Generate API keys and access tokens with "Read" permissions
   - For bookmarks access, you may need Twitter API v2 with appropriate scopes
   ```bash
   export TWITTER_API_KEY="your-api-key"
   export TWITTER_API_SECRET="your-api-secret"
   export TWITTER_ACCESS_TOKEN="your-access-token"
   export TWITTER_ACCESS_TOKEN_SECRET="your-access-token-secret"
   ```
3. Test connection with `--test-connection` flag

**Arguments:**
- `--mcp-server`: Command to start the Twitter MCP server
- `--username`: Filter bookmarks by username (optional)
- `--max-bookmarks`: Maximum bookmarks to fetch (default: 1000)
- `--no-tweet-content`: Exclude tweet content, only metadata
- `--no-metadata`: Exclude engagement metadata

</details>

<details>
<summary><strong>💡 Click to expand: Example queries you can try</strong></summary>

**Slack Queries:**
- "What did the team discuss about the project deadline?"
- "Find messages about the new feature launch"
- "Show me conversations about budget planning"
- "What decisions were made in the dev-team channel?"

**Twitter Queries:**
- "What AI articles did I bookmark last month?"
- "Find tweets about machine learning techniques"
- "Show me bookmarked threads about startup advice"
- "What Python tutorials did I save?"

</details>
<summary><strong>🔧 Using MCP with CLI Commands</strong></summary>

**Want to use MCP data with regular LEANN CLI?** You can combine MCP apps with CLI commands:

```bash
# Step 1: Use MCP app to fetch and index data
python -m apps.slack_rag --mcp-server "slack-mcp-server" --workspace-name "my-team"

# Step 2: The data is now indexed and available via CLI
leann search slack_messages "project deadline"
leann ask slack_messages "What decisions were made about the product launch?"

# Same for Twitter bookmarks
python -m apps.twitter_rag --mcp-server "twitter-mcp-server"
leann search twitter_bookmarks "machine learning articles"
```

**MCP vs Manual Export:**
- **MCP**: Live data, automatic updates, requires server setup
- **Manual Export**: One-time setup, works offline, requires manual data export

</details>

<details>
<summary><strong>🔧 Adding New MCP Platforms</strong></summary>

Want to add support for other platforms? LEANN's MCP integration is designed for easy extension:

1. **Find or create an MCP server** for your platform
2. **Create a reader class** following the pattern in `apps/slack_data/slack_mcp_reader.py`
3. **Create a RAG application** following the pattern in `apps/slack_rag.py`
4. **Test and contribute** back to the community!

**Popular MCP servers to explore:**
- GitHub repositories and issues
- Discord messages
- Notion pages
- Google Drive documents
- And many more in the MCP ecosystem!

</details>

### 🚀 Claude Code Integration: Transform Your Development Workflow!

<details>
<summary><strong>AST‑Aware Code Chunking</strong></summary>

LEANN features intelligent code chunking that preserves semantic boundaries (functions, classes, methods) for Python, Java, C#, and TypeScript, improving code understanding compared to text-based chunking.

📖 Read the [AST Chunking Guide →](docs/ast_chunking_guide.md)

</details>

**The future of code assistance is here.** Transform your development workflow with LEANN's native MCP integration for Claude Code. Index your entire codebase and get intelligent code assistance directly in your IDE.

**Key features:**
- 🔍 **Semantic code search** across your entire project, fully local index and lightweight
- 🧠 **AST-aware chunking** preserves code structure (functions, classes)
- 📚 **Context-aware assistance** for debugging and development
- 🚀 **Zero-config setup** with automatic language detection

```bash
# Install LEANN globally for MCP integration
uv tool install leann-core --with leann
claude mcp add --scope user leann-server -- leann_mcp
# Setup is automatic - just start using Claude Code!
```
Try our fully agentic pipeline with auto query rewriting, semantic search planning, and more:

![LEANN MCP Integration](assets/mcp_leann.png)

**🔥 Ready to supercharge your coding?** [Complete Setup Guide →](packages/leann-mcp/README.md)

## Command Line Interface

LEANN includes a powerful CLI for document processing and search. Perfect for quick document indexing and interactive chat.

### Installation

If you followed the Quick Start, `leann` is already installed in your virtual environment:
```bash
source .venv/bin/activate
leann --help
```

**To make it globally available:**
```bash
# Install the LEANN CLI globally using uv tool
uv tool install leann-core --with leann


# Now you can use leann from anywhere without activating venv
leann --help
```

> **Note**: Global installation is required for Claude Code integration. The `leann_mcp` server depends on the globally available `leann` command.



### Usage Examples

```bash
# build from a specific directory, and my_docs is the index name(Here you can also build from multiple dict or multiple files)
leann build my-docs --docs ./your_documents

# Search your documents
leann search my-docs "machine learning concepts"

# Interactive chat with your documents
leann ask my-docs --interactive

# Ask a single question (non-interactive)
leann ask my-docs "Where are prompts configured?"

# Detect file changes since last build/watch checkpoint
leann watch my-docs

# List all your indexes
leann list

# Remove an index
leann remove my-docs
```

**Key CLI features:**
- Auto-detects document formats (PDF, TXT, MD, DOCX, PPTX + code files)
- **🧠 AST-aware chunking** for Python, Java, C#, TypeScript files
- Smart text chunking with overlap for all other content
- **📂 File change detection** via Merkle tree snapshots (`leann watch`)
- Multiple LLM providers (Ollama, OpenAI, HuggingFace)
- Organized index storage in `.leann/indexes/` (project-local)
- Support for advanced search parameters

<details>
<summary><strong>📋 Click to expand: Complete CLI Reference</strong></summary>

You can use `leann --help`, or `leann build --help`, `leann search --help`, `leann watch --help`, `leann ask --help`, `leann list --help`, `leann remove --help` to get the complete CLI reference.

**Build Command:**
```bash
leann build INDEX_NAME --docs DIRECTORY|FILE [DIRECTORY|FILE ...] [OPTIONS]

Options:
  --backend {hnsw,diskann}     Backend to use (default: hnsw)
  --embedding-model MODEL      Embedding model (default: facebook/contriever)
  --graph-degree N             Graph degree (default: 32)
  --complexity N               Build complexity (default: 64)
  --force                      Force rebuild existing index
  --compact / --no-compact     Use compact storage (default: true). Must be `no-compact` for `no-recompute` build.
  --recompute / --no-recompute Enable recomputation (default: true)
```

**Search Command:**
```bash
leann search INDEX_NAME QUERY [OPTIONS]

Options:
  --top-k N                     Number of results (default: 5)
  --complexity N                Search complexity (default: 64)
  --recompute / --no-recompute  Enable/disable embedding recomputation (default: enabled). Should not do a `no-recompute` search in a `recompute` build.
  --pruning-strategy {global,local,proportional}
```

**Watch Command:**
```bash
leann watch INDEX_NAME

# Compares the current file system state against the last checkpoint (Merkle tree snapshot)
# and reports which files have been added, removed, or modified, along with their chunk IDs.
#
# - Automatically saves a new checkpoint after detecting changes
# - Each subsequent run compares against the most recent checkpoint
# - File change detection uses SHA-256 content hashing via a Merkle tree
#
# Example output:
#   === Changes since last checkpoint ===
#   modified (1):
#     - /path/to/file.py
#       chunks: 42, 43, 44
```

**Ask Command:**
```bash
leann ask INDEX_NAME [OPTIONS]

Options:
  --llm {ollama,openai,hf,anthropic}    LLM provider (default: ollama)
  --model MODEL                         Model name (default: qwen3:8b)
  --interactive                         Interactive chat mode
  --top-k N                             Retrieval count (default: 20)
```

**List Command:**
```bash
leann list

# Lists all indexes across all projects with status indicators:
# ✅ - Index is complete and ready to use
# ❌ - Index is incomplete or corrupted
# 📁 - CLI-created index (in .leann/indexes/)
# 📄 - App-created index (*.leann.meta.json files)
```

**Remove Command:**
```bash
leann remove INDEX_NAME [OPTIONS]

Options:
  --force, -f    Force removal without confirmation

# Smart removal: automatically finds and safely removes indexes
# - Shows all matching indexes across projects
# - Requires confirmation for cross-project removal
# - Interactive selection when multiple matches found
# - Supports both CLI and app-created indexes
```

</details>

## 🚀 Advanced Features

### 🎯 Metadata Filtering

LEANN supports a simple metadata filtering system to enable sophisticated use cases like document filtering by date/type, code search by file extension, and content management based on custom criteria.

```python
# Add metadata during indexing
builder.add_text(
    "def authenticate_user(token): ...",
    metadata={"file_extension": ".py", "lines_of_code": 25}
)

# Search with filters
results = searcher.search(
    query="authentication function",
    metadata_filters={
        "file_extension": {"==": ".py"},
        "lines_of_code": {"<": 100}
    }
)
```

**Supported operators**: `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not_in`, `contains`, `starts_with`, `ends_with`, `is_true`, `is_false`

📖 **[Complete Metadata filtering guide →](docs/metadata_filtering.md)**

### 🔍 Grep Search

For exact text matching instead of semantic search, use the `use_grep` parameter:

```python
# Exact text search
results = searcher.search("banana‑crocodile", use_grep=True, top_k=1)
```

**Use cases**: Finding specific code patterns, error messages, function names, or exact phrases where semantic similarity isn't needed.

📖 **[Complete grep search guide →](docs/grep_search.md)**

## 🏗️ Architecture & How It Works

<p align="center">
  <img src="assets/arch.png" alt="LEANN Architecture" width="800">
</p>

**The magic:** Most vector DBs store every single embedding (expensive). LEANN stores a pruned graph structure (cheap) and recomputes embeddings only when needed (fast).

**Core techniques:**
- **Graph-based selective recomputation:** Only compute embeddings for nodes in the search path
- **High-degree preserving pruning:** Keep important "hub" nodes while removing redundant connections
- **Dynamic batching:** Efficiently batch embedding computations for GPU utilization
- **Two-level search:** Smart graph traversal that prioritizes promising nodes

**Backends:**
- **HNSW** (default): Ideal for most datasets with maximum storage savings through full recomputation
- **DiskANN**: Advanced option with superior search performance, using PQ-based graph traversal with real-time reranking for the best speed-accuracy trade-off

## Benchmarks

**[DiskANN vs HNSW Performance Comparison →](benchmarks/diskann_vs_hnsw_speed_comparison.py)** - Compare search performance between both backends

**[Simple Example: Compare LEANN vs FAISS →](benchmarks/compare_faiss_vs_leann.py)** - See storage savings in action

### 📊 Storage Comparison

| System | DPR (2.1M) | Wiki (60M) | Chat (400K) | Email (780K) | Browser (38K) |
|--------|-------------|------------|-------------|--------------|---------------|
| Traditional vector database (e.g., FAISS) | 3.8 GB      | 201 GB     | 1.8 GB     | 2.4 GB      | 130 MB        |
| LEANN  | 324 MB      | 6 GB       | 64 MB       | 79 MB       | 6.4 MB        |
| Savings| 91%         | 97%        | 97%         | 97%         | 95%           |



## Reproduce Our Results

```bash
uv run benchmarks/run_evaluation.py    # Will auto-download evaluation data and run benchmarks
uv run benchmarks/run_evaluation.py benchmarks/data/indices/rpj_wiki/rpj_wiki --num-queries 2000    # After downloading data, you can run the benchmark with our biggest index
```

The evaluation script downloads data automatically on first run. The last three results were tested with partial personal data, and you can reproduce them with your own data!
## 🔬 Paper

If you find Leann useful, please cite:

**[LEANN: A Low-Storage Vector Index](https://arxiv.org/abs/2506.08276)**

```bibtex
@misc{wang2025leannlowstoragevectorindex,
      title={LEANN: A Low-Storage Vector Index},
      author={Yichuan Wang and Shu Liu and Zhifei Li and Yongji Wu and Ziming Mao and Yilong Zhao and Xiao Yan and Zhiying Xu and Yang Zhou and Ion Stoica and Sewon Min and Matei Zaharia and Joseph E. Gonzalez},
      year={2025},
      eprint={2506.08276},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2506.08276},
}
```

## ✨ [Detailed Features →](docs/features.md)

## 🤝 [CONTRIBUTING →](docs/CONTRIBUTING.md)


## ❓ [FAQ →](docs/faq.md)


## 📈 [Roadmap →](docs/roadmap.md)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Core Contributors: [Yichuan Wang](https://yichuan-w.github.io/) & [Zhifei Li](https://github.com/andylizf).

Active Contributors: [Gabriel Dehan](https://github.com/gabriel-dehan), [Aakash Suresh](https://github.com/ASuresh0524)


We welcome more contributors! Feel free to open issues or submit PRs.

This work is done at [**Berkeley Sky Computing Lab**](https://sky.cs.berkeley.edu/).

## Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=StarTrail-org/LEANN&type=Date)](https://star-history.dera.page/#StarTrail-org/LEANN&Date)
<p align="center">
  <strong>⭐ Star us on GitHub if Leann is useful for your research or applications!</strong>
</p>

<p align="center">
  Made with ❤️ by the Leann team
</p>

## 🤖 Explore LEANN with AI

LEANN is indexed on [DeepWiki](https://deepwiki.com/StarTrail-org/LEANN), so you can ask questions to LLMs using Deep Research to explore the codebase and get help to add new features.
