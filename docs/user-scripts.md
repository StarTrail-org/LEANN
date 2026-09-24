# User Scripts: Daily Life with LEANN

This documentation describes the automation scripts prepared for using LEANN in daily life.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yichuan-w/LEANN.git
cd LEANN
```

### 2. Copy Scripts to ~/bin Folder

```bash
# Create ~/bin directory if it doesn't exist
mkdir -p ~/bin

# Copy scripts
cp bin/leann-sync-all.sh ~/bin/
cp bin/leann-sync-dev.sh ~/bin/
cp bin/leann-sync-personal.sh ~/bin/
cp bin/leann-sync-brave.sh ~/bin/
cp bin/leann-sync-mail.sh ~/bin/
cp bin/leann-sync-imessage.sh ~/bin/
cp bin/leann-sync-calendar.sh ~/bin/

# Make executable
chmod +x ~/bin/leann-*.sh
```

### 3. Add to PATH

```bash
# Add to ~/.zshrc or ~/.bashrc
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### 4. Install Ollama (for Embedding)

```bash
# Start Ollama
ollama serve &

# Download embedding model
ollama pull nomic-embed-text
```

### 5. Install LEANN

```bash
cd LEANN
uv sync --extra diskann
```

## Usage

### Quick Start

```bash
# Update all indexes
leann-sync-all.sh
```

### Individual Scripts

| Script | Description |
|--------|-------------|
| `leann-sync-all.sh` | Updates all indexes sequentially |
| `leann-sync-dev.sh` | Indexes development environment code |
| `leann-sync-personal.sh` | Indexes personal documents (Documents, Nextcloud) |
| `leann-sync-brave.sh` | Indexes Brave browser history |
| `leann-sync-mail.sh` | Indexes Apple Mail emails |
| `leann-sync-imessage.sh` | Indexes iMessage messages |
| `leann-sync-calendar.sh` | Indexes Apple Calendar events |

### Example Usage Scenarios

#### Scenario 1: Daily Development Workflow

```bash
# Update your development environment every morning
leann-sync-dev.sh
```

This command:
- Scans code in ~/Development folder
- Preserves code structure with AST-aware chunking
- Creates index with DiskANN backend

#### Scenario 2: Personal Document Search

```bash
# Index your personal documents
leann-sync-personal.sh
```

This command:
- Scans ~/Documents, ~/Nextcloud, ~/Nextcloud2 folders
- Indexes all documents (PDF, TXT, MD, Word, Excel, PowerPoint)

#### Scenario 3: Browser History Search

```bash
# Index your Brave browser history
leann-sync-brave.sh
```

#### Scenario 4: Email Search

```bash
# Index your Apple Mail emails
leann-sync-mail.sh
```

#### Scenario 5: iMessage Search

```bash
# Index your iMessage messages
leann-sync-imessage.sh
```

## Script Customization

### Create Your Own Script

```bash
#!/bin/bash
# my-custom-index.sh

export LEANN_HOME="$HOME/.leann"
export OLLAMA_HOST="http://localhost:11434"

leann build my-custom-index \
  --docs ~/MyDocuments \
  --embedding-mode ollama \
  --embedding-model nomic-embed-text \
  --backend-name diskann \
  --force
```

### Modify Parameters

You can modify parameters in scripts according to your needs:

```bash
# Larger chunk size
--doc-chunk-size 2048

# Change embedding model
--embedding-model BAAI/bge-base-en-v1.5

# Use HNSW backend
--backend-name hnsw
```

## Frequently Asked Questions

### Q: I get "leann: command not found" error

Make sure LEANN installation is in your PATH:
```bash
export PATH="/path/to/LEANN/packages/leann-core:$PATH"
```

### Q: Ollama connection failed

Make sure Ollama is running:
```bash
ollama serve &
ollama list
```

### Q: Index creation is too slow

Try with a smaller dataset:
```bash
--max-count 1000
```

## Advanced Usage

### LEANN CLI Commands

```bash
# Create index
leann build my-index --docs ./documents

# Search
leann search my-index "search query"

# Ask questions
leann ask my-index --interactive

# List indexes
leann list

# Remove index
leann remove my-index
```

### Embedding Modes

| Mode | Description |
|------|-------------|
| `sentence-transformers` | HuggingFace models |
| `openai` | OpenAI API |
| `mlx` | Apple Silicon MLX |
| `ollama` | Local Ollama |

### Backend Selection

| Backend | Use Case |
|---------|----------|
| `hnsw` | Small-medium scale (<10M vectors) |
| `diskann` | Large scale, disk-based |

---

This documentation is prepared to make LEANN easy to use in daily life. If you have any questions, please ask on GitHub Issues.
