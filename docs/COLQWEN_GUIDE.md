# ColQwen Integration Guide

Easy-to-use multimodal PDF retrieval with ColQwen2/ColPali models.

## Quick Start

> **🍎 Mac Users**: ColQwen is optimized for Apple Silicon with MPS acceleration for faster inference!

### 1. Install Dependencies
```bash
uv pip install colpali_engine pdf2image pillow matplotlib qwen_vl_utils einops seaborn
brew install poppler  # macOS only, for PDF processing
```

### 2. Basic Usage
```bash
# Build index from PDFs
python -m apps.colqwen_rag build --pdfs ./my_papers/ --index research_papers

# Search with text queries
python -m apps.colqwen_rag search research_papers "How does attention mechanism work?"

# Interactive Q&A
python -m apps.colqwen_rag ask research_papers --interactive
```

## Commands

### Build Index
```bash
python -m apps.colqwen_rag build \
  --pdfs ./pdf_directory/ \
  --index my_index \
  --model colqwen2 \
  --pages-dir ./page_images/  # Optional: save page images
```

**Options:**
- `--pdfs`: Directory containing PDF files (or single PDF path)
- `--index`: Name for the index (required)
- `--model`: `colqwen2` (default) or `colpali`
- `--pages-dir`: Directory to save page images (optional)

### Search Index
```bash
python -m apps.colqwen_rag search my_index "your question here" --top-k 5
```

**Options:**
- `--top-k`: Number of results to return (default: 5)
- `--model`: Model used for search (should match build model)

### Interactive Q&A
```bash
python -m apps.colqwen_rag ask my_index --interactive
```

**Commands in interactive mode:**
- Type your questions naturally
- `help`: Show available commands
- `quit`/`exit`/`q`: Exit interactive mode

## 🧪 Test & Reproduce Results

Run the reproduction test for issue #119:
```bash
python test_colqwen_reproduction.py
```

This will:
1. ✅ Check dependencies
2. 📥 Download sample PDF (Attention Is All You Need paper)
3. 🏗️ Build test index
4. 🔍 Run sample queries
5. 📊 Show how to generate similarity maps

## 🎨 Advanced: Similarity Maps

For visual similarity analysis, use the existing advanced script:
```bash
cd apps/multimodal/vision-based-pdf-multi-vector/
python multi-vector-leann-similarity-map.py
```

Edit the script to customize:
- `QUERY`: Your question
- `MODEL`: "colqwen2" or "colpali"
- `USE_HF_DATASET`: Use HuggingFace dataset or local PDFs
- `SIMILARITY_MAP`: Generate heatmaps
- `ANSWER`: Enable Qwen-VL answer generation

## 🔧 How It Works

### ColQwen2 vs ColPali
- **ColQwen2** (`vidore/colqwen2-v1.0`): Latest vision-language model
- **ColPali** (`vidore/colpali-v1.2`): Proven multimodal retriever

### Architecture
1. **PDF → Images**: Convert PDF pages to images (150 DPI)
2. **Vision Encoding**: Process images with ColQwen2/ColPali
3. **Multi-Vector Index**: Build LEANN HNSW index with multiple embeddings per page
4. **Query Processing**: Encode text queries with same model
5. **Similarity Search**: Find most relevant pages/regions
6. **Visual Maps**: Generate attention heatmaps (optional)

### Device Support
- **CUDA**: Best performance with GPU acceleration
- **MPS**: Apple Silicon Mac support
- **CPU**: Fallback for any system (slower)

Auto-detection: CUDA > MPS > CPU

## 📊 Performance Tips

### For Best Performance:
```bash
# Use ColQwen2 for latest features
--model colqwen2

# Save page images for reuse
--pages-dir ./cached_pages/

# Adjust batch size based on GPU memory
# (automatically handled)
```

### For Large Document Sets:
- Process PDFs in batches
- Use SSD storage for index files
- Consider using CUDA if available

## 🔗 Related Resources

- **Fast-PLAID**: https://github.com/lightonai/fast-plaid
- **Pylate**: https://github.com/lightonai/pylate
- **ColBERT**: https://github.com/stanford-futuredata/ColBERT
- **ColPali Paper**: Vision-Language Models for Document Retrieval
- **Issue #119**: https://github.com/StarTrail-org/LEANN/issues/119

## 🐛 Troubleshooting

### PDF Conversion Issues (macOS)
```bash
# Install poppler
brew install poppler
which pdfinfo && pdfinfo -v
```

### Memory Issues
- Reduce batch size (automatically handled)
- Use CPU instead of GPU: `export CUDA_VISIBLE_DEVICES=""`
- Process fewer PDFs at once

### Model Download Issues
- Ensure internet connection for first run
- Models are cached after first download
- Use HuggingFace mirrors if needed

### Import Errors
```bash
# Ensure all dependencies installed
uv pip install colpali_engine pdf2image pillow matplotlib qwen_vl_utils einops seaborn

# Check PyTorch installation
python -c "import torch; print(torch.__version__)"
```

## 💡 Examples

### Research Paper Analysis
```bash
# Index your research papers
python -m apps.colqwen_rag build --pdfs ~/Papers/AI/ --index ai_papers

# Ask research questions
python -m apps.colqwen_rag search ai_papers "What are the limitations of transformer models?"
python -m apps.colqwen_rag search ai_papers "How does BERT compare to GPT?"
```

### Document Q&A
```bash
# Index business documents
python -m apps.colqwen_rag build --pdfs ~/Documents/Reports/ --index reports

# Interactive analysis
python -m apps.colqwen_rag ask reports --interactive
```

### Visual Analysis
```bash
# Generate similarity maps for specific queries
cd apps/multimodal/vision-based-pdf-multi-vector/
# Edit multi-vector-leann-similarity-map.py with your query
python multi-vector-leann-similarity-map.py
# Check ./figures/ for generated heatmaps
```

---

**🎯 This integration makes ColQwen as easy to use as other LEANN features while maintaining the full power of multimodal document understanding!**
