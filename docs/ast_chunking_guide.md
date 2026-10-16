# AST-Aware Code chunking guide

## Overview

This guide covers best practices for using AST-aware code chunking in LEANN. AST chunking provides better semantic understanding of code structure compared to traditional text-based chunking.

## Quick Start

### Basic Usage

```bash
# Enable AST chunking for mixed content (code + docs)
python -m apps.document_rag --enable-code-chunking --data-dir ./my_project

# Specialized code repository indexing
python -m apps.code_rag --repo-dir ./my_codebase

# Global CLI with AST support
leann build my-code-index --docs ./src --use-ast-chunking
```

### Installation

```bash
# Install LEANN with AST chunking support
uv pip install -e "."
```

#### For normal users (PyPI install)
- Use `pip install leann` or `uv pip install leann`.
- `astchunk` is pulled automatically from PyPI as a dependency; no extra steps.

#### For developers (from source, editable)
```bash
git clone https://github.com/StarTrail-org/LEANN.git leann
cd leann
git submodule update --init --recursive
uv sync
```
- This repo vendors `astchunk` as a git submodule at `packages/astchunk-leann` (our fork).
- `[tool.uv.sources]` maps the `astchunk` package to that path in editable mode.
- You can edit code under `packages/astchunk-leann` and Python will use your changes immediately (no separate `pip install astchunk` needed).

## Best Practices

### When to Use AST Chunking

✅ **Recommended for:**
- Code repositories with multiple languages
- Mixed documentation and code content
- Complex codebases with deep function/class hierarchies
- When working with Claude Code for code assistance

❌ **Not recommended for:**
- Pure text documents
- Very large files (>1MB)
- Languages not supported by tree-sitter

### Optimal Configuration

```bash
# Recommended settings for most codebases
python -m apps.code_rag \
    --repo-dir ./src \
    --ast-chunk-size 768 \
    --ast-chunk-overlap 96 \
    --exclude-dirs .git __pycache__ node_modules build dist
```

### Supported Languages

| Extension | Language | Status |
|-----------|----------|--------|
| `.py` | Python | ✅ Full support |
| `.java` | Java | ✅ Full support |
| `.cs` | C# | ✅ Full support |
| `.ts`, `.tsx` | TypeScript | ✅ Full support |
| `.js`, `.jsx` | JavaScript | ✅ Via TypeScript parser |

## Integration Examples

### Document RAG with Code Support

```python
# Enable code chunking in document RAG
python -m apps.document_rag \
    --enable-code-chunking \
    --data-dir ./project \
    --query "How does authentication work in the codebase?"
```

### Claude Code Integration

When using with Claude Code MCP server, AST chunking provides better context for:
- Code completion and suggestions
- Bug analysis and debugging
- Architecture understanding
- Refactoring assistance

## Troubleshooting

### Common Issues

1. **Fallback to Traditional Chunking**
   - Normal behavior for unsupported languages
   - Check logs for specific language support

2. **Performance with Large Files**
   - Adjust `--max-file-size` parameter
   - Use `--exclude-dirs` to skip unnecessary directories

3. **Quality Issues**
   - Try different `--ast-chunk-size` values (512, 768, 1024)
   - Adjust overlap for better context preservation

### Debug Mode

```bash
export LEANN_LOG_LEVEL=DEBUG
python -m apps.code_rag --repo-dir ./my_code
```

## Migration from Traditional Chunking

Existing workflows continue to work without changes. To enable AST chunking:

```bash
# Before
python -m apps.document_rag --chunk-size 256

# After (maintains traditional chunking for non-code files)
python -m apps.document_rag --enable-code-chunking --chunk-size 256 --ast-chunk-size 768
```

## References

- [astchunk GitHub Repository](https://github.com/yilinjz/astchunk)
- [LEANN MCP Integration](../packages/leann-mcp/README.md)
- [Research Paper](https://arxiv.org/html/2506.15655v1)

---

**Note**: AST chunking maintains full backward compatibility while enhancing code understanding capabilities.
