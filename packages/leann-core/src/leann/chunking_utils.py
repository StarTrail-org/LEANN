"""
Enhanced chunking utilities with AST-aware code chunking support.
Packaged within leann-core so installed wheels can import it reliably.
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

from llama_index.core.node_parser import SentenceSplitter

logger = logging.getLogger(__name__)

# Flag to ensure AST token warning only shown once per session
_ast_token_warning_shown = False


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for a text string.
    Uses conservative estimation: ~4 characters per token for natural text,
    ~1.2 tokens per character for code (worse tokenization).

    Args:
        text: Input text to estimate tokens for

    Returns:
        Estimated token count
    """
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except ImportError:
        # Fallback: Conservative character-based estimation
        # Assume worst case for code: 1.2 tokens per character
        return int(len(text) * 1.2)


def calculate_safe_chunk_size(
    model_token_limit: int,
    overlap_size: int,
    chunking_mode: str = "traditional",
    safety_factor: float = 0.9,
) -> int:
    """
    Calculate safe chunk size accounting for overlap and safety margin.

    Args:
        model_token_limit: Maximum tokens supported by embedding model
        overlap_size: Overlap units (tokens for traditional, chars for AST)
        chunking_mode: "traditional" (tokens) or "ast" (characters)
        safety_factor: Safety margin (0.9 = 10% safety margin)

    Returns:
        Safe chunk size: tokens for traditional, characters for AST
    """
    safe_limit = int(model_token_limit * safety_factor)

    if chunking_mode == "traditional":
        # Traditional chunking uses tokens
        # Max chunk = chunk_size + overlap, so chunk_size = limit - overlap
        return max(1, safe_limit - overlap_size)
    else:  # AST chunking
        # AST uses characters, need to convert
        # Conservative estimate: 1.2 tokens per char for code
        overlap_chars = int(overlap_size * 3)  # ~3 chars per token for code
        safe_chars = int(safe_limit / 1.2)
        return max(1, safe_chars - overlap_chars)


def validate_chunk_token_limits(chunks: list[str], max_tokens: int = 512) -> tuple[list[str], int]:
    """
    Validate that chunks don't exceed token limits and truncate if necessary.

    Args:
        chunks: List of text chunks to validate
        max_tokens: Maximum tokens allowed per chunk

    Returns:
        Tuple of (validated_chunks, num_truncated)
    """
    validated_chunks = []
    num_truncated = 0

    for i, chunk in enumerate(chunks):
        estimated_tokens = estimate_token_count(chunk)

        if estimated_tokens > max_tokens:
            # Truncate chunk to fit token limit
            try:
                import tiktoken

                encoder = tiktoken.get_encoding("cl100k_base")
                tokens = encoder.encode(chunk)
                if len(tokens) > max_tokens:
                    truncated_tokens = tokens[:max_tokens]
                    truncated_chunk = encoder.decode(truncated_tokens)
                    validated_chunks.append(truncated_chunk)
                    num_truncated += 1
                    logger.warning(
                        f"Truncated chunk {i} from {len(tokens)} to {max_tokens} tokens "
                        f"(from {len(chunk)} to {len(truncated_chunk)} characters)"
                    )
                else:
                    validated_chunks.append(chunk)
            except ImportError:
                # Fallback: Conservative character truncation
                char_limit = int(max_tokens / 1.2)  # Conservative for code
                if len(chunk) > char_limit:
                    truncated_chunk = chunk[:char_limit]
                    validated_chunks.append(truncated_chunk)
                    num_truncated += 1
                    logger.warning(
                        f"Truncated chunk {i} from {len(chunk)} to {char_limit} characters "
                        f"(conservative estimate for {max_tokens} tokens)"
                    )
                else:
                    validated_chunks.append(chunk)
        else:
            validated_chunks.append(chunk)

    if num_truncated > 0:
        logger.warning(f"Truncated {num_truncated}/{len(chunks)} chunks to fit token limits")

    return validated_chunks, num_truncated


# Code file extensions supported by astchunk
CODE_EXTENSIONS = {
    ".py": "python",
    ".java": "java",
    ".cs": "csharp",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
}


def detect_code_files(documents, code_extensions=None) -> tuple[list, list]:
    """Separate documents into code files and regular text files."""
    if code_extensions is None:
        code_extensions = CODE_EXTENSIONS

    code_docs = []
    text_docs = []

    for doc in documents:
        file_path = doc.metadata.get("file_path", "") or doc.metadata.get("file_name", "")
        if file_path:
            file_ext = Path(file_path).suffix.lower()
            if file_ext in code_extensions:
                doc.metadata["language"] = code_extensions[file_ext]
                doc.metadata["is_code"] = True
                code_docs.append(doc)
            else:
                doc.metadata["is_code"] = False
                text_docs.append(doc)
        else:
            doc.metadata["is_code"] = False
            text_docs.append(doc)

    logger.info(f"Detected {len(code_docs)} code files and {len(text_docs)} text files")
    return code_docs, text_docs


def get_language_from_extension(file_path: str) -> Optional[str]:
    """Return language string from a filename/extension using CODE_EXTENSIONS."""
    ext = Path(file_path).suffix.lower()
    return CODE_EXTENSIONS.get(ext)


def _parse_ast_chunk_output(chunk: Any) -> tuple[str | None, dict[str, Any]]:
    """Normalize the various chunk output formats from ASTChunkBuilder.

    astchunk can return objects (with a ``.text`` attr), plain strings, or
    dicts (``{"content": ..., "metadata": ...}``).  This helper returns a
    uniform ``(text, metadata)`` pair regardless of the input shape.
    """
    if hasattr(chunk, "text"):
        return (str(chunk.text) if chunk.text else None, {})
    if isinstance(chunk, str):
        return (chunk, {})
    if isinstance(chunk, dict):
        meta = chunk.get("metadata", {})
        if "content" in chunk:
            return (chunk["content"], meta)
        if "text" in chunk:
            return (chunk["text"], meta)
        return (str(chunk), {})
    return (str(chunk), {})


def create_ast_chunks(
    documents,
    max_chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata_template: str = "default",
) -> list[dict[str, Any]]:
    """Create AST-aware chunks from code documents using astchunk.

    Falls back to traditional chunking if astchunk is unavailable.

    Returns:
        List of dicts with {"text": str, "metadata": dict}
    """
    try:
        from astchunk import ASTChunkBuilder  # optional dependency
    except ImportError as e:
        logger.error(f"astchunk not available: {e}")
        logger.info("Falling back to traditional chunking for code files")
        return create_traditional_chunks(documents, max_chunk_size, chunk_overlap)

    all_chunks = []
    for doc in documents:
        language = doc.metadata.get("language")
        if not language:
            logger.warning("No language detected; falling back to traditional chunking")
            all_chunks.extend(create_traditional_chunks([doc], max_chunk_size, chunk_overlap))
            continue

        try:
            # Warn once if AST chunk size + overlap might exceed common token limits
            # Note: Actual truncation happens at embedding time with dynamic model limits
            global _ast_token_warning_shown
            estimated_max_tokens = int(
                (max_chunk_size + chunk_overlap) * 1.2
            )  # Conservative estimate
            if estimated_max_tokens > 512 and not _ast_token_warning_shown:
                logger.warning(
                    f"AST chunk size ({max_chunk_size}) + overlap ({chunk_overlap}) = {max_chunk_size + chunk_overlap} chars "
                    f"may exceed 512 token limit (~{estimated_max_tokens} tokens estimated). "
                    f"Consider reducing --ast-chunk-size to {int(400 / 1.2)} or --ast-chunk-overlap to {int(50 / 1.2)}. "
                    f"Note: Chunks will be auto-truncated at embedding time based on your model's actual token limit."
                )
                _ast_token_warning_shown = True

            configs = {
                "max_chunk_size": max_chunk_size,
                "language": language,
                "metadata_template": metadata_template,
                "chunk_overlap": chunk_overlap if chunk_overlap > 0 else 0,
            }

            repo_metadata = {
                "file_path": doc.metadata.get("file_path", ""),
                "file_name": doc.metadata.get("file_name", ""),
                "source": doc.metadata.get("source", ""),
                "creation_date": doc.metadata.get("creation_date", ""),
                "last_modified_date": doc.metadata.get("last_modified_date", ""),
            }
            configs["repo_level_metadata"] = repo_metadata

            chunk_builder = ASTChunkBuilder(**configs)
            code_content = doc.get_content()
            if not code_content or not code_content.strip():
                logger.warning("Empty code content, skipping")
                continue

            chunks = chunk_builder.chunkify(code_content)
            for chunk in chunks:
                chunk_text, astchunk_metadata = _parse_ast_chunk_output(chunk)

                if chunk_text and chunk_text.strip():
                    # Extract document-level metadata
                    doc_metadata = {
                        "file_path": doc.metadata.get("file_path", ""),
                        "file_name": doc.metadata.get("file_name", ""),
                        "source": doc.metadata.get("source", ""),
                    }
                    if "creation_date" in doc.metadata:
                        doc_metadata["creation_date"] = doc.metadata["creation_date"]
                    if "last_modified_date" in doc.metadata:
                        doc_metadata["last_modified_date"] = doc.metadata["last_modified_date"]

                    # Merge document metadata + astchunk metadata
                    combined_metadata = {**doc_metadata, **astchunk_metadata}

                    # Trim partial first line left by overlap if using line numbers
                    # (a valid line starts with digits followed by '|')
                    stripped = chunk_text.strip()
                    if stripped:
                        first_line = stripped.split("\n", 1)[0]
                        if "|" in first_line and not re.match(r"^\s*\d+\|", first_line):
                            first_nl = stripped.find("\n")
                            if first_nl != -1:
                                stripped = stripped[first_nl + 1 :]
                    all_chunks.append({"text": stripped, "metadata": combined_metadata})

            logger.info(
                f"Created {len(chunks)} AST chunks from {language} file: {doc.metadata.get('file_name', 'unknown')}"
            )
        except Exception as e:
            logger.warning(f"AST chunking failed for {language} file: {e}")
            logger.info("Falling back to traditional chunking")
            all_chunks.extend(create_traditional_chunks([doc], max_chunk_size, chunk_overlap))

    return all_chunks


def create_traditional_chunks(
    documents,
    chunk_size: int = 256,
    chunk_overlap: int = 128,
    max_tokens_per_chunk: int | None = None,
) -> list[dict[str, Any]]:
    """Create traditional text chunks using LlamaIndex SentenceSplitter.

    Args:
        documents: LlamaIndex Document list.
        chunk_size: Target chunk size in **characters** (approximate tokens).
        chunk_overlap: Overlap between adjacent chunks in characters.
        max_tokens_per_chunk: If set, auto-scale ``chunk_size`` so each
            chunk stays within the model's token budget.  Uses
            ``calculate_safe_chunk_size`` with a 10 % safety margin.
            Additionaly runs ``validate_chunk_token_limits`` post-chunk.

    Returns:
        List of dicts with ``{"text": str, "metadata": dict}``.
    """
    if chunk_size <= 0:
        logger.warning(f"Invalid chunk_size={chunk_size}, using default value of 256")
        chunk_size = 256
    if chunk_overlap < 0:
        chunk_overlap = 0

    # ── Token-aware auto-scaling ──────────────────────────────────────
    if max_tokens_per_chunk and max_tokens_per_chunk > 0:
        chunk_size = calculate_safe_chunk_size(
            max_tokens_per_chunk, chunk_overlap, chunking_mode="traditional"
        )
        logger.info(
            "Token-aware chunking: model limit=%d tokens → safe chunk_size=%d chars "
            "(overlap=%d, safety=0.9)",
            max_tokens_per_chunk,
            chunk_size,
            chunk_overlap,
        )

    # Revalidate after scaling
    if chunk_overlap >= chunk_size:
        old_overlap = chunk_overlap
        chunk_overlap = max(0, chunk_size // 2)
        logger.warning(
            "Token-aware scaling reduced chunk_size below chunk_overlap "
            "(%d → %d); overlap reduced %d → %d",
            chunk_size,
            old_overlap,
            old_overlap,
            chunk_overlap,
        )

    node_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" ",
        paragraph_separator="\n\n",
    )

    result = []
    for doc in documents:
        doc_metadata = dict(doc.metadata) if doc.metadata else {}
        doc_metadata.setdefault("file_path", "")
        doc_metadata.setdefault("file_name", "")
        doc_metadata.setdefault("source", "")

        try:
            nodes = node_parser.get_nodes_from_documents([doc])
            if nodes:
                for node in nodes:
                    result.append({"text": node.get_content(), "metadata": doc_metadata})
        except Exception as e:
            logger.error(f"Traditional chunking failed for document: {e}")
            content = doc.get_content()
            if content and content.strip():
                result.append({"text": content.strip(), "metadata": doc_metadata})

    # ── Post-chunk validation ─────────────────────────────────────────
    if max_tokens_per_chunk and max_tokens_per_chunk > 0 and result:
        _texts = [c["text"] for c in result]
        _validated, _n = validate_chunk_token_limits(_texts, max_tokens_per_chunk)
        if _n > 0:
            logger.warning(
                "Token-aware chunking: %d/%d chunks truncated to fit %d-token limit",
                _n,
                len(result),
                max_tokens_per_chunk,
            )
        for i, chunk_dict in enumerate(result):
            if i < len(_validated):
                chunk_dict["text"] = _validated[i]

    return result


def _traditional_chunks_as_dicts(
    documents,
    chunk_size: int = 256,
    chunk_overlap: int = 128,
    max_tokens_per_chunk: int | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible alias for create_traditional_chunks."""
    return create_traditional_chunks(
        documents,
        chunk_size,
        chunk_overlap,
        max_tokens_per_chunk=max_tokens_per_chunk,
    )


def create_text_chunks(
    documents,
    chunk_size: int = 256,
    chunk_overlap: int = 128,
    use_ast_chunking: bool = False,
    ast_chunk_size: int = 512,
    ast_chunk_overlap: int = 64,
    code_file_extensions: Optional[list[str]] = None,
    ast_fallback_traditional: bool = True,
    max_tokens_per_chunk: int | None = None,
) -> list[dict[str, Any]]:
    """Create text chunks from documents with optional AST support for code files.

    Args:
        documents: LlamaIndex Document list.
        chunk_size: Characters per traditional chunk.
        chunk_overlap: Character overlap between traditional chunks.
        max_tokens_per_chunk: If set, auto-scale chunk_size to keep each
            chunk within the embedding model's token budget.  Also runs
            post-chunk validation and truncation when necessary.

    Returns:
        List of dicts with ``{"text": str, "metadata": dict}``.
    """
    if not documents:
        logger.warning("No documents provided for chunking")
        return []

    local_code_extensions = CODE_EXTENSIONS.copy()
    if code_file_extensions:
        ext_mapping = {
            ".py": "python",
            ".java": "java",
            ".cs": "c_sharp",
            ".ts": "typescript",
            ".tsx": "typescript",
        }
        for ext in code_file_extensions:
            if ext.lower() not in local_code_extensions:
                if ext.lower() in ext_mapping:
                    local_code_extensions[ext.lower()] = ext_mapping[ext.lower()]
                else:
                    logger.warning(f"Unsupported extension {ext}, will use traditional chunking")

    all_chunks = []
    if use_ast_chunking:
        code_docs, text_docs = detect_code_files(documents, local_code_extensions)
        if code_docs:
            try:
                # AST chunking: auto-scale if token limit given
                ast_size = ast_chunk_size
                if max_tokens_per_chunk and max_tokens_per_chunk > 0:
                    ast_size = calculate_safe_chunk_size(
                        max_tokens_per_chunk, ast_chunk_overlap, chunking_mode="ast"
                    )
                    logger.info(
                        "Token-aware AST chunking: limit=%d → safe ast_chunk_size=%d chars "
                        "(overlap=%d, safety=0.9)",
                        max_tokens_per_chunk,
                        ast_size,
                        ast_chunk_overlap,
                    )

                ast_chunks = create_ast_chunks(
                    code_docs, max_chunk_size=ast_size, chunk_overlap=ast_chunk_overlap
                )
                # Prepend line numbers to code chunks for navigation
                for chunk in ast_chunks:
                    start_line = chunk.get("metadata", {}).get("start_line_no")
                    if start_line is not None:
                        lines = chunk["text"].split("\n")
                        end_line = start_line + len(lines) - 1
                        w = len(str(end_line))
                        chunk["text"] = "\n".join(
                            f"{start_line + i:>{w}}|{line}" for i, line in enumerate(lines)
                        )
                all_chunks.extend(ast_chunks)
            except Exception as e:
                logger.error(f"AST chunking failed: {e}")
                if ast_fallback_traditional:
                    all_chunks.extend(
                        create_traditional_chunks(
                            code_docs,
                            chunk_size,
                            chunk_overlap,
                            max_tokens_per_chunk=max_tokens_per_chunk,
                        )
                    )
                else:
                    raise
        if text_docs:
            all_chunks.extend(
                create_traditional_chunks(
                    text_docs,
                    chunk_size,
                    chunk_overlap,
                    max_tokens_per_chunk=max_tokens_per_chunk,
                )
            )
    else:
        all_chunks = create_traditional_chunks(
            documents,
            chunk_size,
            chunk_overlap,
            max_tokens_per_chunk=max_tokens_per_chunk,
        )

    logger.info(f"Total chunks created: {len(all_chunks)}")

    return all_chunks
