"""
Test document_rag functionality using pytest.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir():
    """Return the path to test data directory."""
    return Path("data")


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Skip model tests in CI to avoid MPS memory issues"
)
def test_document_rag_simulated(test_data_dir):
    """Test document_rag with simulated LLM."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        # Use a subdirectory that doesn't exist yet to force index creation
        index_dir = Path(temp_dir) / "test_index"
        cmd = [
            sys.executable,
            "apps/document_rag.py",
            "--llm",
            "simulated",
            "--embedding-model",
            "facebook/contriever",
            "--embedding-mode",
            "sentence-transformers",
            "--index-dir",
            str(index_dir),
            "--data-dir",
            str(test_data_dir),
            "--query",
            "What is Pride and Prejudice about?",
        ]

        env = os.environ.copy()
        env["HF_HUB_DISABLE_SYMLINKS"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", timeout=600, env=env)

        # Check return code
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify output
        output = result.stdout + result.stderr
        assert "Index saved to" in output or "Using existing index" in output
        assert "This is a simulated answer" in output


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Skip AST chunking tests in CI to avoid dependency issues",
)
def test_document_rag_with_ast_chunking(test_data_dir):
    """Test document_rag with AST-aware chunking enabled."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        # Use a subdirectory that doesn't exist yet to force index creation
        index_dir = Path(temp_dir) / "test_ast_index"
        cmd = [
            sys.executable,
            "apps/document_rag.py",
            "--llm",
            "simulated",
            "--embedding-model",
            "facebook/contriever",
            "--embedding-mode",
            "sentence-transformers",
            "--index-dir",
            str(index_dir),
            "--data-dir",
            str(test_data_dir),
            "--enable-code-chunking",  # Enable AST chunking
            "--query",
            "What is Pride and Prejudice about?",
        ]

        env = os.environ.copy()
        env["HF_HUB_DISABLE_SYMLINKS"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", timeout=600, env=env)

        # Check return code
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify output
        output = result.stdout + result.stderr
        assert "Index saved to" in output or "Using existing index" in output
        assert "This is a simulated answer" in output

        # Should mention AST chunking if code files are present
        # (might not be relevant for the test data, but command should succeed)


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OpenAI API key not available")
@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Skip OpenAI tests in CI to avoid API costs"
)
def test_document_rag_openai(test_data_dir):
    """Test document_rag with OpenAI embeddings."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        # Use a subdirectory that doesn't exist yet to force index creation
        index_dir = Path(temp_dir) / "test_index_openai"
        cmd = [
            sys.executable,
            "apps/document_rag.py",
            "--llm",
            "simulated",  # Use simulated LLM to avoid GPT-4 costs
            "--embedding-model",
            "text-embedding-3-small",
            "--embedding-mode",
            "openai",
            "--index-dir",
            str(index_dir),
            "--data-dir",
            str(test_data_dir),
            "--query",
            "What is Pride and Prejudice about?",
        ]

        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", timeout=600, env=env)

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify cosine distance was used
        output = result.stdout + result.stderr
        assert any(
            msg in output
            for msg in [
                "distance_metric='cosine'",
                "Automatically setting distance_metric='cosine'",
                "Using cosine distance",
            ]
        )


def test_document_rag_error_handling(test_data_dir):
    """Test document_rag with invalid parameters."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        cmd = [
            sys.executable,
            "apps/document_rag.py",
            "--llm",
            "invalid_llm_type",
            "--index-dir",
            temp_dir,
            "--data-dir",
            str(test_data_dir),
        ]

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", timeout=60, env=env)

        # Should fail with invalid LLM type
        assert result.returncode != 0
        assert "invalid choice" in result.stderr or "invalid_llm_type" in result.stderr
