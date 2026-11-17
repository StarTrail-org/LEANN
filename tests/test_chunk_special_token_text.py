"""Special-token spellings in document text are ordinary corpus content."""

import pytest
import tiktoken
from leann.chunking_utils import estimate_token_count, validate_chunk_token_limits


@pytest.fixture(autouse=True)
def byte_encoder(monkeypatch):
    # A real, deterministic tiktoken encoder keeps this regression offline.
    encoder = tiktoken.Encoding(
        name="literal-text-test",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={"<|endoftext|>": 256},
    )
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: encoder)


def test_estimate_literal_special_token():
    text = "A tokenizer example: <|endoftext|> is a literal token spelling."
    expected = len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))
    assert estimate_token_count(text) == expected


def test_truncate_literal_special_token():
    text = "<|endoftext|> " * 20
    chunks, truncated = validate_chunk_token_limits([text], max_tokens=16)
    assert truncated == 1
    assert len(chunks) == 1
    assert estimate_token_count(chunks[0]) <= 16


def test_short_literal_special_token_is_preserved():
    text = "<|endoftext|>"
    assert validate_chunk_token_limits([text], max_tokens=32) == ([text], 0)
