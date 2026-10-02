"""Runtime configuration helpers for LEANN."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Default fallbacks to preserve current behaviour while keeping them in one place.
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_DEFAULT_MINIMAX_BASE_URL = "https://api.minimax.io/v1"
_DEFAULT_NOVITA_BASE_URL = "https://api.novita.ai/openai"
_DEFAULT_ATLASCLOUD_BASE_URL = "https://api.atlascloud.ai/v1"
_DEFAULT_CHEAPERINFERENCE_BASE_URL = "https://api.cheaperinference.com/v1"


def _clean_url(value: str) -> str:
    """Normalize URL strings by stripping trailing slashes."""

    return value.rstrip("/") if value else value


def resolve_ollama_host(explicit: str | None = None) -> str:
    """Resolve the Ollama-compatible endpoint to use."""

    candidates = (
        explicit,
        os.getenv("LEANN_LOCAL_LLM_HOST"),
        os.getenv("LEANN_OLLAMA_HOST"),
        os.getenv("OLLAMA_HOST"),
        os.getenv("LOCAL_LLM_ENDPOINT"),
    )

    for candidate in candidates:
        if candidate:
            return _clean_url(candidate)

    return _clean_url(_DEFAULT_OLLAMA_HOST)


def resolve_openai_base_url(explicit: str | None = None) -> str:
    """Resolve the base URL for OpenAI-compatible services."""

    candidates = (
        explicit,
        os.getenv("LEANN_OPENAI_BASE_URL"),
        os.getenv("OPENAI_BASE_URL"),
        os.getenv("LOCAL_OPENAI_BASE_URL"),
    )

    for candidate in candidates:
        if candidate:
            return _clean_url(candidate)

    return _clean_url(_DEFAULT_OPENAI_BASE_URL)


def resolve_anthropic_base_url(explicit: str | None = None) -> str:
    """Resolve the base URL for Anthropic-compatible services."""

    candidates = (
        explicit,
        os.getenv("LEANN_ANTHROPIC_BASE_URL"),
        os.getenv("ANTHROPIC_BASE_URL"),
        os.getenv("LOCAL_ANTHROPIC_BASE_URL"),
    )

    for candidate in candidates:
        if candidate:
            return _clean_url(candidate)

    return _clean_url(_DEFAULT_ANTHROPIC_BASE_URL)


def resolve_openai_api_key(explicit: str | None = None) -> str | None:
    """Resolve the API key for OpenAI-compatible services."""

    if explicit:
        return explicit

    return os.getenv("OPENAI_API_KEY")


def resolve_anthropic_api_key(explicit: str | None = None) -> str | None:
    """Resolve the API key for Anthropic services."""

    if explicit:
        return explicit

    return os.getenv("ANTHROPIC_API_KEY")


def resolve_litellm_base_url(explicit: str | None = None) -> str | None:
    """Resolve an optional base URL for LiteLLM.

    Unlike the single-provider resolvers there is no default: when unset,
    LiteLLM talks directly to the provider inferred from the model string
    (e.g. ``anthropic/claude-...``). Set a base URL only to route through a
    self-hosted LiteLLM proxy.
    """

    candidates = (
        explicit,
        os.getenv("LEANN_LITELLM_BASE_URL"),
        os.getenv("LITELLM_BASE_URL"),
        os.getenv("LITELLM_API_BASE"),
    )

    for candidate in candidates:
        if candidate:
            return _clean_url(candidate)

    return None


def resolve_litellm_api_key(explicit: str | None = None) -> str | None:
    """Resolve an optional API key for LiteLLM.

    When unset, LiteLLM reads the underlying provider's own environment
    variable (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ...). Set this only
    when talking to a LiteLLM proxy that expects a virtual key.
    """

    if explicit:
        return explicit

    return os.getenv("LITELLM_API_KEY") or os.getenv("LEANN_LITELLM_API_KEY")


def resolve_minimax_base_url(explicit: str | None = None) -> str:
    """Resolve the base URL for MiniMax-compatible services."""

    candidates = (
        explicit,
        os.getenv("LEANN_MINIMAX_BASE_URL"),
        os.getenv("MINIMAX_BASE_URL"),
    )

    for candidate in candidates:
        if candidate:
            return _clean_url(candidate)

    return _clean_url(_DEFAULT_MINIMAX_BASE_URL)


def resolve_minimax_api_key(explicit: str | None = None) -> str | None:
    """Resolve the API key for MiniMax services."""

    if explicit:
        return explicit

    return os.getenv("MINIMAX_API_KEY")


def resolve_novita_base_url(explicit: str | None = None) -> str:
    """Resolve the base URL for Novita AI services."""

    candidates = (
        explicit,
        os.getenv("LEANN_NOVITA_BASE_URL"),
        os.getenv("NOVITA_BASE_URL"),
        os.getenv("OPENAI_BASE_URL"),  # Fallback to OpenAI base URL
    )

    for candidate in candidates:
        if candidate:
            return _clean_url(candidate)

    return _clean_url(_DEFAULT_NOVITA_BASE_URL)


def resolve_novita_api_key(explicit: str | None = None) -> str | None:
    """Resolve the API key for Novita AI services."""

    if explicit:
        return explicit

    novita_key = os.getenv("NOVITA_API_KEY")
    if novita_key:
        return novita_key

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        logger.warning(
            "NOVITA_API_KEY not set, falling back to OPENAI_API_KEY. "
            "This may cause authentication issues if the OpenAI key is not valid for Novita AI."
        )
    return openai_key


def resolve_atlascloud_base_url(explicit: str | None = None) -> str:
    """Resolve the base URL for Atlas Cloud OpenAI-compatible services."""

    candidates = (
        explicit,
        os.getenv("LEANN_ATLASCLOUD_BASE_URL"),
        os.getenv("LEANN_ATLAS_CLOUD_BASE_URL"),
        os.getenv("ATLASCLOUD_BASE_URL"),
        os.getenv("ATLAS_CLOUD_BASE_URL"),
        os.getenv("ATLASCLOUD_API_BASE"),
        os.getenv("ATLAS_CLOUD_API_BASE"),
    )

    for candidate in candidates:
        if candidate:
            return _clean_url(candidate)

    return _clean_url(_DEFAULT_ATLASCLOUD_BASE_URL)


def resolve_atlascloud_api_key(explicit: str | None = None) -> str | None:
    """Resolve the API key for Atlas Cloud services."""

    if explicit:
        return explicit

    return os.getenv("ATLASCLOUD_API_KEY") or os.getenv("ATLAS_CLOUD_API_KEY")


def resolve_cheaperinference_base_url(explicit: str | None = None) -> str:
    """Resolve the base URL for Cheaper Inference OpenAI-compatible services."""

    candidates = (
        explicit,
        os.getenv("LEANN_CHEAPER_INFERENCE_BASE_URL"),
        os.getenv("CHEAPER_INFERENCE_BASE_URL"),
    )

    for candidate in candidates:
        if candidate:
            return _clean_url(candidate)

    return _clean_url(_DEFAULT_CHEAPERINFERENCE_BASE_URL)


def resolve_cheaperinference_api_key(explicit: str | None = None) -> str | None:
    """Resolve the API key for Cheaper Inference services."""

    if explicit:
        return explicit

    return os.getenv("CHEAPER_INFERENCE_API_KEY")


def encode_provider_options(options: dict[str, Any] | None) -> str | None:
    """Serialize provider options for child processes."""

    if not options:
        return None

    try:
        return json.dumps(options)
    except (TypeError, ValueError):
        # Fall back to empty payload if serialization fails
        return None
