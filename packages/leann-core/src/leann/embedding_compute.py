"""
Unified embedding computation module
Consolidates all embedding computation logic using SentenceTransformer
Preserves all optimization parameters to ensure performance
"""

import json
import logging
import os
import subprocess
import time
from typing import Any, Optional, Protocol, cast

import numpy as np

from .settings import resolve_ollama_host, resolve_openai_api_key, resolve_openai_base_url

# torch and tiktoken are imported lazily inside the functions that use them, so
# `import leann` (e.g. for MCP search over an existing index, BM25-only flows,
# or non-embedding utilities) doesn't pull torch's ~1 GB of state into memory.

# Set up logger with proper level
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LEANN_LOG_LEVEL", "WARNING").upper()
log_level = getattr(logging, LOG_LEVEL, logging.WARNING)
logger.setLevel(log_level)


class _SentenceTransformerLike(Protocol):
    def eval(self) -> Any: ...
    def parameters(self) -> Any: ...
    def encode(self, *args: Any, **kwargs: Any) -> Any: ...
    def half(self) -> Any: ...


# Token limit registry for embedding models
# Used as fallback when dynamic discovery fails (e.g., LM Studio, OpenAI)
# Ollama models use dynamic discovery via /api/show
EMBEDDING_MODEL_LIMITS = {
    # Nomic models (common across servers)
    "nomic-embed-text": 2048,  # Corrected from 512 - verified via /api/show
    "nomic-embed-text-v1.5": 2048,
    "nomic-embed-text-v2": 512,
    # Other embedding models
    "mxbai-embed-large": 512,
    "all-minilm": 512,
    "bge-m3": 8192,
    "snowflake-arctic-embed": 512,
    # OpenAI models
    "text-embedding-3-small": 8192,
    "text-embedding-3-large": 8192,
    "text-embedding-ada-002": 8192,
}

# Runtime cache for dynamically discovered token limits
# Key: (model_name, base_url), Value: token_limit
# Prevents repeated SDK/API calls for the same model
_token_limit_cache: dict[tuple[str, str], int] = {}


def get_model_token_limit(
    model_name: str,
    base_url: Optional[str] = None,
    default: int = 2048,
) -> int:
    """
    Get token limit for a given embedding model.
    Uses hybrid approach: dynamic discovery for Ollama, registry fallback for others.
    Caches discovered limits to prevent repeated API/SDK calls.

    Args:
        model_name: Name of the embedding model
        base_url: Base URL of the embedding server (for dynamic discovery)
        default: Default token limit if model not found

    Returns:
        Token limit for the model in tokens
    """
    # Check cache first to avoid repeated SDK/API calls
    cache_key = (model_name, base_url or "")
    if cache_key in _token_limit_cache:
        cached_limit = _token_limit_cache[cache_key]
        logger.debug(f"Using cached token limit for {model_name}: {cached_limit}")
        return cached_limit

    # Try Ollama dynamic discovery if base_url provided
    if base_url:
        # Detect Ollama servers by port or "ollama" in URL
        if "11434" in base_url or "ollama" in base_url.lower():
            limit = _query_ollama_context_limit(model_name, base_url)
            if limit:
                _token_limit_cache[cache_key] = limit
                return limit

        # Try LM Studio SDK discovery
        if "1234" in base_url or "lmstudio" in base_url.lower() or "lm.studio" in base_url.lower():
            # Convert HTTP to WebSocket URL
            ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
            # Remove /v1 suffix if present
            if ws_url.endswith("/v1"):
                ws_url = ws_url[:-3]

            limit = _query_lmstudio_context_limit(model_name, ws_url)
            if limit:
                _token_limit_cache[cache_key] = limit
                return limit

    # Fallback to known model registry with version handling (from PR #154)
    # Handle versioned model names (e.g., "nomic-embed-text:latest" -> "nomic-embed-text")
    base_model_name = model_name.split(":")[0]

    # Check exact match first
    if model_name in EMBEDDING_MODEL_LIMITS:
        limit = EMBEDDING_MODEL_LIMITS[model_name]
        _token_limit_cache[cache_key] = limit
        return limit

    # Check base name match
    if base_model_name in EMBEDDING_MODEL_LIMITS:
        limit = EMBEDDING_MODEL_LIMITS[base_model_name]
        _token_limit_cache[cache_key] = limit
        return limit

    # Local Hugging Face models (sentence-transformers) publish their sequence limit
    if not base_url and "/" in model_name:
        limit = _query_huggingface_seq_limit(model_name)
        if limit:
            _token_limit_cache[cache_key] = limit
            return limit

    # Check partial matches for common patterns
    for known_model, registry_limit in EMBEDDING_MODEL_LIMITS.items():
        if known_model in base_model_name or base_model_name in known_model:
            _token_limit_cache[cache_key] = registry_limit
            return registry_limit

    # Default fallback
    logger.warning(f"Unknown model '{model_name}', using default {default} token limit")
    _token_limit_cache[cache_key] = default
    return default


def truncate_to_token_limit(texts: list[str], token_limit: int) -> list[str]:
    """
    Truncate texts to fit within token limit using tiktoken.

    Args:
        texts: List of text strings to truncate
        token_limit: Maximum number of tokens allowed

    Returns:
        List of truncated texts (same length as input)
    """
    if not texts:
        return []

    import tiktoken

    # Use tiktoken with cl100k_base encoding
    enc = tiktoken.get_encoding("cl100k_base")

    truncated_texts = []
    truncation_count = 0
    total_tokens_removed = 0
    max_original_length = 0

    for i, text in enumerate(texts):
        tokens = enc.encode(text)
        original_length = len(tokens)

        if original_length <= token_limit:
            # Text is within limit, keep as is
            truncated_texts.append(text)
        else:
            # Truncate to token_limit
            truncated_tokens = tokens[:token_limit]
            truncated_text = enc.decode(truncated_tokens)
            truncated_texts.append(truncated_text)

            # Track truncation statistics
            truncation_count += 1
            tokens_removed = original_length - token_limit
            total_tokens_removed += tokens_removed
            max_original_length = max(max_original_length, original_length)

            # Log individual truncation at WARNING level (first few only)
            if truncation_count <= 3:
                logger.warning(
                    f"Text {i + 1} truncated: {original_length} → {token_limit} tokens "
                    f"({tokens_removed} tokens removed)"
                )
            elif truncation_count == 4:
                logger.warning("Further truncation warnings suppressed...")

    # Log summary at INFO level
    if truncation_count > 0:
        logger.warning(
            f"Truncation summary: {truncation_count}/{len(texts)} texts truncated "
            f"(removed {total_tokens_removed} tokens total, longest was {max_original_length} tokens)"
        )
    else:
        logger.debug(
            f"No truncation needed - all {len(texts)} texts within {token_limit} token limit"
        )

    return truncated_texts


def _query_huggingface_seq_limit(model_name: str) -> Optional[int]:
    """
    Read ``max_seq_length`` from a Hugging Face sentence-transformers repo.

    Args:
        model_name: Hugging Face repo id, e.g. ``sentence-transformers/all-MiniLM-L6-v2``

    Returns:
        Sequence limit in tokens if published, None otherwise
    """
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_name, "sentence_bert_config.json")
        with open(config_path, encoding="utf-8") as f:
            limit = json.load(f).get("max_seq_length")
        if isinstance(limit, int) and limit > 0:
            logger.info(f"Detected {model_name} sequence limit: {limit} tokens")
            return limit
    except Exception as e:
        logger.debug(f"Failed to read Hugging Face sequence limit: {e}")

    return None


def _query_ollama_context_limit(model_name: str, base_url: str) -> Optional[int]:
    """
    Query Ollama /api/show for model context limit.

    Args:
        model_name: Name of the Ollama model
        base_url: Base URL of the Ollama server

    Returns:
        Context limit in tokens if found, None otherwise
    """
    try:
        import requests

        response = requests.post(
            f"{base_url}/api/show",
            json={"name": model_name},
            timeout=5,
        )
        if response.status_code == 200:
            data = response.json()
            if "model_info" in data:
                # Look for *.context_length in model_info
                for key, value in data["model_info"].items():
                    if "context_length" in key and isinstance(value, int):
                        logger.info(f"Detected {model_name} context limit: {value} tokens")
                        return value
    except Exception as e:
        logger.debug(f"Failed to query Ollama context limit: {e}")

    return None


def _query_lmstudio_context_limit(model_name: str, base_url: str) -> Optional[int]:
    """
    Query LM Studio SDK for model context length via Node.js subprocess.

    Args:
        model_name: Name of the LM Studio model
        base_url: Base URL of the LM Studio server (WebSocket format, e.g., "ws://localhost:1234")

    Returns:
        Context limit in tokens if found, None otherwise
    """
    # Inline JavaScript using @lmstudio/sdk
    # Note: Load model temporarily for metadata, then unload to respect JIT auto-evict
    js_code = f"""
    const {{ LMStudioClient }} = require('@lmstudio/sdk');
    (async () => {{
        try {{
            const client = new LMStudioClient({{ baseUrl: '{base_url}' }});
            const model = await client.embedding.load('{model_name}', {{ verbose: false }});
            const contextLength = await model.getContextLength();
            await model.unload();  // Unload immediately to respect JIT auto-evict settings
            console.log(JSON.stringify({{ contextLength, identifier: '{model_name}' }}));
        }} catch (error) {{
            console.error(JSON.stringify({{ error: error.message }}));
            process.exit(1);
        }}
    }})();
    """

    try:
        # Set NODE_PATH to include global modules for @lmstudio/sdk resolution
        env = os.environ.copy()

        # Try to get npm global root (works with nvm, brew node, etc.)
        try:
            npm_root = subprocess.run(
                ["npm", "root", "-g"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if npm_root.returncode == 0:
                global_modules = npm_root.stdout.strip()
                # Append to existing NODE_PATH if present
                existing_node_path = env.get("NODE_PATH", "")
                env["NODE_PATH"] = (
                    f"{global_modules}:{existing_node_path}"
                    if existing_node_path
                    else global_modules
                )
        except Exception:
            # If npm not available, continue with existing NODE_PATH
            pass

        result = subprocess.run(
            ["node", "-e", js_code],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )

        if result.returncode != 0:
            logger.debug(f"LM Studio SDK error: {result.stderr}")
            return None

        data = json.loads(result.stdout)
        context_length = data.get("contextLength")

        if context_length and context_length > 0:
            logger.info(f"LM Studio SDK detected {model_name} context length: {context_length}")
            return context_length

    except FileNotFoundError:
        logger.debug("Node.js not found - install Node.js for LM Studio SDK features")
    except subprocess.TimeoutExpired:
        logger.debug("LM Studio SDK query timeout")
    except json.JSONDecodeError:
        logger.debug("LM Studio SDK returned invalid JSON")
    except Exception as e:
        logger.debug(f"LM Studio SDK query failed: {e}")

    return None


# Global model cache to avoid repeated loading
_model_cache: dict[str, Any] = {}

_DEFAULT_CUDA_BATCH_SIZE = 256
_DEFAULT_MPS_BATCH_SIZE = 128


def _parse_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default
    if value < 1:
        logger.warning("%s must be >= 1; using default %d", name, default)
        return default
    return value


def _resolve_cpu_thread_count() -> int:
    return _parse_positive_int_env("LEANN_CPU_THREADS", min(8, os.cpu_count() or 4))


def _resolve_adaptive_batch_size(device: str, model_name: str) -> int:
    if device == "cuda":
        return _parse_positive_int_env("LEANN_CUDA_BATCH_SIZE", _DEFAULT_CUDA_BATCH_SIZE)
    if device == "mps":
        default = 32 if model_name == "Qwen/Qwen3-Embedding-0.6B" else _DEFAULT_MPS_BATCH_SIZE
        return _parse_positive_int_env("LEANN_MPS_BATCH_SIZE", default)
    return 32


def _cap_cuda_batch_by_vram(requested: int, max_length: int = 512) -> int:
    auto = os.getenv("LEANN_CUDA_AUTO_BATCH", "1").lower()
    if auto in ("0", "false", "no"):
        return requested
    import torch

    if not torch.cuda.is_available():
        return requested
    try:
        free_bytes, _total = torch.cuda.mem_get_info()
    except Exception:
        return requested

    # Eager-attention peak memory scales ~O(seq^2) per sequence in the batch.
    bytes_per_seq = max(8_000_000, max_length * max_length * 32)
    budget = int(free_bytes * 0.2)
    max_by_vram = max(1, budget // bytes_per_seq)
    capped = min(requested, max_by_vram)
    if capped < requested:
        logger.info(
            "Capping CUDA embedding batch size %d -> %d (%.2f GiB free VRAM)",
            requested,
            capped,
            free_bytes / (1024**3),
        )
    return capped


def _encode_with_oom_retry(
    model: _SentenceTransformerLike,
    texts: list[str],
    batch_size: int,
    *,
    is_build: bool,
    device: str,
) -> Any:
    import torch

    current = batch_size
    while True:
        try:
            with torch.inference_mode():
                return model.encode(
                    texts,
                    batch_size=current,
                    show_progress_bar=is_build,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    device=device,
                )
        except RuntimeError as exc:
            oom = "out of memory" in str(exc).lower()
            if torch.cuda.is_available():
                oom = oom or isinstance(exc, torch.cuda.OutOfMemoryError)
            if not oom or current <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            next_batch = max(1, current // 2)
            logger.warning(
                "CUDA OOM at embedding batch_size=%d; retrying with batch_size=%d",
                current,
                next_batch,
            )
            current = next_batch


def compute_embeddings(
    texts: list[str],
    model_name: str,
    mode: str = "sentence-transformers",
    is_build: bool = False,
    batch_size: int = 32,
    adaptive_optimization: bool = True,
    manual_tokenize: bool = False,
    max_length: int = 512,
    provider_options: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """
    Unified embedding computation entry point

    Args:
        texts: List of texts to compute embeddings for
        model_name: Model name
        mode: Computation mode ('sentence-transformers', 'openai', 'mlx', 'ollama')
        is_build: Whether this is a build operation (shows progress bar)
        batch_size: Batch size for processing
        adaptive_optimization: Whether to use adaptive optimization based on batch size

    Returns:
        Normalized embeddings array, shape: (len(texts), embedding_dim)
    """
    provider_options = provider_options or {}
    wrapper_start_time = time.time()
    logger.debug(
        f"[compute_embeddings] entry: mode={mode}, model='{model_name}', text_count={len(texts)}"
    )

    # Allow batch_size override from provider_options (disables adaptive_optimization)
    if "batch_size" in provider_options:
        batch_size = provider_options["batch_size"]
        adaptive_optimization = False  # User-specified batch_size takes precedence

    if mode == "sentence-transformers":
        inner_start_time = time.time()
        result = compute_embeddings_sentence_transformers(
            texts,
            model_name,
            is_build=is_build,
            batch_size=batch_size,
            adaptive_optimization=adaptive_optimization,
            manual_tokenize=manual_tokenize,
            max_length=max_length,
        )
        inner_end_time = time.time()
        wrapper_end_time = time.time()
        logger.debug(
            "[compute_embeddings] sentence-transformers timings: "
            f"inner={inner_end_time - inner_start_time:.6f}s, "
            f"wrapper_total={wrapper_end_time - wrapper_start_time:.6f}s"
        )
        return result
    elif mode == "openai":
        return compute_embeddings_openai(
            texts,
            model_name,
            base_url=provider_options.get("base_url"),
            api_key=provider_options.get("api_key"),
            provider_options=provider_options,
            is_build=is_build,
        )
    elif mode == "mlx":
        return compute_embeddings_mlx(texts, model_name, is_build=is_build)
    elif mode == "ollama":
        return compute_embeddings_ollama(
            texts,
            model_name,
            is_build=is_build,
            host=provider_options.get("host"),
            provider_options=provider_options,
        )
    elif mode == "gemini":
        return compute_embeddings_gemini(texts, model_name, is_build=is_build)
    else:
        raise ValueError(f"Unsupported embedding mode: {mode}")


def _resolve_trust_remote_code() -> bool:
    """Whether to execute model-repository code when loading an embedding model.

    Off by default: trust_remote_code runs arbitrary Python from the model repo.
    Set LEANN_TRUST_REMOTE_CODE=1 for a model that needs it;
    _assert_remote_code_available names the variable when it hits one.
    """
    return os.getenv("LEANN_TRUST_REMOTE_CODE", "").strip().lower() in {"1", "true", "yes"}


def _assert_remote_code_available(model_name: str, trust_remote_code: bool) -> None:
    """Refuse to load a model whose real architecture would be silently discarded.

    jinaai/jina-embeddings-v2-base-code and other custom encoders declare their
    architecture through config.auto_map while keeping model_type="bert". Loaded
    without trust_remote_code, transformers builds a vanilla BertModel instead,
    drops every weight it cannot map, randomly initialises the rest, and returns
    one constant vector for every input. Nothing raises: the index builds, search
    returns results, and the ranking is noise. Raise here instead.
    """
    if trust_remote_code:
        return
    try:
        from transformers import AutoConfig

        auto_map = getattr(AutoConfig.from_pretrained(model_name), "auto_map", None)
    except Exception as exc:
        # Offline, a private repo, or not a transformers model: nothing to assert.
        logger.debug(f"Could not read the config for {model_name}: {exc}")
        return
    if auto_map:
        raise ValueError(
            f"{model_name} ships its own modeling code (config.auto_map: "
            f"{sorted(auto_map)}). Loading it without trust_remote_code produces a "
            "randomly initialised model that returns an identical embedding for every "
            "input, so the index would rank by noise. Set LEANN_TRUST_REMOTE_CODE=1 to "
            "run the model repository's code, or pick a model that does not need it."
        )


def compute_embeddings_sentence_transformers(
    texts: list[str],
    model_name: str,
    use_fp16: bool = True,
    device: str = "auto",
    batch_size: int = 32,
    is_build: bool = False,
    adaptive_optimization: bool = True,
    manual_tokenize: bool = False,
    max_length: int = 512,
) -> np.ndarray:
    """
    Compute embeddings using SentenceTransformer with model caching and adaptive optimization

    Args:
        texts: List of texts to compute embeddings for
        model_name: Model name
        use_fp16: Whether to use FP16 precision
        device: Device to use ('auto', 'cuda', 'mps', 'cpu')
        batch_size: Batch size for processing
        is_build: Whether this is a build operation (shows progress bar)
        adaptive_optimization: Whether to use adaptive optimization based on batch size
    """
    import torch

    outer_start_time = time.time()
    # Handle empty input
    if not texts:
        raise ValueError("Cannot compute embeddings for empty text list")
    logger.info(
        f"Computing embeddings for {len(texts)} texts using SentenceTransformer, model: '{model_name}'"
    )

    # Auto-detect device
    if device == "auto":
        # Check environment variable first
        env_device = os.getenv("LEANN_EMBEDDING_DEVICE")
        if env_device:
            device = env_device
            logger.info(f"Using device from LEANN_EMBEDDING_DEVICE: {device}")
        elif torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # CPU torch ships no fp16 kernels for several ops (LayerNorm raises
    # "LayerNormKernelImpl" not implemented for 'Half') - force fp32 on CPU.
    if device == "cpu" and use_fp16:
        use_fp16 = False
        logger.info("CPU device detected: disabling fp16 (no fp16 CPU kernels)")

    # Apply optimizations based on benchmark results
    if adaptive_optimization:
        batch_size = _resolve_adaptive_batch_size(device, model_name)

    trust_remote_code = _resolve_trust_remote_code()
    _assert_remote_code_available(model_name, trust_remote_code)

    # Create cache key
    cache_key = (
        f"sentence_transformers_{model_name}_{device}_{use_fp16}_{trust_remote_code}_optimized"
    )

    pre_model_init_end_time = time.time()
    logger.debug(
        "compute_embeddings_sentence_transformers pre-model-init time "
        f"(device/batch selection etc.): {pre_model_init_end_time - outer_start_time:.6f}s"
    )

    # Check if model is already cached
    start_time = time.time()
    if cache_key in _model_cache:
        logger.info(f"Using cached optimized model: {model_name}")
        model = cast(_SentenceTransformerLike, _model_cache[cache_key])
    else:
        logger.info(f"Loading and caching optimized SentenceTransformer model: {model_name}")
        from sentence_transformers import SentenceTransformer

        logger.info(f"Using device: {device}")

        # Apply hardware optimizations
        if device == "cuda":
            # TODO: Haven't tested this yet
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.set_per_process_memory_fraction(0.9)
        elif device == "mps":
            # No device-level init for MPS. set_per_process_memory_fraction causes
            # greedy allocation; torch.compile causes graph buffer bloat. Cache
            # clearing is handled per-batch in the compute loop below.
            pass
        elif device == "cpu":
            # TODO: Haven't tested this yet
            torch.set_num_threads(_resolve_cpu_thread_count())
            try:
                torch.backends.mkldnn.enabled = True
            except AttributeError:
                pass

        # Prepare optimized model and tokenizer parameters
        model_kwargs = {
            "torch_dtype": torch.float16 if use_fp16 else torch.float32,
            "low_cpu_mem_usage": True,
            "_fast_init": True,
            "attn_implementation": "eager",  # Use eager attention for speed
        }

        tokenizer_kwargs = {
            "use_fast": True,
            "padding": True,
            "truncation": True,
        }

        try:
            # Try loading with advanced parameters first (newer versions)
            local_model_kwargs = model_kwargs.copy()
            local_tokenizer_kwargs = tokenizer_kwargs.copy()
            local_model_kwargs["local_files_only"] = True
            local_tokenizer_kwargs["local_files_only"] = True

            model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=trust_remote_code,
                model_kwargs=local_model_kwargs,
                tokenizer_kwargs=local_tokenizer_kwargs,
                local_files_only=True,
            )
            logger.info("Model loaded successfully! (local + optimized)")
        except TypeError as e:
            if "model_kwargs" in str(e) or "tokenizer_kwargs" in str(e):
                logger.warning(
                    f"Advanced parameters not supported ({e}), using basic initialization..."
                )
                # Fallback to basic initialization for older versions
                try:
                    model = SentenceTransformer(
                        model_name,
                        device=device,
                        trust_remote_code=trust_remote_code,
                        local_files_only=True,
                    )
                    logger.info("Model loaded successfully! (local + basic)")
                except Exception as e2:
                    logger.warning(f"Local loading failed ({e2}), trying network download...")
                    model = SentenceTransformer(
                        model_name,
                        device=device,
                        trust_remote_code=trust_remote_code,
                        local_files_only=False,
                    )
                    logger.info("Model loaded successfully! (network + basic)")
            else:
                raise
        except Exception as e:
            logger.warning(f"Local loading failed ({e}), trying network download...")
            # Fallback to network loading with advanced parameters
            try:
                network_model_kwargs = model_kwargs.copy()
                network_tokenizer_kwargs = tokenizer_kwargs.copy()
                network_model_kwargs["local_files_only"] = False
                network_tokenizer_kwargs["local_files_only"] = False

                model = SentenceTransformer(
                    model_name,
                    device=device,
                    trust_remote_code=trust_remote_code,
                    model_kwargs=network_model_kwargs,
                    tokenizer_kwargs=network_tokenizer_kwargs,
                    local_files_only=False,
                )
                logger.info("Model loaded successfully! (network + optimized)")
            except TypeError as e2:
                if "model_kwargs" in str(e2) or "tokenizer_kwargs" in str(e2):
                    logger.warning(
                        f"Advanced parameters not supported ({e2}), using basic network loading..."
                    )
                    model = SentenceTransformer(
                        model_name,
                        device=device,
                        trust_remote_code=trust_remote_code,
                        local_files_only=False,
                    )
                    logger.info("Model loaded successfully! (network + basic)")
                else:
                    raise

        # Apply additional optimizations based on mode
        if use_fp16 and device in ["cuda", "mps"]:
            try:
                model = model.half()
                logger.info(f"Applied FP16 precision: {model_name}")
            except Exception as e:
                logger.warning(f"FP16 optimization failed: {e}")

        # Apply torch.compile optimization
        if device == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead", dynamic=True)
                logger.info(f"Applied torch.compile optimization: {model_name}")
            except Exception as e:
                logger.warning(f"torch.compile optimization failed: {e}")

        model = cast(_SentenceTransformerLike, model)

        # Set model to eval mode and disable gradients for inference
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        # Cache the model
        _model_cache[cache_key] = model
        logger.info(f"Model cached: {cache_key}")

        end_time = time.time()

        # Compute embeddings with optimized inference mode
        logger.info(
            f"Starting embedding computation... (batch_size: {batch_size}, manual_tokenize={manual_tokenize})"
        )
        logger.info(f"start sentence transformers {model} takes {end_time - start_time}")

    if device == "cuda" and adaptive_optimization:
        batch_size = _cap_cuda_batch_by_vram(batch_size, max_length=max_length)

    start_time = time.time()
    if not manual_tokenize:
        # Use SentenceTransformer's optimized encode path (default)
        embeddings = _encode_with_oom_retry(
            model,
            texts,
            batch_size,
            is_build=is_build,
            device=device,
        )
        # Synchronize if CUDA to measure accurate wall time
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
    else:
        # Manual tokenization + forward pass using HF AutoTokenizer/AutoModel.
        # This path is reserved for an aggressively optimized FP pipeline
        # (no quantization), mainly for experimentation.
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:
            raise ImportError(f"transformers is required for manual_tokenize=True: {e}")

        tok_cache_key = f"hf_tokenizer_{model_name}"
        mdl_cache_key = f"hf_model_{model_name}_{device}_{use_fp16}_fp"

        if tok_cache_key in _model_cache and mdl_cache_key in _model_cache:
            hf_tokenizer = _model_cache[tok_cache_key]
            hf_model = _model_cache[mdl_cache_key]
            logger.info("Using cached HF tokenizer/model for manual FP path")
        else:
            logger.info("Loading HF tokenizer/model for manual FP path")
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

            torch_dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32
            hf_model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
            )
            hf_model.to(device)

            hf_model.eval()
            # Optional compile on supported devices
            if device == "cuda":
                try:
                    hf_model = torch.compile(hf_model, mode="reduce-overhead", dynamic=True)
                    logger.info(
                        f"Applied torch.compile to HF model for {model_name} "
                        f"(device={device}, dtype={torch_dtype})"
                    )
                except Exception as exc:
                    logger.warning(f"torch.compile optimization failed: {exc}")

            _model_cache[tok_cache_key] = hf_tokenizer
            _model_cache[mdl_cache_key] = hf_model

        all_embeddings: list[np.ndarray] = []
        # Progress bar when building or for large inputs
        show_progress = is_build or len(texts) > 32
        try:
            if show_progress:
                from tqdm import tqdm

                batch_iter = tqdm(
                    range(0, len(texts), batch_size),
                    desc="Embedding (manual)",
                    unit="batch",
                )
            else:
                batch_iter = range(0, len(texts), batch_size)
        except Exception:
            batch_iter = range(0, len(texts), batch_size)

        start_time_manual = time.time()
        with torch.inference_mode():
            for start_index in batch_iter:
                end_index = min(start_index + batch_size, len(texts))
                batch_texts = texts[start_index:end_index]
                inputs = hf_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = hf_model(**inputs)
                last_hidden_state = outputs.last_hidden_state  # (B, L, H)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    pooled = last_hidden_state.mean(dim=1)
                else:
                    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
                    masked = last_hidden_state * mask
                    lengths = mask.sum(dim=1).clamp(min=1)
                    pooled = masked.sum(dim=1) / lengths
                batch_embeddings = pooled.detach().to("cpu").float().numpy()
                all_embeddings.append(batch_embeddings)
                if device == "mps":
                    try:
                        torch.mps.empty_cache()
                    except (RuntimeError, AttributeError):
                        pass

        embeddings = np.vstack(all_embeddings).astype(np.float32, copy=False)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        end_time = time.time()
        logger.info(f"Manual tokenize time taken: {end_time - start_time_manual} seconds")
    end_time = time.time()
    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")
    logger.info(f"Time taken: {end_time - start_time} seconds")

    # Validate results
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise RuntimeError(f"Detected NaN or Inf values in embeddings, model: {model_name}")

    outer_end_time = time.time()
    logger.debug(
        "compute_embeddings_sentence_transformers total time "
        f"(function entry -> return): {outer_end_time - outer_start_time:.6f}s"
    )

    return embeddings


def compute_embeddings_openai(
    texts: list[str],
    model_name: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    provider_options: Optional[dict[str, Any]] = None,
    is_build: bool = False,
) -> np.ndarray:
    """Compute embeddings using OpenAI API"""
    try:
        import openai
    except ImportError as e:
        raise ImportError(f"OpenAI package not installed: {e}")

    # Validate input list
    if not texts:
        raise ValueError("Cannot compute embeddings for empty text list")
    # Extra validation: abort early if any item is empty/whitespace
    invalid_count = sum(1 for t in texts if not isinstance(t, str) or not t.strip())
    if invalid_count > 0:
        raise ValueError(
            f"Found {invalid_count} empty/invalid text(s) in input. Upstream should filter before calling OpenAI."
        )

    # Extract base_url and api_key from provider_options if not provided directly
    provider_options = provider_options or {}
    effective_base_url = base_url or provider_options.get("base_url")
    effective_api_key = api_key or provider_options.get("api_key")

    resolved_base_url = resolve_openai_base_url(effective_base_url)
    resolved_api_key = resolve_openai_api_key(effective_api_key)

    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    # Create OpenAI client
    client = openai.OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)

    logger.info(
        f"Computing embeddings for {len(texts)} texts using OpenAI API, model: '{model_name}'"
    )

    # Apply prompt template if provided
    # Priority: build_prompt_template (new format) > prompt_template (old format)
    prompt_template = provider_options.get("build_prompt_template") or provider_options.get(
        "prompt_template"
    )

    if prompt_template:
        logger.warning(f"Applying prompt template: '{prompt_template}'")
        texts = [f"{prompt_template}{text}" for text in texts]

    # Query token limit and apply truncation
    token_limit = get_model_token_limit(model_name, base_url=effective_base_url)
    logger.info(f"Using token limit: {token_limit} for model '{model_name}'")
    texts = truncate_to_token_limit(texts, token_limit)

    # OpenAI has limits on batch size and input length
    max_batch_size = 800  # Conservative batch size because the token limit is 300K
    all_embeddings = []
    # get the avg len of texts
    avg_len = sum(len(text) for text in texts) / len(texts)
    # if avg len is less than 1000, use the max batch size
    if avg_len > 300:
        max_batch_size = 500

    # Gemini's OpenAI-compatible endpoint hard-limits embedding batches to 100 inputs per request.
    # If we exceed this, the API returns:
    #   "BatchEmbedContentsRequest.requests: at most 100 requests can be in one batch"
    if "generativelanguage.googleapis.com" in (resolved_base_url or ""):
        max_batch_size = min(max_batch_size, 100)
        logger.info(
            "Detected Gemini OpenAI-compatible base_url; capping embedding batch_size to %d.",
            max_batch_size,
        )

    # Alibaba Cloud DashScope's OpenAI-compatible endpoint hard-limits embedding batches
    # to 10 inputs per request (e.g. text-embedding-v4). Exceeding this returns HTTP 400:
    #   "InternalError.Algo.InvalidParameter: Value error, batch size is invalid,
    #    it should not be larger than 10.: input.contents"
    if "dashscope" in (resolved_base_url or ""):
        max_batch_size = min(max_batch_size, 10)
        logger.info(
            "Detected DashScope OpenAI-compatible base_url; capping embedding batch_size to %d.",
            max_batch_size,
        )

    # if avg len is less than 1000, use the max batch size

    try:
        from tqdm import tqdm

        total_batches = (len(texts) + max_batch_size - 1) // max_batch_size
        batch_range = range(0, len(texts), max_batch_size)
        if is_build:
            batch_iterator = tqdm(
                batch_range, desc="Computing embeddings", unit="batch", total=total_batches
            )
        else:
            batch_iterator = batch_range
    except ImportError:
        # Fallback when tqdm is not available
        batch_iterator = range(0, len(texts), max_batch_size)

    for i in batch_iterator:
        batch_texts = texts[i : i + max_batch_size]

        try:
            response = client.embeddings.create(model=model_name, input=batch_texts)
            batch_embeddings = [embedding.embedding for embedding in response.data]

            # Verify we got the expected number of embeddings
            if len(batch_embeddings) != len(batch_texts):
                logger.warning(
                    f"Expected {len(batch_texts)} embeddings but got {len(batch_embeddings)}"
                )

            # Only take the number of embeddings that match the batch size
            all_embeddings.extend(batch_embeddings[: len(batch_texts)])
        except Exception as e:
            logger.error(f"Batch {i} failed: {e}")
            raise

    embeddings = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")
    return embeddings


def compute_embeddings_mlx(
    chunks: list[str], model_name: str, batch_size: int = 16, is_build: bool = False
) -> np.ndarray:
    """Computes embeddings using an MLX model."""
    try:
        import mlx.core as mx
        from mlx_lm.utils import load
    except ImportError as e:
        raise RuntimeError(
            "MLX or related libraries not available. Install with: uv pip install mlx mlx-lm"
        ) from e

    logger.info(
        f"Computing embeddings for {len(chunks)} chunks using MLX model '{model_name}' with batch_size={batch_size}..."
    )

    # Cache MLX model and tokenizer
    cache_key = f"mlx_{model_name}"
    if cache_key in _model_cache:
        logger.info(f"Using cached MLX model: {model_name}")
        model, tokenizer = _model_cache[cache_key]
    else:
        logger.info(f"Loading and caching MLX model: {model_name}")
        model, tokenizer = load(model_name)
        _model_cache[cache_key] = (model, tokenizer)
        logger.info(f"MLX model cached: {cache_key}")

    # Process chunks in batches with progress bar
    all_embeddings = []

    try:
        from tqdm import tqdm

        batch_range = range(0, len(chunks), batch_size)
        if is_build:
            batch_iterator = tqdm(batch_range, desc="Computing embeddings", unit="batch")
        else:
            batch_iterator = batch_range
    except ImportError:
        batch_iterator = range(0, len(chunks), batch_size)

    for i in batch_iterator:
        batch_chunks = chunks[i : i + batch_size]

        # Tokenize all chunks in the batch
        batch_token_ids = []
        for chunk in batch_chunks:
            token_ids = tokenizer.encode(chunk)
            batch_token_ids.append(token_ids)

        # Pad sequences to the same length for batch processing
        max_length = max(len(ids) for ids in batch_token_ids)
        padded_token_ids = []
        for token_ids in batch_token_ids:
            # Pad with tokenizer.pad_token_id or 0
            padded = token_ids + [0] * (max_length - len(token_ids))
            padded_token_ids.append(padded)

        # Convert to MLX array with batch dimension
        input_ids = mx.array(padded_token_ids)

        # Get embeddings for the batch
        embeddings = model(input_ids)

        # Mean pooling for each sequence in the batch
        pooled = embeddings.mean(axis=1)  # Shape: (batch_size, hidden_size)

        # Convert batch embeddings to numpy
        for j in range(len(batch_chunks)):
            pooled_list = pooled[j].tolist()  # Convert to list
            pooled_numpy = np.array(pooled_list, dtype=np.float32)
            all_embeddings.append(pooled_numpy)

    # Stack numpy arrays
    return np.stack(all_embeddings)


def compute_embeddings_ollama(
    texts: list[str],
    model_name: str,
    is_build: bool = False,
    host: Optional[str] = None,
    provider_options: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """
    Compute embeddings using Ollama API with true batch processing.

    Uses the /api/embed endpoint which supports batch inputs.
    Batch size: 32 for MPS/CPU, 128 for CUDA to optimize performance.

    Args:
        texts: List of texts to compute embeddings for
        model_name: Ollama model name (e.g., "nomic-embed-text", "mxbai-embed-large")
        is_build: Whether this is a build operation (shows progress bar)
        host: Ollama host URL (defaults to environment or http://localhost:11434)
        provider_options: Optional provider-specific options (e.g., prompt_template)

    Returns:
        Normalized embeddings array, shape: (len(texts), embedding_dim)
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "The 'requests' library is required for Ollama embeddings. Install with: uv pip install requests"
        )

    if not texts:
        raise ValueError("Cannot compute embeddings for empty text list")

    resolved_host = resolve_ollama_host(host)

    logger.info(
        f"Computing embeddings for {len(texts)} texts using Ollama API, model: '{model_name}', host: '{resolved_host}'"
    )

    # Check if Ollama is running
    try:
        response = requests.get(f"{resolved_host}/api/version", timeout=5)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        error_msg = (
            f"❌ Could not connect to Ollama at {resolved_host}.\n\n"
            "Please ensure Ollama is running:\n"
            "  • macOS/Linux: ollama serve\n"
            "  • Windows: Make sure Ollama is running in the system tray\n\n"
            "Installation: https://ollama.com/download"
        )
        raise RuntimeError(error_msg)
    except Exception as e:
        raise RuntimeError(f"Unexpected error connecting to Ollama: {e}")

    # Check if model exists and provide helpful suggestions
    try:
        response = requests.get(f"{resolved_host}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json()
        model_names = [model["name"] for model in models.get("models", [])]

        # Filter for embedding models (models that support embeddings)
        embedding_models = []
        suggested_embedding_models = [
            "nomic-embed-text",
            "mxbai-embed-large",
            "bge-m3",
            "all-minilm",
            "snowflake-arctic-embed",
        ]

        for model in model_names:
            # Check if it's an embedding model (by name patterns or known models)
            base_name = model.split(":")[0]
            if any(emb in base_name for emb in ["embed", "bge", "minilm", "e5"]):
                embedding_models.append(model)

        # Check if model exists (handle versioned names) and resolve to full name
        resolved_model_name = None
        for name in model_names:
            # Exact match
            if model_name == name:
                resolved_model_name = name
                break
            # Match without version tag (use the versioned name)
            elif model_name == name.split(":")[0]:
                resolved_model_name = name
                break

        if not resolved_model_name:
            error_msg = f"❌ Model '{model_name}' not found in local Ollama.\n\n"

            # Suggest pulling the model
            error_msg += "📦 To install this embedding model:\n"
            error_msg += f"   ollama pull {model_name}\n\n"

            # Show available embedding models
            if embedding_models:
                error_msg += "✅ Available embedding models:\n"
                for model in embedding_models[:5]:
                    error_msg += f"   • {model}\n"
                if len(embedding_models) > 5:
                    error_msg += f"   ... and {len(embedding_models) - 5} more\n"
            else:
                error_msg += "💡 Popular embedding models to install:\n"
                for model in suggested_embedding_models[:3]:
                    error_msg += f"   • ollama pull {model}\n"

            error_msg += "\n📚 Browse more: https://ollama.com/library"
            raise ValueError(error_msg)

        # Use the resolved model name for all subsequent operations
        if resolved_model_name != model_name:
            logger.info(f"Resolved model name '{model_name}' to '{resolved_model_name}'")
        model_name = resolved_model_name

        # Verify the model supports embeddings by testing it with /api/embed
        try:
            test_response = requests.post(
                f"{resolved_host}/api/embed",
                json={"model": model_name, "input": "test"},
                timeout=10,
            )
            if test_response.status_code != 200:
                error_msg = (
                    f"⚠️ Model '{model_name}' exists but may not support embeddings.\n\n"
                    f"Please use an embedding model like:\n"
                )
                for model in suggested_embedding_models[:3]:
                    error_msg += f"   • {model}\n"
                raise ValueError(error_msg)
        except requests.exceptions.RequestException:
            # If test fails, continue anyway - model might still work
            pass

    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not verify model existence: {e}")

    # Determine batch size based on device availability
    # Check for CUDA/MPS availability using torch if available
    batch_size = 32  # Default for MPS/CPU
    try:
        import torch

        if torch.cuda.is_available():
            batch_size = 128  # CUDA gets larger batch size
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            batch_size = 32  # MPS gets smaller batch size
    except ImportError:
        # If torch is not available, use conservative batch size
        batch_size = 32

    logger.info(f"Using batch size: {batch_size} for true batch processing")

    # Apply prompt template if provided
    provider_options = provider_options or {}
    # Priority: build_prompt_template (new format) > prompt_template (old format)
    prompt_template = provider_options.get("build_prompt_template") or provider_options.get(
        "prompt_template"
    )

    if prompt_template:
        logger.warning(f"Applying prompt template: '{prompt_template}'")
        texts = [f"{prompt_template}{text}" for text in texts]

    # Get model token limit and apply truncation before batching
    token_limit = get_model_token_limit(model_name, base_url=resolved_host)
    logger.info(f"Model '{model_name}' token limit: {token_limit}")

    # Apply truncation to all texts before batch processing
    # Function logs truncation details internally
    texts = truncate_to_token_limit(texts, token_limit)

    def get_batch_embeddings(batch_texts):
        """Get embeddings for a batch of texts using /api/embed endpoint."""
        max_retries = 3
        retry_count = 0

        # Texts are already truncated to token limit by the outer function
        while retry_count < max_retries:
            try:
                # Use /api/embed endpoint with "input" parameter for batch processing
                response = requests.post(
                    f"{resolved_host}/api/embed",
                    json={"model": model_name, "input": batch_texts},
                    timeout=60,  # Increased timeout for batch processing
                )
                response.raise_for_status()

                result = response.json()
                batch_embeddings = result.get("embeddings")

                if batch_embeddings is None:
                    raise ValueError("No embeddings returned from API")

                if not isinstance(batch_embeddings, list):
                    raise ValueError(f"Invalid embeddings format: {type(batch_embeddings)}")

                if len(batch_embeddings) != len(batch_texts):
                    raise ValueError(
                        f"Mismatch: requested {len(batch_texts)} embeddings, got {len(batch_embeddings)}"
                    )

                return batch_embeddings, []

            except requests.exceptions.Timeout:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.warning(f"Timeout for batch after {max_retries} retries")
                    return None, list(range(len(batch_texts)))

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    # Enhanced error detection for token limit violations
                    error_msg = str(e).lower()
                    if "token" in error_msg and (
                        "limit" in error_msg or "exceed" in error_msg or "length" in error_msg
                    ):
                        logger.error(
                            f"Token limit exceeded for batch. Error: {e}. "
                            f"Consider reducing chunk sizes or check token truncation."
                        )
                    else:
                        logger.error(f"Failed to get embeddings for batch: {e}")
                    return None, list(range(len(batch_texts)))

        return None, list(range(len(batch_texts)))

    # Process texts in batches
    all_embeddings = []
    all_failed_indices = []

    # Setup progress bar if needed
    show_progress = is_build or len(texts) > 10
    try:
        if show_progress:
            from tqdm import tqdm
    except ImportError:
        show_progress = False

    # Process batches
    num_batches = (len(texts) + batch_size - 1) // batch_size

    if show_progress:
        batch_iterator = tqdm(range(num_batches), desc="Computing Ollama embeddings (batched)")
    else:
        batch_iterator = range(num_batches)

    for batch_idx in batch_iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        batch_embeddings, batch_failed = get_batch_embeddings(batch_texts)

        if batch_embeddings is not None:
            all_embeddings.extend(batch_embeddings)
        else:
            # Entire batch failed, add None placeholders
            all_embeddings.extend([None] * len(batch_texts))
            # Adjust failed indices to global indices
            global_failed = [start_idx + idx for idx in batch_failed]
            all_failed_indices.extend(global_failed)

    # Handle failed embeddings
    if all_failed_indices:
        if len(all_failed_indices) == len(texts):
            raise RuntimeError("Failed to compute any embeddings")

        logger.warning(
            f"Failed to compute embeddings for {len(all_failed_indices)}/{len(texts)} texts"
        )

        # Use zero embeddings as fallback for failed ones
        valid_embedding = next((e for e in all_embeddings if e is not None), None)
        if valid_embedding:
            embedding_dim = len(valid_embedding)
            for i, embedding in enumerate(all_embeddings):
                if embedding is None:
                    all_embeddings[i] = [0.0] * embedding_dim

    # Remove None values
    all_embeddings = [e for e in all_embeddings if e is not None]

    if not all_embeddings:
        raise RuntimeError("No valid embeddings were computed")

    # Validate embedding dimensions
    expected_dim = len(all_embeddings[0])
    inconsistent_dims = []
    for i, embedding in enumerate(all_embeddings):
        if len(embedding) != expected_dim:
            inconsistent_dims.append((i, len(embedding)))

    if inconsistent_dims:
        error_msg = f"Ollama returned inconsistent embedding dimensions. Expected {expected_dim}, but got:\n"
        for idx, dim in inconsistent_dims[:10]:  # Show first 10 inconsistent ones
            error_msg += f"  - Text {idx}: {dim} dimensions\n"
        if len(inconsistent_dims) > 10:
            error_msg += f"  ... and {len(inconsistent_dims) - 10} more\n"
        error_msg += f"\nThis is likely an Ollama API bug with model '{model_name}'. Please try:\n"
        error_msg += "1. Restart Ollama service: 'ollama serve'\n"
        error_msg += f"2. Re-pull the model: 'ollama pull {model_name}'\n"
        error_msg += (
            "3. Use sentence-transformers instead: --embedding-mode sentence-transformers\n"
        )
        error_msg += "4. Report this issue to Ollama: https://github.com/ollama/ollama/issues"
        raise ValueError(error_msg)

    # Convert to numpy array and normalize
    embeddings = np.array(all_embeddings, dtype=np.float32)

    # Normalize embeddings (L2 normalization)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)  # Add small epsilon to avoid division by zero

    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")

    return embeddings


def compute_embeddings_gemini(
    texts: list[str], model_name: str = "text-embedding-004", is_build: bool = False
) -> np.ndarray:
    """
    Compute embeddings using Google Gemini API.

    Args:
        texts: List of texts to compute embeddings for
        model_name: Gemini model name (default: "text-embedding-004")
        is_build: Whether this is a build operation (shows progress bar)

    Returns:
        Embeddings array, shape: (len(texts), embedding_dim)
    """
    try:
        import os

        import google.genai as genai
    except ImportError as e:
        raise ImportError(f"Google GenAI package not installed: {e}")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    # Cache Gemini client
    cache_key = "gemini_client"
    if cache_key in _model_cache:
        client = _model_cache[cache_key]
    else:
        client = genai.Client(api_key=api_key)
        _model_cache[cache_key] = client
        logger.info("Gemini client cached")

    logger.info(
        f"Computing embeddings for {len(texts)} texts using Gemini API, model: '{model_name}'"
    )

    # Gemini supports batch embedding
    max_batch_size = 100  # Conservative batch size for Gemini
    all_embeddings = []

    try:
        from tqdm import tqdm

        total_batches = (len(texts) + max_batch_size - 1) // max_batch_size
        batch_range = range(0, len(texts), max_batch_size)
        batch_iterator = tqdm(
            batch_range, desc="Computing embeddings", unit="batch", total=total_batches
        )
    except ImportError:
        # Fallback when tqdm is not available
        batch_iterator = range(0, len(texts), max_batch_size)

    for i in batch_iterator:
        batch_texts = texts[i : i + max_batch_size]

        try:
            # Use the embed_content method from the new Google GenAI SDK
            response = client.models.embed_content(
                model=model_name,
                contents=batch_texts,
                config=genai.types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"  # For document embedding
                ),
            )

            # Extract embeddings from response
            for embedding_data in response.embeddings:
                all_embeddings.append(embedding_data.values)
        except Exception as e:
            logger.error(f"Batch {i} failed: {e}")
            raise

    embeddings = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")

    return embeddings
