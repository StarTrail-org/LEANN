"""HFChat must be constructible outside the main thread."""

import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from leann import chat


@pytest.fixture
def fake_hf(monkeypatch):
    """Stub torch and transformers so no model is downloaded."""
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    torch.backends.mps.is_available.return_value = False
    transformers = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.delenv("LEANN_LLM_DEVICE", raising=False)
    monkeypatch.setattr(chat, "validate_model_and_suggest", lambda *args: None)
    return transformers


def test_hf_chat_loads_in_worker_thread(fake_hf):
    with ThreadPoolExecutor(max_workers=1) as pool:
        llm = pool.submit(chat.HFChat, "some/model").result()

    assert llm.model is fake_hf.AutoModelForCausalLM.from_pretrained.return_value
    assert llm.tokenizer is fake_hf.AutoTokenizer.from_pretrained.return_value


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM is not available")
def test_hf_chat_keeps_load_timeout_in_main_thread(fake_hf, monkeypatch):
    alarms = []
    monkeypatch.setattr(signal, "alarm", alarms.append)
    previous = signal.getsignal(signal.SIGALRM)

    chat.HFChat("some/model")

    assert alarms == [60, 0]
    assert signal.getsignal(signal.SIGALRM) == previous
