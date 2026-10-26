"""Guards against silently loading a custom encoder as a vanilla BertModel.

jinaai/jina-embeddings-v2-base-code declares its architecture through
config.auto_map while keeping model_type="bert". Loaded without
trust_remote_code, transformers builds a plain BertModel, discards the weights
it cannot map, randomly initialises ~70 tensors, and returns one constant
vector for every input. Nothing raises, so an index builds and searches
normally while ranking by noise. These tests pin the refusal and the opt-in.
"""

import unittest
from unittest.mock import patch

from leann.embedding_compute import _assert_remote_code_available, _resolve_trust_remote_code

JINA_AUTO_MAP = {
    "AutoConfig": "jinaai/jina-bert-v2-qk-post-norm--configuration_bert.JinaBertConfig",
    "AutoModel": "jinaai/jina-bert-v2-qk-post-norm--modeling_bert.JinaBertModel",
}


class _Config:
    def __init__(self, auto_map=None):
        if auto_map is not None:
            self.auto_map = auto_map


class TestResolveTrustRemoteCode(unittest.TestCase):
    def test_absent_env_is_off(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(_resolve_trust_remote_code())

    def test_truthy_values_enable_it(self):
        for value in ("1", "true", "TRUE", "yes", " Yes "):
            with patch.dict("os.environ", {"LEANN_TRUST_REMOTE_CODE": value}, clear=True):
                self.assertTrue(_resolve_trust_remote_code(), value)

    def test_other_values_leave_it_off(self):
        for value in ("0", "false", "no", "", "maybe"):
            with patch.dict("os.environ", {"LEANN_TRUST_REMOTE_CODE": value}, clear=True):
                self.assertFalse(_resolve_trust_remote_code(), value)


class TestAssertRemoteCodeAvailable(unittest.TestCase):
    def _with_config(self, config):
        return patch("transformers.AutoConfig.from_pretrained", return_value=config)

    def test_rejects_auto_map_model_when_not_trusted(self):
        with self._with_config(_Config(JINA_AUTO_MAP)):
            with self.assertRaises(ValueError) as ctx:
                _assert_remote_code_available("jinaai/jina-embeddings-v2-base-code", False)
        message = str(ctx.exception)
        self.assertIn("LEANN_TRUST_REMOTE_CODE", message)
        self.assertIn("identical embedding", message)

    def test_allows_auto_map_model_when_trusted(self):
        with self._with_config(_Config(JINA_AUTO_MAP)):
            _assert_remote_code_available("jinaai/jina-embeddings-v2-base-code", True)

    def test_allows_ordinary_model(self):
        with self._with_config(_Config()):
            _assert_remote_code_available("sentence-transformers/all-MiniLM-L6-v2", False)

    def test_allows_empty_auto_map(self):
        with self._with_config(_Config({})):
            _assert_remote_code_available("some/model", False)

    def test_unreadable_config_does_not_block_loading(self):
        # Offline, a private repo, or a non-transformers model: nothing to assert.
        with patch("transformers.AutoConfig.from_pretrained", side_effect=OSError("offline")):
            _assert_remote_code_available("some/model", False)


if __name__ == "__main__":
    unittest.main()
