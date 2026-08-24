"""Tests for the HF backend's chat-prompt fallback (no model, no network).

The ``hf`` backend must work with tokenizers that have no ``chat_template``
(e.g. the Gemma-4 family, plain GPT-2). These tests drive the manual prompt
builders with a fake tokenizer and assert the exact rendered format.
"""

import unittest

from backend import HFBackend


class _FakeTok:
    """Minimal stand-in for a tokenizer with configurable special tokens."""

    def __init__(
        self,
        eot_token=None,
        sot_token=None,
        bos_token="<bos>",
        unk_token_id=3,
        chat_template=None,
        known_tokens=(),
    ):
        self.eot_token = eot_token
        self.sot_token = sot_token
        self.bos_token = bos_token
        self.unk_token_id = unk_token_id
        self.chat_template = chat_template
        self._known = set(known_tokens)
        self.rendered = []  # every string passed to __call__

    def convert_tokens_to_ids(self, token):
        return 1 if token in self._known else self.unk_token_id

    def __call__(self, text, **kwargs):
        self.rendered.append(text)
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}


def make_backend(fake) -> HFBackend:
    backend = HFBackend(model_name="fake")
    backend.tokenizer = fake
    backend._loaded = True  # skip _ensure_loaded
    return backend


class TestGemma4Prompt(unittest.TestCase):
    def setUp(self):
        fake = _FakeTok(eot_token="<turn|>", sot_token="<|turn>", bos_token="<bos>")
        self.backend = make_backend(fake)

    def test_nonthinking_system_user(self):
        msgs = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "hello"},
        ]
        expected = (
            "<bos><|turn>system\nYou are a helper.<turn|>\n"
            "<|turn>user\nhello<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>"
        )
        self.assertEqual(self.backend._gemma4_chat_prompt(msgs), expected)

    def test_thinking_injects_think_token(self):
        self.backend.template_kwargs = {"enable_thinking": True}
        msgs = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "hello"},
        ]
        expected = (
            "<bos><|turn>system\n<|think|>\nYou are a helper.<turn|>\n"
            "<|turn>user\nhello<turn|>\n"
            "<|turn>model\n"
        )
        self.assertEqual(self.backend._gemma4_chat_prompt(msgs), expected)

    def test_assistant_role_maps_to_model(self):
        msgs = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "U2"},
        ]
        expected = (
            "<bos><|turn>system\nSYS<turn|>\n"
            "<|turn>user\nU1<turn|>\n"
            "<|turn>model\nA1<turn|>\n"
            "<|turn>user\nU2<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>"
        )
        self.assertEqual(self.backend._gemma4_chat_prompt(msgs), expected)


class TestOtherFallbacks(unittest.TestCase):
    def test_chatml_fallback(self):
        fake = _FakeTok(bos_token="<bos>", known_tokens=("<|user|>",))
        backend = make_backend(fake)
        msgs = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "hello"},
        ]
        expected = (
            "<bos><|system|>\nYou are a helper.<|end|>\n"
            "<|user|>\nhello<|end|>\n"
            "<|assistant|>\n"
        )
        self.assertEqual(backend._chatml_chat_prompt(msgs), expected)
        self.assertEqual(backend._manual_chat_prompt(msgs), expected)

    def test_generic_fallback(self):
        fake = _FakeTok(bos_token="<|endoftext|>")
        backend = make_backend(fake)
        msgs = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "hello"},
        ]
        expected = "<|endoftext|>System: You are a helper.\n\nUser: hello\n\nAssistant:"
        self.assertEqual(backend._generic_chat_prompt(msgs), expected)
        self.assertEqual(backend._manual_chat_prompt(msgs), expected)

    def test_gemma4_detected_by_eot_token(self):
        fake = _FakeTok(eot_token="<turn|>")
        backend = make_backend(fake)
        self.assertEqual(backend._manual_chat_prompt(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        ), backend._gemma4_chat_prompt(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        ))


class TestTokenizeDispatch(unittest.TestCase):
    def test_uses_apply_chat_template_when_configured(self):
        fake = _FakeTok(chat_template="fake-template", known_tokens=("<|user|>",))

        def apply_chat_template(messages, **kwargs):
            return {"input_ids": [[9, 9]], "attention_mask": [[1, 1]]}

        fake.apply_chat_template = apply_chat_template
        backend = make_backend(fake)
        ids, mask = backend._tokenize(
            [{"role": "user", "content": "hi"}]
        )
        self.assertEqual(ids, [[9, 9]])
        self.assertEqual(mask, [[1, 1]])

    def test_manual_path_without_template(self):
        fake = _FakeTok(eot_token="<turn|>", bos_token="<bos>")
        backend = make_backend(fake)
        ids, mask = backend._tokenize(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        )
        self.assertEqual(ids, [[1, 2, 3]])
        self.assertEqual(mask, [[1, 1, 1]])
        # The rendered prompt was sent through the tokenizer unchanged.
        self.assertEqual(len(fake.rendered), 1)
        self.assertTrue(fake.rendered[0].startswith("<bos><|turn>system\nS<turn|>\n"))
        self.assertTrue(fake.rendered[0].endswith("<|turn>model\n<|channel>thought\n<channel|>"))


if __name__ == "__main__":
    unittest.main()
