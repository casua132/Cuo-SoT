"""Tests for the HF backend's chat-prompt fallback (no model, no network).

The ``hf`` backend must work with tokenizers that have no ``chat_template``
(e.g. the Gemma-4 family, plain GPT-2). These tests drive the manual prompt
builders with a fake tokenizer and assert the exact rendered format.
"""

import unittest
from collections import UserDict

from backend import HFBackend, StubBackend

try:  # torch is only needed by the HF-backend batching tests below
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - environment without torch
    torch = None
    _HAS_TORCH = False


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
            # Real ``transformers`` returns a ``BatchEncoding``, a UserDict — not
            # a plain dict. This is the return type that used to break the
            # ``isinstance(chat, dict)`` guard in ``_tokenize``.
            return UserDict({"input_ids": [[9, 9]], "attention_mask": [[1, 1]]})

        fake.apply_chat_template = apply_chat_template
        backend = make_backend(fake)
        ids, mask = backend._tokenize(
            [{"role": "user", "content": "hi"}]
        )
        self.assertEqual(ids, [[9, 9]])
        self.assertEqual(mask, [[1, 1]])

    def test_uses_apply_chat_template_when_configured_plain_dict(self):
        # Some transformers versions return a plain dict — both must work.
        fake = _FakeTok(chat_template="fake-template", known_tokens=("<|user|>",))

        def apply_chat_template(messages, **kwargs):
            return {"input_ids": [[7]], "attention_mask": [[1]]}

        fake.apply_chat_template = apply_chat_template
        backend = make_backend(fake)
        ids, mask = backend._tokenize([{"role": "user", "content": "hi"}])
        self.assertEqual(ids, [[7]])
        self.assertEqual(mask, [[1]])

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


class TestStubCompleteBatch(unittest.TestCase):
    def test_loops_in_order_with_per_item_responses(self):
        seen = []

        def response_fn(messages):
            seen.append(messages[0]["content"])
            return f"({chr(ord('a') + len(seen) - 1)})"

        backend = StubBackend(response_fn=response_fn)
        msgs_list = [[{"role": "system", "content": f"m{i}"}] for i in range(3)]
        out = backend.complete_batch(msgs_list, max_new_tokens=8)
        self.assertEqual(out, ["(a)", "(b)", "(c)"])
        self.assertEqual(seen, ["m0", "m1", "m2"])
        self.assertEqual(backend.call_count, 3)

    def test_empty_list(self):
        backend = StubBackend(answer_response="(a)")
        self.assertEqual(backend.complete_batch([]), [])

    def test_state_answers_keep_their_response(self):
        backend = StubBackend(answer_response="(a)", state_response="**name**: unknown")
        msgs_list = [
            [{"role": "system", "content": "You are a psychological expert..."}],
            [{"role": "system", "content": "plain"}],
        ]
        self.assertEqual(backend.complete_batch(msgs_list), ["**name**: unknown", "(a)"])


class _FakeBatchedModel:
    """Records generate() inputs and returns per-row distinct filler tokens."""

    def __init__(self):
        self.device = "cpu"
        self.calls = []

    def generate(self, input_ids, attention_mask=None, max_new_tokens=None, do_sample=None):
        self.calls.append({
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask.clone() if attention_mask is not None else None,
            "max_new_tokens": max_new_tokens,
        })
        # Append row-distinct filler (100 + row_index) so the order is checkable.
        B, _ = input_ids.shape
        filler = torch.arange(B, dtype=torch.long).unsqueeze(1).expand(B, max_new_tokens) + 100
        return torch.cat([input_ids, filler], dim=-1)


class _FakeBatchedTok:
    """Tokenizes through the chat-template path; length tracks message content."""

    def __init__(self):
        self.eot_token = None
        self.sot_token = None
        self.bos_token = "<bos>"
        self.unk_token_id = 3
        self.chat_template = "fake"
        self.pad_token_id = 0
        self.eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        content = messages[-1]["content"]
        n = 2 + len(content)
        return {
            "input_ids": torch.full((1, n), 5, dtype=torch.long),
            "attention_mask": torch.ones(1, n, dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        return "|".join(str(int(t)) for t in ids)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestHFCompleteBatch(unittest.TestCase):
    def setUp(self):
        self.model = _FakeBatchedModel()
        self.tok = _FakeBatchedTok()
        self.backend = HFBackend(model_name="fake")
        self.backend.model = self.model
        self.backend.tokenizer = self.tok
        self.backend._loaded = True

    def test_single_generate_call_with_left_padding(self):
        msgs_list = [
            [{"role": "user", "content": "x"}],
            [{"role": "user", "content": "xx"}],
            [{"role": "user", "content": "xxx"}],
        ]
        self.backend.complete_batch(msgs_list, max_new_tokens=2)

        self.assertEqual(len(self.model.calls), 1)
        call = self.model.calls[0]
        ids, mask = call["input_ids"], call["attention_mask"]
        self.assertEqual(call["max_new_tokens"], 2)
        # content lengths 3/4/5 -> one row padded to 5, pad id 0 on the left.
        self.assertEqual(ids.shape, (3, 5))
        self.assertEqual(ids.tolist(), [
            [0, 0, 5, 5, 5],
            [0, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
        ])
        self.assertEqual(mask.tolist(), [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ])

    def test_order_preserved_and_per_row_slice(self):
        msgs_list = [
            [{"role": "user", "content": "x"}],
            [{"role": "user", "content": "xx"}],
            [{"role": "user", "content": "xxx"}],
        ]
        out = self.backend.complete_batch(msgs_list, max_new_tokens=2)
        # filler = 100 + row_index, repeated max_new_tokens times.
        self.assertEqual(out, ["100|100", "101|101", "102|102"])

    def test_empty_list_returns_empty(self):
        self.assertEqual(self.backend.complete_batch([]), [])

    def test_falls_back_to_eos_pad_when_no_pad_token(self):
        self.tok.pad_token_id = None
        self.backend.complete_batch(
            [[{"role": "user", "content": "x"}], [{"role": "user", "content": "xx"}]],
            max_new_tokens=1,
        )
        self.assertEqual(self.tok.pad_token_id, 2)  # eos_token_id


if __name__ == "__main__":
    unittest.main()
