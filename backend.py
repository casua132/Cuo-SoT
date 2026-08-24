"""Inference backends for the cot / cot-opt pipeline.

Three backends are provided:

- ``stub`` — deterministic, no model. Used for tests and pipeline smoke checks.
- ``hf``   — local inference through ``transformers`` (lazy import; requires
  ``torch`` + ``transformers`` to be installed).
- ``api``  — any OpenAI-compatible chat-completions endpoint (OpenRouter, vLLM,
  LM Studio, ...). Requires the ``openai`` package and credentials.

All backends share the same interface::

    backend.complete(messages, max_new_tokens=1024) -> str

where ``messages`` is a chat message list, e.g.
``[{"role": "system", "content": ...}, {"role": "user", "content": ...}]``.
"""

from __future__ import annotations

import os

from utils import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL


class LLMBackend:
    """Base class: the chat-completion interface."""

    name = "base"

    def complete(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Stub backend (deterministic, no model)
# ---------------------------------------------------------------------------
def default_state_text() -> str:
    """A valid all-``unknown`` state block, used by the stub when no state response is given."""
    from state import empty_state, format_user_state

    return format_user_state(empty_state())


class StubBackend(LLMBackend):
    """Deterministic backend for tests and dry runs.

    Returns ``state_response`` for state-update calls (identified by the
    ``intent_induce`` system prompt) and ``answer_response`` for answer calls.
    A custom ``response_fn(messages) -> str | None`` can override this.
    """

    name = "stub"

    def __init__(
        self,
        answer_response: str = "(a)",
        state_response: str | None = None,
        response_fn=None,
        max_new_tokens: int | None = None,
    ) -> None:
        self.answer_response = answer_response
        self.state_response = state_response if state_response is not None else default_state_text()
        self.response_fn = response_fn
        self.max_new_tokens = max_new_tokens
        self.call_count = 0
        self.calls: list[tuple[list[dict], int | None]] = []

    def complete(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        self.call_count += 1
        self.calls.append((messages, max_new_tokens))
        if self.response_fn is not None:
            out = self.response_fn(messages)
            if out is not None:
                return out
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        if "psychological expert" in system:
            return self.state_response
        return self.answer_response


# ---------------------------------------------------------------------------
# HuggingFace / transformers backend
# ---------------------------------------------------------------------------
def _message_text(message: dict) -> str:
    """Render a message's ``content`` as plain text (handles parts-lists)."""
    content = message.get("content") or ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content if p.get("type") == "text"
        ).strip()
    return str(content).strip()


class HFBackend(LLMBackend):
    """Local inference through ``transformers``.

    The model is loaded lazily on the first call so that importing this module
    never requires ``torch``/``transformers`` to be installed.

    Chat messages are tokenized with ``tokenizer.apply_chat_template`` when the
    tokenizer has a ``chat_template``. Some model families ship a tokenizer with
    no chat template at all (Gemma-4, plain GPT-2); for those the prompt is built
    manually with a template matched to the model family (see
    :meth:`_manual_chat_prompt`).
    """

    name = "hf"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        template_kwargs: dict | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.template_kwargs = template_kwargs or {}
        self.max_new_tokens = max_new_tokens
        self._loaded = False
        self.model = None
        self.tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "The 'hf' backend requires 'torch' and 'transformers'. "
                "Install them with: pip install torch transformers\n"
                "Alternatively use --backend api (hosted model) or --backend stub (dry run)."
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=self.torch_dtype, device_map=self.device_map
            )
        except Exception:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                self.model_name, torch_dtype=self.torch_dtype, device_map=self.device_map
            )
        self._loaded = True

    def complete(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        self._ensure_loaded()
        max_new_tokens = max_new_tokens or self.max_new_tokens

        input_ids, attention_mask = self._tokenize(messages)
        input_len = input_ids.shape[-1]
        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    # ------------------------------------------------------------------ tokenize
    def _tokenize(self, messages: list[dict]) -> tuple:
        """Tokenize a chat message list into ``(input_ids, attention_mask)``.

        Uses ``tokenizer.apply_chat_template`` when a ``chat_template`` is
        configured. Models without one (e.g. the Gemma-4 family) fall back to a
        manual prompt builder matched to the model family.
        """
        if self.tokenizer.chat_template is not None:
            chat = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **self.template_kwargs,
            )
            # ``apply_chat_template(tokenize=True, return_dict=True)`` returns a
            # ``transformers.BatchEncoding``, which is a ``UserDict`` — NOT a
            # plain ``dict``. An ``isinstance(chat, dict)`` check silently misses
            # it and would return the whole encoding as ``input_ids``. Duck-type
            # on the key instead so both return types are handled.
            if "input_ids" in chat:
                return chat["input_ids"], chat.get("attention_mask")
            return chat, None

        prompt = self._manual_chat_prompt(messages)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return enc["input_ids"], enc.get("attention_mask")

    # ------------------------------------------- manual chat prompts (fallback)
    def _manual_chat_prompt(self, messages: list[dict]) -> str:
        """Build a prompt string for a tokenizer that has no ``chat_template``.

        The format is matched to the model family by the special tokens it
        declares: Gemma-4's ``<|turn>`` format, then Gemma-2/3's
        ``<start_of_turn>`` format, then ChatML, then a plain labeled fallback.
        """
        tok = self.tokenizer
        if getattr(tok, "eot_token", None) == "<turn|>" or getattr(tok, "sot_token", None) == "<|turn>":
            return self._gemma4_chat_prompt(messages)
        if self._has_token("<start_of_turn>"):
            return self._gemma3_chat_prompt(messages)
        if self._has_token("<|user|>"):
            return self._chatml_chat_prompt(messages)
        return self._generic_chat_prompt(messages)

    def _has_token(self, token: str) -> bool:
        try:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
        except Exception:
            return False
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        return unk_id is None or token_id != unk_id

    def _gemma4_chat_prompt(self, messages: list[dict]) -> str:
        """Gemma-4 ``<|turn>`` format (canonical template, text-only path).

        Mirrors the Google canonical chat template for the text, non-tooling
        case: a system turn (with ``<|think|>`` when thinking is enabled), one
        ``<|turn>{role}\\n...<turn|>\\n`` block per message, and a generation
        prompt that closes with ``<|turn>model\\n`` plus an empty thought channel
        in non-thinking mode.
        """
        enable_thinking = bool(self.template_kwargs.get("enable_thinking", False))
        tok = self.tokenizer
        parts = [tok.bos_token or ""]
        if enable_thinking or (messages and messages[0]["role"] in ("system", "developer")):
            head = ["<|turn>system\n"]
            if enable_thinking:
                head.append("<|think|>\n")
            if messages and messages[0]["role"] in ("system", "developer"):
                head.append(_message_text(messages[0]))
                messages = messages[1:]
            head.append("<turn|>\n")
            parts.append("".join(head))
        for m in messages:
            role = "model" if m["role"] == "assistant" else m["role"]
            parts.append(f"<|turn>{role}\n{_message_text(m)}<turn|>\n")
        parts.append("<|turn>model\n")
        if not enable_thinking:
            # Canonical non-thinking generation prompt: a closed empty thought
            # channel, after which the model emits its answer directly.
            parts.append("<|channel>thought\n<channel|>")
        return "".join(parts)

    def _gemma3_chat_prompt(self, messages: list[dict]) -> str:
        """Gemma-2/3 ``<start_of_turn>`` format."""
        tok = self.tokenizer
        parts = [tok.bos_token or ""]
        for m in messages:
            role = "model" if m["role"] == "assistant" else m["role"]
            parts.append(f"<start_of_turn>{role}\n{_message_text(m)}<end_of_turn>\n")
        parts.append("<start_of_turn>model\n")
        return "".join(parts)

    def _chatml_chat_prompt(self, messages: list[dict]) -> str:
        """ChatML ``<|role|>`` format."""
        tok = self.tokenizer
        parts = [tok.bos_token or ""]
        for m in messages:
            parts.append(f"<|{m['role']}|>\n{_message_text(m)}<|end|>\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def _generic_chat_prompt(self, messages: list[dict]) -> str:
        """Plain labeled fallback that works with any tokenizer."""
        tok = self.tokenizer
        parts = [tok.bos_token or ""]
        for m in messages:
            label = {"system": "System", "user": "User", "assistant": "Assistant"}.get(
                m["role"], m["role"].capitalize()
            )
            parts.append(f"{label}: {_message_text(m)}\n\n")
        parts.append("Assistant:")
        return "".join(parts)


# ---------------------------------------------------------------------------
# OpenAI-compatible API backend
# ---------------------------------------------------------------------------
class APIBackend(LLMBackend):
    """Chat completions through any OpenAI-compatible endpoint.

    Credentials and endpoint are read from ``LLM_API_KEY`` / ``LLM_API_BASE_URL``
    environment variables (or constructor arguments).
    """

    name = "api"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.environ.get("LLM_API_MODEL", DEFAULT_MODEL)
        self.base_url = base_url or os.environ.get("LLM_API_BASE_URL")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def complete(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("The 'api' backend requires the 'openai' package: pip install openai") from e
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is required for the 'api' backend (or pass --api-key).")

        client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens or self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_backend(name: str, model: str = DEFAULT_MODEL, **kwargs) -> LLMBackend:
    """Build a backend by name (``hf``, ``api`` or ``stub``).

    ``kwargs`` are forwarded to the backend constructor; for the ``api`` backend a
    ``model`` key in ``kwargs`` overrides the default ``model`` argument.
    """
    name = name.lower()
    if name == "stub":
        return StubBackend(**kwargs)
    if name == "hf":
        return HFBackend(model_name=model, **kwargs)
    if name == "api":
        if "model" in kwargs:
            model = kwargs.pop("model") or model
        return APIBackend(model=model, **kwargs)
    raise ValueError(f"Unknown backend '{name}'. Choose from: hf, api, stub")
