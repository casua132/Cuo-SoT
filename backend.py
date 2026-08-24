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
class HFBackend(LLMBackend):
    """Local inference through ``transformers``.

    The model is loaded lazily on the first call so that importing this module
    never requires ``torch``/``transformers`` to be installed.
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
        import torch

        chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **self.template_kwargs,
        )
        if isinstance(chat, dict) and "input_ids" in chat:
            input_ids = chat["input_ids"]
            attention_mask = chat.get("attention_mask")
        else:
            input_ids = chat
            attention_mask = None

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
