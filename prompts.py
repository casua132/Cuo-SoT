"""Prompt-template loading, rendering and input-block formatting.

Each ``prompt/*.md`` file is a template with ``{placeholder}`` slots. ``render``
substitutes the provided values and fails loudly if a required slot is missing.
Templates may also contain literal example text with no ``{}`` slots, so leftover
braces are left untouched rather than treated as placeholders.
"""

from __future__ import annotations

import re

from utils import PROMPT_DIR

# Required placeholders per template name (used for validation).
_TEMPLATE_PLACEHOLDERS = {
    "cot": {"conversations", "user_query", "candidate_responses"},
    "cot_opt": {"implicit_state", "user_query", "candidate_responses"},
    "intent_induce": {"user_previous_state", "new_information"},
    "great_exp_summarize": {"great_experience", "max_chars"},
}

# System-prompt file that pairs with each task template.
_SYSTEM_TEMPLATE = {
    "cot": "cot_sys",
    "cot_opt": "cot_opt_sys",
    "intent_induce": "intent_induce_sys",
    "great_exp_summarize": "great_exp_summarize_sys",
}

# Distinctive phrase that appears verbatim in ``great_exp_summarize_sys.md``. The
# stub backend uses it to recognize condensation calls (so they are not confused
# with ``intent_induce`` state updates). It must not appear in any other prompt.
GREAT_EXP_SUMMARIZE_MARKER = "distilling a person's accumulated life experiences"

_cache: dict[str, str] = {}


def load_prompt(name: str) -> str:
    """Load a prompt template from the ``prompt/`` directory (cached)."""
    if name not in _cache:
        path = PROMPT_DIR / f"{name}.md"
        _cache[name] = path.read_text(encoding="utf-8")
    return _cache[name]


def render(template_name: str, **kwargs) -> str:
    """Render a template by substituting its ``{placeholders}``.

    Raises ``ValueError`` if a required placeholder is not provided.
    """
    template = load_prompt(template_name)
    required = _TEMPLATE_PLACEHOLDERS[template_name]
    missing = required - set(kwargs)
    if missing:
        raise ValueError(f"Missing placeholders for template '{template_name}': {sorted(missing)}")
    for key, value in kwargs.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def system_prompt(template_name: str) -> str:
    """Load the system prompt that pairs with a task template."""
    return load_prompt(_SYSTEM_TEMPLATE[template_name])


# ---------------------------------------------------------------------------
# Input-block formatting
# ---------------------------------------------------------------------------
_ROLE_PREFIX_RE = re.compile(r"^\s*(?:user|assistant|system)\s*:\s*", re.IGNORECASE)
_OPTION_PREFIX_RE = re.compile(r"^\s*[\(\[（]\s*[a-zA-Z]\s*[\)\]）]\.?\s*")


def _strip_role_prefix(content: str) -> str:
    """Remove a leading ``User:``/``Assistant:`` label from message content."""
    return _ROLE_PREFIX_RE.sub("", content, count=1).strip()


def _strip_option_prefix(option: str) -> str:
    """Remove a leading ``(a)``-style label from an option string."""
    return _OPTION_PREFIX_RE.sub("", option, count=1).strip()


_ROLE_LABELS = {"user": "User", "assistant": "Assistant", "system": "System"}


def format_conversation(messages: list[dict]) -> str:
    """Render chat messages into the ``{conversations}`` block.

    Every message is labelled by role (``User``/``Assistant``/``System``); a label
    already embedded in the content is not duplicated. System messages (persona
    profiles) are included so the model sees the full context.
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = _strip_role_prefix(msg.get("content", ""))
        label = _ROLE_LABELS.get(role, role)
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def format_candidates(options: list[str]) -> str:
    """Render candidate options into the ``{candidate_responses}`` block.

    Each option is labelled with its letter (``a``, ``b``, ...); a letter prefix
    already present in the option text is not duplicated.
    """
    lines = []
    for i, option in enumerate(options):
        letter = chr(ord("a") + i)
        text = _strip_option_prefix(option)
        lines.append(f"({letter}) {text}")
    return "\n".join(lines)
