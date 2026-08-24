"""User implicit state: schema definition, parsing and formatting.

The schema here mirrors the twelve fields used verbatim in the prompt files
(``prompt/cot.md``, ``prompt/cot_opt.md``, ``prompt/intent_induce.md``).
"""

from __future__ import annotations

import re

# Canonical field names, exactly as they appear in the prompt files.
STATE_FIELDS = [
    "name",
    "age",
    "gender",
    "location",
    "preference",
    "occupation",
    "interest",
    "emotion",
    "objective",
    "knowledge",
    "Great_experience",
    "character",
]

UNKNOWN = "unknown"

# Matches a field marker such as "**name**:", tolerating the stray-quote
# variants a model might emit ("**Great_experience**\" :").
_FIELD_MARKER_RE = re.compile(r"\*\*([A-Za-z_]+)\*\*\s*\"?\s*:")

_DIVIDER_RE = re.compile(r"^\**$")

# The cot output format appends the answer label after the state block; it is
# not part of any state value, so everything from this label onward is cut off.
_ANSWER_LABEL_RE = re.compile(r"Selected Candidate Response Identifier", re.IGNORECASE)


def empty_state() -> dict:
    """A state with every field set to ``unknown``."""
    return {field: UNKNOWN for field in STATE_FIELDS}


def is_known(value: str) -> bool:
    """Whether a state value carries real information (not blank/unknown)."""
    return bool(value) and value.strip().lower() != UNKNOWN


def format_user_state(state: dict | None) -> str:
    """Render a state dict into the ``**field**: value`` block used in the prompts."""
    state = state or empty_state()
    lines = [f"**{field}**: {state.get(field, UNKNOWN)}" for field in STATE_FIELDS]
    return "\n".join(lines)


def _clean_value(value: str) -> str:
    """Collapse a raw field value: drop divider/header lines, keep the rest."""
    kept = []
    for line in value.splitlines():
        s = line.strip()
        if not s:
            continue
        if _DIVIDER_RE.match(s) or s == "User Implicit State:":
            continue
        kept.append(s)
    return " ".join(kept).strip()


def parse_user_state(text: str | None) -> dict:
    """Parse a model-generated state block into a dict with the canonical fields.

    Values may span multiple lines; each value runs until the next ``**field**:``
    marker. Fields that are missing from the output (or cannot be read) are filled
    with ``unknown``. Unknown extra markers are ignored but still terminate the
    previous field's value.
    """
    state = empty_state()
    if not text:
        return state

    # Strip the trailing answer label that may follow a cot-format state block.
    text = _ANSWER_LABEL_RE.split(text, maxsplit=1)[0]

    markers = list(_FIELD_MARKER_RE.finditer(text))
    if not markers:
        return state

    for i, marker in enumerate(markers):
        field = marker.group(1)
        start = marker.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        if field in state:
            value = _clean_value(text[start:end])
            state[field] = value if value else UNKNOWN
    return state
