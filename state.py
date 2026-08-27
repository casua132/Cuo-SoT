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

# Sentinel a field carries in an ``intent_induce`` update when the new information
# does NOT change it. ``merge_state`` then keeps the previous value verbatim. This
# is distinct from ``UNKNOWN``, which means the field DID change but its new value
# cannot be determined (the old value no longer holds).
UNCHANGED = "unchanged"

# Accepted spellings of the carry-forward sentinel (checked against the whole value).
_UNCHANGED_MARKERS = {"unchanged", "no change", "[unchanged]"}


def is_unchanged(value) -> bool:
    """Whether a field value is the ``unchanged`` carry-forward sentinel."""
    return str(value).strip().lower() in _UNCHANGED_MARKERS

# Hard upper bound on any single state field. The full state is embedded into
# every intent_induce / cot_opt prompt, so an unbounded field would make the
# context grow every turn (and the KV cache with it). Values longer than this
# keep only their most recent (current) portion. This is a backstop on top of
# the prompt's "current state, never accumulate" instruction.
MAX_FIELD_LEN = 2000

# Hard upper bound on the TOTAL length of all state fields combined. The full
# state is embedded into every intent_induce / cot_opt prompt, so the total
# bounds the input length — and therefore the KV cache — no matter how the
# model behaves (a rewritten prompt does not guarantee the model stops
# accumulating). When the budget is exceeded, characters are trimmed from the
# currently-longest field until the total fits. ``Great_experience`` is trimmed
# only as a last resort (see :func:`_cap_state`). For gemma-4-E4B the KV cache
# is ~86 KB per token per batch row, so at a 37-context walk batch a 2500-char
# (≈625-token) state keeps the whole run ≈24–26 GB — well inside a 40 GB GPU.
MAX_STATE_LEN = 2500

# Character budget for the ``Great_experience`` field — the one field that is
# meant to ACCUMULATE the user's significant experiences rather than hold a
# current snapshot. When it would exceed this budget, the pipeline calls the LLM
# once to condense it (fact-preserving) instead of silently truncating it. When
# parsing a state output, the field is hard-capped at
# ``min(MAX_STATE_LEN, 2 * great_exp_max)`` so a single runaway output stays
# bounded; the caller then condenses anything above ``great_exp_max``. Tunable at
# runtime via ``main.py --great-exp-max``.
GREAT_EXP_MAX = 1500

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


def _cap_state(state: dict) -> dict:
    """Enforce the total state budget, preserving ``Great_experience``.

    Trims characters from the currently-longest field until the sum of all field
    lengths is within ``MAX_STATE_LEN``. ``Great_experience`` gets lowest
    priority: it is the one accumulating field (preserved whole up to its own
    hard cap, see :func:`parse_user_state`), so it is trimmed only when every
    other non-empty field has already been emptied.
    """
    total = sum(len(v) for v in state.values())
    if total <= MAX_STATE_LEN:
        return state
    over = total - MAX_STATE_LEN
    while over > 0:
        non_empty = [f for f in state if state[f]]
        if not non_empty:
            break
        # Prefer trimming non-Great_experience fields; fall back to Great_experience.
        candidates = [f for f in non_empty if f != "Great_experience"] or non_empty
        longest = max(candidates, key=lambda f: len(state[f]))
        cut = min(len(state[longest]), over)
        state[longest] = state[longest][:-cut]
        over -= cut
    return state


def needs_great_exp_summary(state: dict, great_exp_max: int = GREAT_EXP_MAX) -> bool:
    """Whether ``state``'s ``Great_experience`` exceeded its budget and needs condensing.

    The stored state keeps ``Great_experience`` at most ``great_exp_max`` chars
    (after a condensation call); anything above that has not been condensed yet.
    """
    return len(state.get("Great_experience", "")) > great_exp_max


def clean_summary_response(text: str | None, max_chars: int = GREAT_EXP_MAX) -> str:
    """Extract the condensed experience text from a summarization response.

    The condensation prompt asks for the text only, but a model may echo a field
    marker or a full state block; strip those and collapse whitespace. The result
    is truncated to the last ``max_chars`` characters (the most recent content)
    as a final backstop, or ``UNKNOWN`` when nothing survives.
    """
    if not text:
        return UNKNOWN
    # A full state-block echo is reduced to its Great_experience value.
    if len(list(_FIELD_MARKER_RE.finditer(text))) > 1:
        value = parse_user_state(text, great_exp_max=max_chars).get("Great_experience", UNKNOWN)
    else:
        m = _FIELD_MARKER_RE.search(text)
        if m and m.group(1) == "Great_experience":
            text = text[m.end():]
        value = _clean_value(text)
    if not value or value == UNKNOWN:
        return UNKNOWN
    if len(value) > max_chars:
        value = value[-max_chars:]
    return value


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


def parse_user_state(text: str | None, great_exp_max: int = GREAT_EXP_MAX) -> dict:
    """Parse a model-generated state block into a dict with the canonical fields.

    Values may span multiple lines; each value runs until the next ``**field**:``
    marker. Fields that are missing from the output (or cannot be read) are filled
    with ``unknown``. Unknown extra markers are ignored but still terminate the
    previous field's value.

    Values longer than their per-field cap are trimmed to their tail (the most
    recent, i.e. current, portion): other fields are capped at ``MAX_FIELD_LEN``;
    ``Great_experience`` — the one accumulating field — is hard-capped at
    ``min(MAX_STATE_LEN, 2 * great_exp_max)`` so a single runaway output stays
    bounded (never above the total budget), while the caller is expected to
    condense anything above ``great_exp_max`` (see
    :func:`needs_great_exp_summary`). The total across all fields is capped at
    ``MAX_STATE_LEN`` with ``Great_experience`` trimmed last.
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
            if value:
                cap = min(MAX_STATE_LEN, 2 * great_exp_max) if field == "Great_experience" else MAX_FIELD_LEN
                if len(value) > cap:
                    value = value[-cap:]
                state[field] = value
            else:
                state[field] = UNKNOWN
    return _cap_state(state)


def merge_state(prev: dict | None, new: dict) -> dict:
    """Merge an update into the previous state, applying only genuine changes.

    The model judges each field against the new information and writes one of
    three things: its updated current value (the field changed and the new value
    is determinable), the ``UNKNOWN`` sentinel (the field changed but its new
    value cannot be determined — the old value no longer holds), or the
    ``UNCHANGED`` sentinel (there is no evidence the field changed — its previous
    value is kept as-is). Fields carrying ``UNCHANGED`` keep the previous value
    verbatim; everything else — including a deliberate ``UNKNOWN`` — is applied.
    The total is re-capped so the merged state stays bounded.
    """
    prev = prev or {}
    merged = dict(new)
    for field in STATE_FIELDS:
        if is_unchanged(new.get(field, "")):
            merged[field] = prev.get(field, UNKNOWN)
    return _cap_state(merged)
