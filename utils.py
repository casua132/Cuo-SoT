"""Shared constants and generic helpers for the cot / cot-opt personalization pipeline."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
PROMPT_DIR = ROOT_DIR / "prompt"
BENCHMARK_DIR = ROOT_DIR / "benchmark"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get("PERSONA_MEM_MODEL", "google/gemma-4-E4B")
DEFAULT_SIZE = "32k"
DEFAULT_METHOD = "cot"
DEFAULT_MAX_NEW_TOKENS = 1024

BENCHMARK_SIZES = ("32k", "128k", "1M")
METHODS = ("cot", "cot_opt")
BACKEND_NAMES = ("hf", "api", "stub")

QUESTION_FILENAME = "questions_{size}.csv"
CONTEXT_FILENAME = "shared_contexts_{size}.jsonl"


def question_csv_path(size: str = DEFAULT_SIZE) -> Path:
    """Path of the questions CSV for a benchmark size."""
    return BENCHMARK_DIR / QUESTION_FILENAME.format(size=size)


def context_jsonl_path(size: str = DEFAULT_SIZE) -> Path:
    """Path of the shared-contexts JSONL for a benchmark size."""
    return BENCHMARK_DIR / CONTEXT_FILENAME.format(size=size)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def parse_options(raw: str) -> list[str]:
    """Parse the `all_options` CSV field (a Python-list literal) into a list of strings.

    Falls back to a regex split if the literal cannot be parsed.
    """
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass

    # Fallback: split before each parenthesized option letter, e.g. "(a) ... (b) ...".
    parts = re.split(r"(?=\s*\(\s*[a-zA-Z]\s*\))", raw.strip())
    return [re.sub(r"^[\[\'\"]+|[\]\'\"]+$", "", p).strip() for p in parts if p.strip()]
