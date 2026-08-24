"""Download the PersonaMem benchmark data files into ``benchmark/`` if they are missing.

Usage
-----
    python data_download.py                          # v1, all sizes
    python data_download.py --size 32k               # v1, one size
    python data_download.py --benchmark v2 --size 32k  # PersonaMem-v2, one size
    python data_download.py --force                  # re-download even if present

PersonaMem-v2 is not stored in the same layout as v1: it ships one
``benchmark/text/benchmark.csv`` plus per-persona ``chat_history_{size}`` JSON
files. ``data_download`` downloads those raw files and **converts** them into v1's
on-disk format (``questions_{size}.csv`` + ``shared_contexts_{size}.jsonl``) under
``benchmark/v2/``, so the whole evaluation pipeline is shared unchanged.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from utils import (
    BENCHMARK_DIR,
    BENCHMARK_NAMES,
    BENCHMARK_SIZES,
    BENCHMARK_SIZES_BY_BENCHMARK,
    CONTEXT_FILENAME,
    QUESTION_FILENAME,
    benchmark_dir,
    context_jsonl_path,
    question_csv_path,
)

REPO_ID_V1 = "bowen-upenn/PersonaMem-v1"
REPO_ID_V2 = "bowen-upenn/PersonaMem-v2"
REPO_TYPE = "dataset"

FILES = {size: (f"questions_{size}.csv", f"shared_contexts_{size}.jsonl") for size in BENCHMARK_SIZES}

V2_BENCHMARK_CSV = "benchmark/text/benchmark.csv"


# ---------------------------------------------------------------------------
# PersonaMem-v2 conversion into the v1 on-disk format
# ---------------------------------------------------------------------------
def _parse_literal(s: str):
    """Safe literal-eval of a ``user_query`` dict or ``incorrect_answers`` list."""
    return ast.literal_eval(s)


def _shuffle_options(question_id: str, correct: str, incorrects: list[str]):
    """Deterministically shuffle ``[correct] + incorrects``.

    Returns ``(all_options, correct_index)``. The shuffle is seeded by the stable
    ``question_id`` so repeated conversions (and repeated evaluations on the same
    files) always place the correct answer at the same position.
    """
    rng = random.Random(question_id)
    order = list(range(1 + len(incorrects)))
    rng.shuffle(order)
    all_options = [correct] + list(incorrects)
    correct_index = order.index(0)
    return [all_options[i] for i in order], correct_index


def convert_v2(raw_dir: Path, size: str, out_dir: Path) -> tuple[int, int]:
    """Convert a raw PersonaMem-v2 snapshot into v1-format files.

    ``raw_dir`` must contain ``benchmark/text/benchmark.csv`` and
    ``data/chat_history_{size}/*.json`` (the output of
    :func:`hf_hub_snapshot_download` with the right ``allow_patterns``).

    Returns ``(n_converted_questions, n_skipped_rows_without_context)`` and writes
    ``questions_{size}.csv`` + ``shared_contexts_{size}.jsonl`` into ``out_dir``.

    Mapping from v2 columns to v1's questions CSV (the subset :mod:`data` reads):

    * ``persona_id`` / ``shared_context_id``  <- v2 ``persona_id``
    * ``question_type``                       <- v2 ``pref_type`` (e.g. ``anti_stereotypical_pref``)
    * ``topic``                               <- v2 ``topic_query`` (e.g. ``Health``)
    * ``user_question_or_message``            <- v2 ``user_query.content`` (query is *not* in history)
    * ``all_options`` / ``correct_answer``    <- v2 ``correct_answer`` + ``incorrect_answers``,
                                                 deterministically shuffled
    * ``end_index_in_shared_context``         <- len(chat_history) (the whole history precedes the query)

    The chat-history JSONL keeps the leading ``system`` persona message; the
    evaluation layer's ``dialogue_messages()`` strips it, preserving the
    no-persona-leakage rule.

    Only the chat-history files *referenced by* ``benchmark.csv`` are converted —
    the repo contains extra ``chat_history_{size}`` files whose persona_ids collide
    with the referenced ones, and picking them up would silently swap in the wrong
    history for a persona.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- questions first: they define which chat histories are referenced
    csv_path = raw_dir / V2_BENCHMARK_CSV
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    referenced_names = {
        Path(r[f"chat_history_{size}_link"]).name for r in rows if r.get(f"chat_history_{size}_link")
    }

    # ---- contexts: one JSONL line per referenced persona -> {persona_id: [messages]}
    history_dir = raw_dir / "data" / f"chat_history_{size}"
    contexts: dict[str, list[dict]] = {}
    for fp in sorted(history_dir.glob("*.json")):
        if fp.name not in referenced_names:
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        persona_id = str(data.get("metadata", {}).get("persona_id", fp.stem))
        contexts[persona_id] = data.get("chat_history", [])
    ctx_path = out_dir / CONTEXT_FILENAME.format(size=size)
    with open(ctx_path, "w", encoding="utf-8") as f:
        for persona_id in sorted(contexts):
            f.write(json.dumps({persona_id: contexts[persona_id]}) + "\n")

    # ---- questions: one CSV row per benchmark row
    q_rows, skipped = [], 0
    for i, r in enumerate(rows):
        persona_id = r["persona_id"]
        if persona_id not in contexts:
            skipped += 1
            continue
        user_query = _parse_literal(r["user_query"])
        query = user_query["content"]
        correct = r["correct_answer"]
        incorrects = _parse_literal(r["incorrect_answers"])
        question_id = f"{persona_id}:{i}"
        options, correct_index = _shuffle_options(question_id, correct, incorrects)
        q_rows.append({
            "persona_id": persona_id,
            "question_id": question_id,
            "question_type": r["pref_type"],
            "topic": r["topic_query"],
            "user_question_or_message": query,
            "correct_answer": f"({chr(97 + correct_index)})",
            "all_options": repr(options),
            "shared_context_id": persona_id,
            "end_index_in_shared_context": len(contexts[persona_id]),
        })
    q_path = out_dir / QUESTION_FILENAME.format(size=size)
    with open(q_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(q_rows[0].keys()))
        writer.writeheader()
        writer.writerows(q_rows)
    return len(q_rows), skipped


def download_v2(size: str, force: bool = False) -> list[str]:
    """Download PersonaMem-v2 for ``size`` and convert it into v1-format files."""
    raw_files = (question_csv_path(size, "v2"), context_jsonl_path(size, "v2"))
    if all(p.exists() for p in raw_files) and not force:
        print(f"[data_download] v2 {size} already converted, skipping (use --force to re-download)")
        return [str(p) for p in raw_files]

    out_dir = benchmark_dir("v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        print(f"[data_download] downloading PersonaMem-v2 {size} raw files ...")
        snapshot_download(
            repo_id=REPO_ID_V2,
            repo_type=REPO_TYPE,
            allow_patterns=[f"data/chat_history_{size}/*.json", V2_BENCHMARK_CSV],
            local_dir=str(raw),
        )
        n, skipped = convert_v2(raw, size, out_dir)
        print(f"[data_download] converted {n} questions "
              f"({skipped} rows skipped: persona context not downloaded) -> {out_dir}")
    return [str(p) for p in raw_files]


# ---------------------------------------------------------------------------
# PersonaMem-v1 (native layout, no conversion)
# ---------------------------------------------------------------------------
def _already_present(size: str, filename: str, benchmark: str) -> bool:
    path = question_csv_path(size, benchmark) if filename.startswith("questions") \
        else context_jsonl_path(size, benchmark)
    return path.exists()


def download(size: str | None = None, force: bool = False, benchmark: str = "v1") -> list[str]:
    """Download the benchmark files for ``size`` (or all sizes) into ``benchmark/``.

    Returns the list of paths downloaded (or skipped).
    """
    if benchmark == "v2":
        sizes = [size] if size else list(BENCHMARK_SIZES_BY_BENCHMARK["v2"])
        paths: list[str] = []
        for s in sizes:
            paths.extend(download_v2(s, force))
        return paths

    sizes = [size] if size else list(BENCHMARK_SIZES)
    paths = []
    for s in sizes:
        for filename in FILES[s]:
            if not force and _already_present(s, filename, benchmark):
                print(f"[data_download] {filename} already present, skipping")
                continue
            path = hf_hub_download(
                repo_id=REPO_ID_V1,
                filename=filename,
                repo_type=REPO_TYPE,
                local_dir=str(BENCHMARK_DIR),
            )
            paths.append(path)
            print(f"[data_download] downloaded {path}")
    if not paths:
        print("[data_download] nothing to do; all files present (use --force to re-download).")
    return paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Download PersonaMem benchmark data into benchmark/.")
    parser.add_argument("--benchmark", choices=BENCHMARK_NAMES, default="v1",
                        help="which benchmark to download (v1 native, v2 converted into v1 format)")
    parser.add_argument("--size", default=None, help="only this size (default: all sizes of the benchmark)")
    parser.add_argument("--force", action="store_true", help="re-download even if already present")
    args = parser.parse_args(argv)

    if args.size and args.size not in BENCHMARK_SIZES_BY_BENCHMARK[args.benchmark]:
        parser.error(f"--size {args.size} is not available for benchmark {args.benchmark} "
                     f"(choose from {', '.join(BENCHMARK_SIZES_BY_BENCHMARK[args.benchmark])})")
    download(size=args.size, force=args.force, benchmark=args.benchmark)


if __name__ == "__main__":
    main()
