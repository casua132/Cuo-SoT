"""CLI entry point for evaluating the ``cot`` / ``cot_opt`` solutions on PersonaMem.

Examples
--------
# Dry run with a deterministic stub backend (no model, just plumbing):
python main.py --method cot_opt --backend stub --limit 5 --verbose

# Real evaluation with a local transformers model:
python main.py --method cot --backend hf --model google/gemma-4-E4B --size 32k --verbose

# Real evaluation through an OpenAI-compatible endpoint:
LLM_API_KEY=... LLM_API_BASE_URL=... \
    python main.py --method cot_opt --backend api --model gemma-4-E4B --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

from backend import create_backend
from benchmark.personaMem import PersonaMemV1
from utils import (
    BACKEND_NAMES,
    BENCHMARK_NAMES,
    BENCHMARK_SIZES,
    BENCHMARK_SIZES_BY_BENCHMARK,
    DEFAULT_BENCHMARK,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_METHOD,
    DEFAULT_MODEL,
    DEFAULT_SIZE,
    METHODS,
)


def _batch_size(value) -> int:
    """Parse ``--batch``: ``None``/``0``/``1`` → sequential; ``N > 1`` → batching."""
    if value is None:
        return 1
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("none", "", "null"):
            return 1
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("--batch must be None (sequential) or a positive integer")
    return n


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the cot / cot-opt personalization solutions on PersonaMem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD,
                        help="which solution to evaluate")
    parser.add_argument("--benchmark", choices=BENCHMARK_NAMES, default=DEFAULT_BENCHMARK,
                        help="which benchmark to evaluate (v1 = PersonaMem-v1, v2 = PersonaMem-v2)")
    parser.add_argument("--size", choices=BENCHMARK_SIZES, default=DEFAULT_SIZE,
                        help="benchmark context size")
    parser.add_argument("--backend", choices=BACKEND_NAMES, default="hf",
                        help="inference backend (hf=local transformers, api=OpenAI-compatible, stub=dry run)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name (hf or api backend)")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N questions (default: all)")
    parser.add_argument("--seed-persona", action="store_true",
                        help="cot_opt only (ablation): fold the system persona messages into the state. "
                             "Off by default because the persona blocks contain the ground-truth profile.")
    parser.add_argument("--no-cache", action="store_true",
                        help="cot_opt only: do not reuse the per-context state walk across questions")
    parser.add_argument("--batch", type=_batch_size, default=1,
                        help="hf backend: samples per generate() call. None/1 = sequential "
                             "single-inference (the default, exact per-sample results); "
                             "4/8 = batch that many independent samples per call "
                             "(faster, tiny fp differences near logit ties)")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--great-exp-max", type=int, default=None,
                        help="cot_opt only: character budget for the Great_experience field. "
                             "When an update would grow it past this, the field is condensed "
                             "by one LLM call (fact-preserving) instead of truncated. "
                             "Default: 1500. Lower it on a smaller GPU.")
    parser.add_argument("--output", default=None,
                        help="optional path to write per-question results as CSV")
    parser.add_argument("--api-base-url", default=None, help="api backend: base URL")
    parser.add_argument("--api-key", default=None, help="api backend: API key (or LLM_API_KEY)")
    parser.add_argument("--api-model", default=None, help="api backend: model override")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.size not in BENCHMARK_SIZES_BY_BENCHMARK[args.benchmark]:
        parser.error(f"--size {args.size} is not available for benchmark {args.benchmark} "
                     f"(choose from {', '.join(BENCHMARK_SIZES_BY_BENCHMARK[args.benchmark])})")
    return args


def write_results(path: str, summary) -> None:
    rows = [r.as_row() for r in summary.results]
    if not rows:
        print(f"[main] no results to write to {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[main] wrote {len(rows)} results to {path}")


def print_summary(summary) -> None:
    n = len(summary.results)
    print("=" * 60)
    print(f"Results over {n} question(s):")
    print(f"  overall accuracy: {summary.accuracy:.4f}")
    for qtype, info in sorted(summary.by_type().items()):
        print(f"  {qtype}: {info['correct']}/{info['total']} = {info['accuracy']:.4f}")
    print("=" * 60)


def main(argv=None) -> int:
    args = parse_args(argv)

    backend_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.backend == "api":
        if args.api_key:
            backend_kwargs["api_key"] = args.api_key
        if args.api_base_url:
            backend_kwargs["base_url"] = args.api_base_url
        if args.api_model:
            backend_kwargs["model"] = args.api_model
    backend = create_backend(args.backend, model=args.model, **backend_kwargs)

    benchmark = PersonaMemV1(
        backend=backend,
        size=args.size,
        benchmark=args.benchmark,
        seed_persona=args.seed_persona,
        cache=not args.no_cache,
        max_new_tokens=args.max_new_tokens,
        great_exp_max=args.great_exp_max,
    )

    print(f"[main] benchmark={args.benchmark} method={args.method} size={args.size} "
          f"backend={args.backend} model={args.model} seed_persona={args.seed_persona} "
          f"cache={not args.no_cache} batch={args.batch} "
          f"great_exp_max={benchmark.great_exp_max}")

    summary = benchmark.evaluate(args.method, limit=args.limit, verbose=args.verbose,
                                 batch_size=args.batch)
    print_summary(summary)

    if args.output:
        write_results(args.output, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
