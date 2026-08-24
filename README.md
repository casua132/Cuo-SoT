# LLM Personalization Enhancement: `cot` and `cot_opt`

Two techniques for improving LLM response personalization, evaluated on the
[PersonaMem-v1](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1) benchmark.

## Techniques

**`cot`** — *reason-then-answer.* Before selecting a response, the model is asked to
infer the user's **implicit state** — who the user is, how they feel, what they want —
from the full conversation history, then pick the best candidate response. The implicit
state acts as a structured chain-of-thought that grounds the selection.

**`cot_opt`** — *incremental state maintenance.* Instead of re-deriving the user's
implicit state from the entire history on every new query (which re-reads up to 1M tokens
each time), the state is **maintained across turns**: each user message updates the state
through a cheap `intent_induce` step, and the final query is answered directly from the
current state. At serving time this avoids re-reading the whole history on every turn.

On the offline benchmark, `cot_opt` is evaluated faithfully by walking the history once per
context, caching the intermediate states, and answering each question from the state at its
cut-off point.

## Repository layout

```
.
├── prompt/                       # LLM prompts (markdown templates)
│   ├── cot.md / cot_sys.md             # cot: reason state, then select
│   ├── cot_opt.md / cot_opt_sys.md     # cot_opt: select from a maintained state
│   └── intent_induce.md / intent_induce_sys.md  # cot_opt: fold new info into the state
├── benchmark/
│   ├── personaMem.py             # PersonaMemV1 evaluation harness
│   ├── questions_*.csv           # benchmark questions (32k / 128k / 1M)
│   └── shared_contexts_*.jsonl   # shared multi-turn conversation histories
├── utils.py                      # constants + generic helpers
├── prompts.py                    # prompt loading / rendering / input formatting
├── state.py                      # implicit-state schema, parse & format
├── parsing.py                    # answer extraction (a/b/c/d)
├── data.py                       # benchmark data loading & slicing
├── backend.py                    # inference backends: stub / hf / api
├── main.py                       # CLI entry point
├── data_download.py              # download benchmark files from Hugging Face
└── tests/                        # unit + integration tests (no model inference)
```

## Quick start

```bash
# 0. Optional: fetch the benchmark data into benchmark/
python data_download.py --size 32k

# 1. Dry run with a deterministic stub backend (no model, validates the plumbing)
python main.py --method cot_opt --backend stub --limit 5 --verbose

# 2. Real evaluation with a local transformers model
#    (use the instruction-tuned gemma-4 variant, which is what the model card
#    recommends for chat; the base model also works via the manual template)
python main.py --method cot --backend hf --model google/gemma-4-E4B-it --size 32k --output results.csv

# 3. Real evaluation through an OpenAI-compatible endpoint
LLM_API_KEY=... LLM_API_BASE_URL=... \
    python main.py --method cot_opt --backend api --model gemma-4-E4B --output results.csv

# 4. Run the test suite (pure Python, no model, no network)
python -m unittest discover -s tests -v
```

### CLI options

| Flag | Meaning |
| --- | --- |
| `--method cot\|cot_opt` | which solution to evaluate |
| `--size 32k\|128k\|1M` | benchmark context size |
| `--backend hf\|api\|stub` | inference backend |
| `--model NAME` | model id (default `google/gemma-4-E4B`) |
| `--limit N` | evaluate only the first N questions |
| `--seed-persona` | `cot_opt` only (ablation): fold the persona system messages into the state |
| `--no-cache` | `cot_opt` only: re-walk the state for every question |
| `--output PATH` | write per-question results as CSV |
| `--api-base-url`, `--api-key`, `--api-model` | settings for the `api` backend |
| `--max-new-tokens N`, `--verbose` | generation limit / progress output |

## Implicit state

The user's implicit state is a 12-field structured representation (defined in
`log.md` and mirrored by `STATE_FIELDS` in `state.py`):

`name, age, gender, location, preference, occupation, interest, emotion, objective,
knowledge, Great_experience, character`

Values are meant to be concrete and vivid (e.g. *"a little excited, but also a little shy"*)
and default to `unknown` when they cannot be inferred.

## Design notes (deviations from the initial draft)

The initial draft's design is sound; the implementation below fixes several correctness
and robustness issues:

1. **State carry-forward.** The original `intent_induce` prompt said *"Do not keep the
   user's previous state as the current state"*, which invites the model to *drop* stable
   facts (name, age, long-standing preferences) on every update — the exact failure mode an
   incremental-state method must avoid. The prompt now explicitly says to carry forward
   every attribute that remains valid and only change fields with new evidence.

2. **No persona leakage.** The dataset's `system` messages are the ground-truth user
   profile (e.g. *"Current user persona: Name: Kanoa Manu — a 32-year-old software
   engineer…"*). Feeding them to the model would let it answer the benchmark questions
   without actually reasoning about the user's implicit state. **Both** methods therefore
   evaluate on dialogue-only histories: `cot` renders `User`/`Assistant` turns only, and
   `cot_opt` updates its state from user turns only. The persona blocks can be included
   purely as an ablation via `--seed-persona`.

3. **State-walk caching.** Questions in PersonaMem share `shared_context_id` and differ
   only by `end_index`. The walk is therefore done once per context and checkpointed at
   every message index, so all questions reusing a context share the same inference
   (~2.5× fewer state-update calls on the 32k split, ~40× on 1M). Use a deterministic
   backend (temperature 0) for exactly reproducible results.

4. **Prompt templates.** `cot.md`'s output-format example used `{user_name}`-style
   placeholders that collide with template rendering; replaced with concrete example
   values. Fixed the `**Great_experience**"` typo. `cot_opt`/`cot` output-format examples
   now show a concrete `(c)` identifier.

5. **Backends.** Model inference is abstracted behind `backend.LLMBackend`:
   - `stub` — deterministic responses for tests and dry runs (no model, no network);
   - `hf` — local `transformers` inference, lazily imported;
   - `api` — any OpenAI-compatible endpoint.
   The original code imported a `transformers` model at module import time and used a
   non-standard loader/API; that is fixed and isolated.

   The `hf` backend also handles tokenizers with **no `chat_template`**. The Gemma-4
   family is one such case: the base `google/gemma-4-E4B` tokenizer has no
   `chat_template`, so `apply_chat_template` raises `ValueError`. When the tokenizer
   has no template, the backend builds the prompt manually in the canonical Gemma-4
   format (`<|turn>…<turn|>` turns, `<|think|>` thinking gate, empty
   `<|channel>thought\n<channel|>` block on the generation prompt), and falls back to
   Gemma-2/3, ChatML, or a plain labeled format for other template-less models.
   `tests/test_backend.py` pins the exact rendered prompts. Note that the
   instruction-tuned **`google/gemma-4-E4B-it`** ships the official template and is
   what the model card recommends for chat; prefer it over the base model for this
   evaluation.

6. **Answer extraction** is robust to the two output formats (state + identifier for
   `cot`; bare identifier for `cot_opt`) and to free-form model output.

## Testing without expensive model runs

The test suite in `tests/` is pure Python — it exercises prompt rendering, state/answer
parsing, data loading, and full end-to-end runs with the `stub` backend (verifying the
state-walk ordering and the cache behavior via backend call counts). No model is loaded and
no network is used:

```bash
python -m unittest discover -s tests -v
```

# Acknowledgements

- *Hongru Wang* (2023). **Cue-CoT: Chain-of-thought Prompting for Responding to In-depth Dialogue Questions with LLMs**. 
[![arXiv](https://img.shields.io/badge/arXiv-2305.11792-b31b1b.svg)](https://arxiv.org/abs/2305.11792)
- *Bowen Jiang* (2025). **Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale**. [![arXiv](https://img.shields.io/badge/arXiv-2504.14225-b31b1b.svg)](https://arxiv.org/abs/2504.14225)