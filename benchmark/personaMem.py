"""PersonaMem benchmark evaluation for the ``cot`` and ``cot_opt`` solutions.

Evaluation strategies
---------------------
``cot``
    For each question, the dialogue history (``context[:end_index]``, excluding
    ``system`` persona messages), the current query, and the candidate options are
    sent to the model with the ``cot`` prompt. The model reasons about the user's
    implicit state from the dialogue and selects one candidate. Accuracy = fraction
    of correctly selected options.

``cot_opt``
    The user's implicit state is maintained incrementally instead of being
    re-derived from the full history on every question. The dialogue history is
    walked once per ``shared_context_id``:

      * a ``user`` turn updates the state through the ``intent_induce`` prompt;
      * ``assistant`` turns carry no new user state and are skipped;
      * ``system`` persona messages are excluded by default (they encode the
        ground-truth profile and would let the model answer without reasoning).
        They can be included as an ablation via ``seed_persona=True``.

    The final query is then answered with the ``cot_opt`` prompt using the state
    reached at the question's ``end_index``. Because the walk is cached per context
    and checkpointed at every message index, all questions sharing a context reuse
    the same inference (questions only differ by their ``end_index`` cut-off).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend import LLMBackend
from data import BenchmarkData, Result, dialogue_messages
from parsing import extract_answer
from prompts import format_candidates, format_conversation, render, system_prompt
from state import STATE_FIELDS, empty_state, format_user_state, parse_user_state

_ROLE_PREFIX_RE = re.compile(r"^\s*(?:user|assistant|system)\s*:\s*", re.IGNORECASE)


def _truncate(text, cap: int = 1200) -> str:
    """Shorten a long string for logging, marking the truncation."""
    text = str(text)
    if len(text) <= cap:
        return text
    return f"{text[:cap]} ... [truncated, {len(text)} chars]"


@dataclass
class _Summary:
    results: list[Result]

    @property
    def accuracy(self) -> float:
        return sum(r.correct for r in self.results) / len(self.results) if self.results else 0.0

    def by_type(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in self.results:
            bucket = out.setdefault(r.question_type, [0, 0])
            bucket[1] += 1
            bucket[0] += int(r.correct)
        return {t: {"correct": c, "total": n, "accuracy": c / n} for t, (c, n) in out.items()}


class PersonaMemV1:
    """Evaluation harness over the PersonaMem-v1 benchmark data.

    Args:
        backend: an :class:`~backend.LLMBackend` instance used for all inference.
        size: benchmark context size, one of ``32k``/``128k``/``1M``.
        seed_persona: ablation only. When ``True``, the ``system`` persona messages
            are folded into the ``cot_opt`` state. Off by default because those
            messages contain the ground-truth profile and would let the model answer
            without reasoning from the dialogue.
        cache: whether to cache the per-context state walk so that questions
            sharing a context reuse the same inference. Requires a deterministic
            backend (use temperature 0) for exact results.
        max_new_tokens: generation limit passed to the backend.
        questions_path / contexts_path: optional explicit data-file paths
            (mainly used in tests).
    """

    def __init__(
        self,
        backend: LLMBackend,
        size: str = "32k",
        seed_persona: bool = False,
        cache: bool = True,
        max_new_tokens: int = 1024,
        questions_path=None,
        contexts_path=None,
    ) -> None:
        self.backend = backend
        self.seed_persona = seed_persona
        self.cache = cache
        self.max_new_tokens = max_new_tokens
        self.data = BenchmarkData(size=size, questions_path=questions_path, contexts_path=contexts_path)
        # shared_context_id -> {message_index: state}; one full-context walk per context.
        self._checkpoints: dict[str, dict[int, dict]] = {}
        # Toggled by evaluate_*; read by the per-call inference log.
        self._verbose = False

    # ------------------------------------------------------------------
    # Verbose inference log (per model call)
    # ------------------------------------------------------------------
    @staticmethod
    def _log_question_header(i: int, n: int, question) -> None:
        print("=" * 78)
        print(f"Question {i}/{n}  qid={question.question_id}  "
              f"type={question.question_type}  end_index={question.end_index}")
        print("=" * 78)

    @staticmethod
    def _log_answer_call(method: str, question, response: str, predicted) -> None:
        """Log one answer-selection model call: query, raw output, extraction."""
        print(f"[answer:{method}] query={_truncate(question.query, 120)!r}  "
              f"options={len(question.all_options)}")
        print(f"  raw output: {_truncate(response)}")
        print(f"  extracted: {predicted!r}   expected: {question.correct_answer!r}")

    @staticmethod
    def _state_delta(prev: dict, new: dict) -> list[str]:
        """Human-readable list of implicit-state fields that changed."""
        changes = []
        for key in STATE_FIELDS:
            old, cur = str(prev.get(key, "")).strip(), str(new.get(key, "")).strip()
            if old != cur:
                changes.append(f"{key}: {old!r} -> {cur!r}")
        return changes

    # ------------------------------------------------------------------
    # State maintenance (used by cot_opt)
    # ------------------------------------------------------------------
    def _update_state(self, prev_state: dict, new_information: str) -> dict:
        """Ask the model to fold ``new_information`` into the user's implicit state."""
        user_prompt = render(
            "intent_induce",
            user_previous_state=format_user_state(prev_state),
            new_information=new_information,
        )
        messages = [
            {"role": "system", "content": system_prompt("intent_induce")},
            {"role": "user", "content": user_prompt},
        ]
        response = self.backend.complete(messages, max_new_tokens=self.max_new_tokens)
        new_state = parse_user_state(response)
        if self._verbose:
            print(f"[state:intent_induce] new_information={_truncate(new_information, 120)!r}")
            print(f"  raw output: {_truncate(response)}")
            changes = self._state_delta(prev_state, new_state)
            print("  state delta:", " | ".join(changes) if changes else "(no fields changed)")
        return new_state

    def _build_checkpoints(self, messages: list[dict]) -> dict[int, dict]:
        """Walk a message list, checkpointing the state after every index.

        ``checkpoints[i]`` is the state after processing ``messages[:i+1]``, so the
        state at a question's ``end_index`` is ``checkpoints[end_index - 1]``.

        Only ``user`` turns update the state (and ``system`` persona messages when
        ``seed_persona`` is enabled as an ablation). ``assistant`` turns and, by
        default, ``system`` persona messages leave the state unchanged — the persona
        blocks encode the ground-truth profile and must not be handed to the model.
        """
        checkpoints: dict[int, dict] = {}
        state = empty_state()
        for idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                if self.seed_persona:
                    state = self._update_state(state, f"Updated user profile:\n{content}")
            elif role == "user":
                query = _ROLE_PREFIX_RE.sub("", content, count=1).strip()
                state = self._update_state(state, f"User message:\n{query}")
            # assistant turns carry no new user state.
            checkpoints[idx] = dict(state)
        return checkpoints

    def _state_for(self, shared_context_id: str, end_index: int) -> dict:
        """Return the implicit state reached at ``end_index`` for a context."""
        end_index = int(end_index)
        if end_index <= 0:
            return empty_state()
        if self.cache:
            checkpoints = self._checkpoints.get(shared_context_id)
            if checkpoints is None:
                messages = self.data.contexts()[shared_context_id]
                checkpoints = self._build_checkpoints(messages)
                self._checkpoints[shared_context_id] = checkpoints
        else:
            messages = self.data.get_context_messages(shared_context_id, end_index)
            checkpoints = self._build_checkpoints(messages)
        return checkpoints.get(end_index - 1, empty_state())

    # ------------------------------------------------------------------
    # Answer selection
    # ------------------------------------------------------------------
    def _select_cot(self, question) -> tuple[str | None, str]:
        """cot: reason the implicit state from the dialogue, then pick a candidate.

        ``system`` persona messages are excluded: they contain the ground-truth
        profile, which would let the model answer without reasoning from the dialogue.

        Returns ``(predicted_letter, raw_model_output)``.
        """
        messages_ctx = self.data.get_context_messages(question.shared_context_id, question.end_index)
        dialogue = dialogue_messages(messages_ctx)
        user_prompt = render(
            "cot",
            conversations=format_conversation(dialogue),
            user_query=question.query,
            candidate_responses=format_candidates(question.all_options),
        )
        messages = [
            {"role": "system", "content": system_prompt("cot")},
            {"role": "user", "content": user_prompt},
        ]
        response = self.backend.complete(messages, max_new_tokens=self.max_new_tokens)
        predicted = extract_answer(response)
        if self._verbose:
            self._log_answer_call("cot", question, response, predicted)
        return predicted, response

    def _select_cot_opt(self, state: dict, question) -> tuple[str | None, str]:
        """cot_opt: use the maintained implicit state directly to pick a candidate.

        Returns ``(predicted_letter, raw_model_output)``.
        """
        user_prompt = render(
            "cot_opt",
            implicit_state=format_user_state(state),
            user_query=question.query,
            candidate_responses=format_candidates(question.all_options),
        )
        messages = [
            {"role": "system", "content": system_prompt("cot_opt")},
            {"role": "user", "content": user_prompt},
        ]
        response = self.backend.complete(messages, max_new_tokens=self.max_new_tokens)
        predicted = extract_answer(response)
        if self._verbose:
            self._log_answer_call("cot_opt", question, response, predicted)
        return predicted, response

    # ------------------------------------------------------------------
    # Evaluation loops
    # ------------------------------------------------------------------
    def evaluate_cot(self, limit: int | None = None, verbose: bool = False) -> _Summary:
        self._verbose = verbose
        results: list[Result] = []
        questions = self.data.load_questions(limit=limit)
        for i, q in enumerate(questions):
            if verbose:
                self._log_question_header(i + 1, len(questions), q)
            predicted, response = self._select_cot(q)
            results.append(
                Result(
                    question_id=q.question_id,
                    question_type=q.question_type,
                    topic=q.topic,
                    correct_answer=q.correct_answer,
                    predicted=predicted,
                    correct=predicted == q.correct_answer,
                    shared_context_id=q.shared_context_id,
                    end_index=q.end_index,
                    response=response,
                )
            )
            if verbose:
                self._log_progress(i + 1, len(questions), predicted, q.correct_answer, results)
        return _Summary(results)

    def evaluate_cot_opt(self, limit: int | None = None, verbose: bool = False) -> _Summary:
        self._verbose = verbose
        results: list[Result] = []
        questions = self.data.load_questions(limit=limit)
        for i, q in enumerate(questions):
            if verbose:
                self._log_question_header(i + 1, len(questions), q)
            state = self._state_for(q.shared_context_id, q.end_index)
            predicted, response = self._select_cot_opt(state, q)
            results.append(
                Result(
                    question_id=q.question_id,
                    question_type=q.question_type,
                    topic=q.topic,
                    correct_answer=q.correct_answer,
                    predicted=predicted,
                    correct=predicted == q.correct_answer,
                    shared_context_id=q.shared_context_id,
                    end_index=q.end_index,
                    response=response,
                )
            )
            if verbose:
                self._log_progress(i + 1, len(questions), predicted, q.correct_answer, results)
        return _Summary(results)

    # Backward-compatible aliases returning the overall accuracy.
    def cot(self, limit: int | None = None) -> float:
        return self.evaluate_cot(limit=limit).accuracy

    def cot_opt(self, limit: int | None = None) -> float:
        return self.evaluate_cot_opt(limit=limit).accuracy

    def evaluate(self, method: str, limit: int | None = None, verbose: bool = False) -> _Summary:
        method = method.lower()
        if method == "cot":
            return self.evaluate_cot(limit=limit, verbose=verbose)
        if method == "cot_opt":
            return self.evaluate_cot_opt(limit=limit, verbose=verbose)
        raise ValueError(f"Unknown method '{method}'. Choose from: cot, cot_opt")

    @staticmethod
    def _log_progress(i: int, n: int, predicted: str | None, expected: str, results: list[Result]) -> None:
        correct = sum(r.correct for r in results)
        print(f"[{i}/{n}] predicted={predicted!r} expected={expected!r} "
              f"running_accuracy={correct / i:.4f}")
