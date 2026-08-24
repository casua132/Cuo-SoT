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
    def _intent_induce_messages(self, prev_state: dict, new_information: str) -> list[dict]:
        """Build the ``intent_induce`` chat messages for one state update."""
        user_prompt = render(
            "intent_induce",
            user_previous_state=format_user_state(prev_state),
            new_information=new_information,
        )
        return [
            {"role": "system", "content": system_prompt("intent_induce")},
            {"role": "user", "content": user_prompt},
        ]

    @classmethod
    def _log_state_update(cls, prev_state, new_state, response, new_information,
                          context_id: str | None = None) -> None:
        """Verbose log of one state-update inference call (sequential or batched)."""
        prefix = "[state:intent_induce]"
        if context_id is not None:
            prefix += f" context={context_id}"
        print(f"{prefix} new_information={_truncate(new_information, 120)!r}")
        print(f"  raw output: {_truncate(response)}")
        changes = cls._state_delta(prev_state, new_state)
        print("  state delta:", " | ".join(changes) if changes else "(no fields changed)")

    def _update_state(self, prev_state: dict, new_information: str) -> dict:
        """Ask the model to fold ``new_information`` into the user's implicit state."""
        messages = self._intent_induce_messages(prev_state, new_information)
        response = self.backend.complete(messages, max_new_tokens=self.max_new_tokens)
        new_state = parse_user_state(response)
        if self._verbose:
            self._log_state_update(prev_state, new_state, response, new_information)
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

    def _prebuild_checkpoints_batched(self, context_ids: list[str], batch_size: int) -> None:
        """Walk several contexts in lockstep, batching ``intent_induce`` across them.

        Each context's walk is still strictly sequential — every state update
        depends on the previous state, so the updates of one context can never be
        batched with each other. But the walks of *different* contexts are fully
        independent, so the update for turn ``t`` of every context is computed in
        one ``complete_batch`` call. The result is stored into ``self._checkpoints``
        exactly like the sequential cache, so :meth:`_state_for` reuses it.

        Only used when ``cache`` is enabled; with ``cache=False`` the walks stay
        sequential (each question's context is short-lived and re-walked on demand),
        and only the answer-selection calls are batched. Contexts already present
        in ``self._checkpoints`` are skipped, so repeated evaluations reuse the
        cached walk.
        """
        context_ids = [cid for cid in context_ids if cid not in self._checkpoints]
        if not context_ids:
            return
        msgs_by_ctx = {cid: self.data.contexts()[cid] for cid in context_ids}
        max_len = max((len(msgs) for msgs in msgs_by_ctx.values()), default=0)
        states = {cid: empty_state() for cid in context_ids}
        checkpoints: dict[str, dict[int, dict]] = {cid: {} for cid in context_ids}
        batch_size = max(1, batch_size)
        for idx in range(max_len):
            # Collect the state updates needed at this turn across all contexts.
            updates: list[tuple[str, str]] = []  # (context_id, new_information)
            for cid, msgs in msgs_by_ctx.items():
                if idx >= len(msgs):
                    continue
                msg = msgs[idx]
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    if self.seed_persona:
                        updates.append((cid, f"Updated user profile:\n{content}"))
                elif role == "user":
                    query = _ROLE_PREFIX_RE.sub("", content, count=1).strip()
                    updates.append((cid, f"User message:\n{query}"))
                # assistant turns carry no new user state.
            for start in range(0, len(updates), batch_size):
                chunk = updates[start:start + batch_size]
                messages_list = [
                    self._intent_induce_messages(states[cid], new_information)
                    for cid, new_information in chunk
                ]
                responses = self.backend.complete_batch(
                    messages_list, max_new_tokens=self.max_new_tokens
                )
                for (cid, new_information), response in zip(chunk, responses):
                    prev = states[cid]
                    states[cid] = parse_user_state(response)
                    if self._verbose:
                        self._log_state_update(
                            prev, states[cid], response, new_information, context_id=cid
                        )
            for cid in context_ids:
                if idx < len(msgs_by_ctx[cid]):
                    checkpoints[cid][idx] = dict(states[cid])
        for cid in context_ids:
            self._checkpoints[cid] = checkpoints[cid]

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
    def _cot_messages(self, question) -> list[dict]:
        """Build the ``cot`` chat messages for one question.

        ``system`` persona messages are excluded: they contain the ground-truth
        profile, which would let the model answer without reasoning from the dialogue.
        """
        messages_ctx = self.data.get_context_messages(question.shared_context_id, question.end_index)
        dialogue = dialogue_messages(messages_ctx)
        user_prompt = render(
            "cot",
            conversations=format_conversation(dialogue),
            user_query=question.query,
            candidate_responses=format_candidates(question.all_options),
        )
        return [
            {"role": "system", "content": system_prompt("cot")},
            {"role": "user", "content": user_prompt},
        ]

    def _cot_opt_messages(self, state: dict, question) -> list[dict]:
        """Build the ``cot_opt`` chat messages for one question from its state."""
        user_prompt = render(
            "cot_opt",
            implicit_state=format_user_state(state),
            user_query=question.query,
            candidate_responses=format_candidates(question.all_options),
        )
        return [
            {"role": "system", "content": system_prompt("cot_opt")},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _complete_or_batch(backend, messages_list: list[list[dict]], batch_size: int,
                           max_new_tokens: int) -> list[str]:
        """Run inference for one chunk, preserving the single-inference path.

        ``batch_size == 1`` must go through ``complete`` (not ``complete_batch``)
        so that ``--batch None`` reproduces the original per-sample behavior
        exactly; larger batches go through ``complete_batch``.
        """
        if batch_size <= 1:
            return [backend.complete(messages_list[0], max_new_tokens=max_new_tokens)]
        return backend.complete_batch(messages_list, max_new_tokens=max_new_tokens)

    @staticmethod
    def _make_result(question, predicted: str | None, response: str) -> Result:
        return Result(
            question_id=question.question_id,
            question_type=question.question_type,
            topic=question.topic,
            correct_answer=question.correct_answer,
            predicted=predicted,
            correct=predicted == question.correct_answer,
            shared_context_id=question.shared_context_id,
            end_index=question.end_index,
            response=response,
        )

    # ------------------------------------------------------------------
    # Evaluation loops
    # ------------------------------------------------------------------
    def evaluate_cot(self, limit: int | None = None, verbose: bool = False,
                     batch_size: int = 1) -> _Summary:
        """Evaluate ``cot``.

        ``batch_size > 1`` groups the independent answer-selection calls into
        ``complete_batch`` calls of that size. ``batch_size == 1`` is the
        sequential single-inference path. Results are identical up to the
        floating-point caveat documented on
        :meth:`backend.LLMBackend.complete_batch`.
        """
        self._verbose = verbose
        results: list[Result] = []
        questions = self.data.load_questions(limit=limit)
        batch_size = max(1, batch_size)
        for start in range(0, len(questions), batch_size):
            chunk = questions[start:start + batch_size]
            messages_list = [self._cot_messages(q) for q in chunk]
            responses = self._complete_or_batch(
                self.backend, messages_list, batch_size, self.max_new_tokens
            )
            for j, (q, response) in enumerate(zip(chunk, responses), start=start):
                if verbose:
                    self._log_question_header(j + 1, len(questions), q)
                predicted = extract_answer(response)
                if verbose:
                    self._log_answer_call("cot", q, response, predicted)
                results.append(self._make_result(q, predicted, response))
                if verbose:
                    self._log_progress(j + 1, len(questions), predicted, q.correct_answer, results)
        return _Summary(results)

    def evaluate_cot_opt(self, limit: int | None = None, verbose: bool = False,
                         batch_size: int = 1) -> _Summary:
        """Evaluate ``cot_opt``.

        ``batch_size > 1`` batches the independent work: the answer-selection
        calls across questions, and — when the state-walk cache is enabled — the
        ``intent_induce`` updates of *different* contexts in lockstep (updates
        within one context stay sequential, since each update depends on the
        previous state). ``batch_size == 1`` is the sequential single-inference
        path.
        """
        self._verbose = verbose
        results: list[Result] = []
        questions = self.data.load_questions(limit=limit)
        batch_size = max(1, batch_size)
        if batch_size > 1 and self.cache:
            context_ids = sorted({q.shared_context_id for q in questions})
            self._prebuild_checkpoints_batched(context_ids, batch_size)
        for start in range(0, len(questions), batch_size):
            chunk = questions[start:start + batch_size]
            states = [self._state_for(q.shared_context_id, q.end_index) for q in chunk]
            messages_list = [
                self._cot_opt_messages(state, q) for state, q in zip(states, chunk)
            ]
            responses = self._complete_or_batch(
                self.backend, messages_list, batch_size, self.max_new_tokens
            )
            for j, (q, response) in enumerate(zip(chunk, responses), start=start):
                if verbose:
                    self._log_question_header(j + 1, len(questions), q)
                predicted = extract_answer(response)
                if verbose:
                    self._log_answer_call("cot_opt", q, response, predicted)
                results.append(self._make_result(q, predicted, response))
                if verbose:
                    self._log_progress(j + 1, len(questions), predicted, q.correct_answer, results)
        return _Summary(results)

    # Backward-compatible aliases returning the overall accuracy.
    def cot(self, limit: int | None = None) -> float:
        return self.evaluate_cot(limit=limit).accuracy

    def cot_opt(self, limit: int | None = None) -> float:
        return self.evaluate_cot_opt(limit=limit).accuracy

    def evaluate(self, method: str, limit: int | None = None, verbose: bool = False,
                 batch_size: int = 1) -> _Summary:
        method = method.lower()
        if method == "cot":
            return self.evaluate_cot(limit=limit, verbose=verbose, batch_size=batch_size)
        if method == "cot_opt":
            return self.evaluate_cot_opt(limit=limit, verbose=verbose, batch_size=batch_size)
        raise ValueError(f"Unknown method '{method}'. Choose from: cot, cot_opt")

    @staticmethod
    def _log_progress(i: int, n: int, predicted: str | None, expected: str, results: list[Result]) -> None:
        correct = sum(r.correct for r in results)
        print(f"[{i}/{n}] predicted={predicted!r} expected={expected!r} "
              f"running_accuracy={correct / i:.4f}")
