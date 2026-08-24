"""Loading and slicing of the PersonaMem benchmark data (CSV questions + JSONL contexts)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from parsing import normalize_answer
from utils import context_jsonl_path, parse_options, question_csv_path


@dataclass
class Question:
    """One benchmark question: a query, its candidate options, and the ground truth."""

    persona_id: str
    question_id: str
    question_type: str
    topic: str
    query: str
    correct_answer: str  # normalized bare letter, e.g. "c"
    all_options: list[str]
    shared_context_id: str
    end_index: int


@dataclass
class Result:
    """Outcome of evaluating one question with one method."""

    question_id: str
    question_type: str
    topic: str
    correct_answer: str
    predicted: str | None
    correct: bool
    shared_context_id: str
    end_index: int
    response: str = ""  # raw model output of the answer call

    def as_row(self) -> dict:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "topic": self.topic,
            "correct_answer": self.correct_answer,
            "predicted": self.predicted or "",
            "correct": int(self.correct),
            "model_response": self.response,
            "shared_context_id": self.shared_context_id,
            "end_index_in_shared_context": self.end_index,
        }


def load_questions(path: str | Path, limit: int | None = None) -> list[Question]:
    """Parse the questions CSV into a list of ``Question`` objects."""
    questions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            questions.append(
                Question(
                    persona_id=row["persona_id"],
                    question_id=row["question_id"],
                    question_type=row["question_type"],
                    topic=row["topic"],
                    query=row["user_question_or_message"],
                    correct_answer=normalize_answer(row["correct_answer"]) or "",
                    all_options=parse_options(row["all_options"]),
                    shared_context_id=row["shared_context_id"],
                    end_index=int(row["end_index_in_shared_context"]),
                )
            )
    return questions


def dialogue_messages(messages: list[dict]) -> list[dict]:
    """Filter out ``system`` persona messages, keeping only dialogue turns.

    Persona system messages in this dataset contain the ground-truth user profile
    (e.g. ``Current user persona: Name: ...``). Feeding them to the model would let it
    answer the benchmark questions without actually reasoning about the user's
    implicit state, so both evaluation methods use dialogue-only histories.
    """
    return [m for m in messages if m.get("role") in ("user", "assistant")]


def load_contexts(path: str | Path) -> dict[str, list[dict]]:
    """Load the shared-contexts JSONL into ``{shared_context_id: [messages]}``."""
    contexts: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for key, messages in json.loads(line).items():
                contexts[key] = messages
    return contexts


class BenchmarkData:
    """Access to the questions and shared contexts for a given benchmark size."""

    def __init__(
        self,
        size: str = "32k",
        questions_path: str | Path | None = None,
        contexts_path: str | Path | None = None,
    ) -> None:
        self.size = size
        self.questions_path = Path(questions_path) if questions_path else question_csv_path(size)
        self.contexts_path = Path(contexts_path) if contexts_path else context_jsonl_path(size)
        self._contexts: dict[str, list[dict]] | None = None

    def load_questions(self, limit: int | None = None) -> list[Question]:
        return load_questions(self.questions_path, limit=limit)

    def contexts(self) -> dict[str, list[dict]]:
        if self._contexts is None:
            self._contexts = load_contexts(self.contexts_path)
        return self._contexts

    def get_context_messages(self, shared_context_id: str, end_index: int | None = None) -> list[dict]:
        """Return the message list for a context, optionally sliced to ``end_index``.

        Slicing follows the benchmark convention: ``context[:end_index]``.
        """
        messages = self.contexts()[shared_context_id]
        if end_index is not None:
            messages = messages[: int(end_index)]
        return messages
