"""Tests for PersonaMem-v2 support: raw → v1-format conversion + end-to-end eval.

These build a small synthetic PersonaMem-v2 snapshot (a ``benchmark.csv`` plus a
couple of ``chat_history_{size}`` JSONs), convert it with :func:`data_download.convert_v2`
and evaluate the converted files with the stub backend. No network and no model.
"""

import csv
import json
import unittest
from pathlib import Path

from backend import StubBackend
from benchmark.personaMem import PersonaMemV1
from data import load_contexts, load_questions
from data_download import convert_v2

PERSONA_MARKER = "Current user persona: Name: Test Person"

# Minimal column set: only the fields convert_v2 reads are required.
CSV_FIELDNAMES = [
    "persona_id", "chat_history_32k_link", "chat_history_128k_link",
    "user_query", "correct_answer", "incorrect_answers",
    "pref_type", "topic_query",
]


def _chat_json(persona_id: str) -> dict:
    return {
        "metadata": {"persona_id": persona_id, "total_messages": 3},
        "chat_history": [
            {"role": "system", "content": PERSONA_MARKER},
            {"role": "user", "content": "What do you recommend? I like hiking."},
            {"role": "assistant", "content": "Try the mountain trail."},
        ],
    }


def build_raw_snapshot(root: Path) -> None:
    """Write a synthetic PersonaMem-v2 raw snapshot under ``root``."""
    history_dir = root / "data" / "chat_history_32k"
    history_dir.mkdir(parents=True)
    text_dir = root / "benchmark" / "text"
    text_dir.mkdir(parents=True)

    # p1 has 2 questions, p2 has 1, p3 has 1 but NO chat file -> must be skipped.
    rows = [
        {
            "persona_id": "p1",
            "chat_history_32k_link": "data/chat_history_32k/p1.json",
            "chat_history_128k_link": "data/chat_history_128k/p1.json",
            "user_query": "{'role': 'user', 'content': 'q1 for p1'}",
            "correct_answer": "Correct answer text one.",
            "incorrect_answers": "['Wrong a.', 'Wrong b.', 'Wrong c.']",
            "pref_type": "ask_to_forget",
            "topic_query": "Health",
        },
        {
            "persona_id": "p1",
            "chat_history_32k_link": "data/chat_history_32k/p1.json",
            "chat_history_128k_link": "data/chat_history_128k/p1.json",
            "user_query": "{'role': 'user', 'content': 'q2 for p1'}",
            "correct_answer": "Correct answer text two.",
            "incorrect_answers": "['Wrong d.', 'Wrong e.', 'Wrong f.']",
            "pref_type": "sensitive_info",
            "topic_query": "Work",
        },
        {
            "persona_id": "p2",
            "chat_history_32k_link": "data/chat_history_32k/p2.json",
            "chat_history_128k_link": "data/chat_history_128k/p2.json",
            "user_query": "{'role': 'user', 'content': 'q1 for p2'}",
            "correct_answer": "Correct answer text three.",
            "incorrect_answers": "['Wrong g.', 'Wrong h.', 'Wrong i.']",
            "pref_type": "neutral_preferences",
            "topic_query": "Travel",
        },
        {
            "persona_id": "p3",
            "chat_history_32k_link": "data/chat_history_32k/p3.json",
            "chat_history_128k_link": "data/chat_history_128k/p3.json",
            "user_query": "{'role': 'user', 'content': 'q1 for p3'}",
            "correct_answer": "Correct answer text four.",
            "incorrect_answers": "['Wrong j.', 'Wrong k.', 'Wrong l.']",
            "pref_type": "therapy_background",
            "topic_query": "Music",
        },
    ]
    with open(text_dir / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    (history_dir / "p1.json").write_text(json.dumps(_chat_json("p1")), encoding="utf-8")
    (history_dir / "p2.json").write_text(json.dumps(_chat_json("p2")), encoding="utf-8")


def converted_out(root: Path) -> Path:
    out = root / "out"
    build_raw_snapshot(root / "raw")
    convert_v2(root / "raw", "32k", out)
    return out


class TestV2Convert(unittest.TestCase):
    def test_converts_to_v1_format(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = converted_out(root)

            questions = load_questions(out / "questions_32k.csv")
            contexts = load_contexts(out / "shared_contexts_32k.jsonl")

            # p3 has no chat file -> its row is skipped; 3 of 4 rows convert.
            self.assertEqual(len(questions), 3)
            self.assertEqual(set(contexts), {"p1", "p2"})

            # The persona system message is kept on disk (the eval layer strips it).
            self.assertTrue(any(m["role"] == "system" and PERSONA_MARKER in m["content"]
                                for m in contexts["p1"]))

            for q in questions:
                # shared_context / end_index point at the persona's full history.
                self.assertIn(q.shared_context_id, contexts)
                self.assertEqual(q.end_index, len(contexts[q.shared_context_id]))
                # correct_answer is the letter at which the correct text landed.
                letter = chr(97 + q.all_options.index(next(
                    t for t in (["Correct answer text one.", "Correct answer text two.",
                                 "Correct answer text three."])
                    if t in q.all_options
                )))
                self.assertEqual(q.correct_answer, letter)
                self.assertEqual(len(q.all_options), 4)
                self.assertIn(q.query, ("q1 for p1", "q2 for p1", "q1 for p2"))

            by_type = {q.question_type for q in questions}
            self.assertEqual(by_type, {"ask_to_forget", "sensitive_info", "neutral_preferences"})
            self.assertEqual({q.topic for q in questions}, {"Health", "Work", "Travel"})

    def test_conversion_is_deterministic(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out1 = converted_out(root)
            out2 = root / "out2"
            convert_v2(root / "raw", "32k", out2)
            for name in ("questions_32k.csv", "shared_contexts_32k.jsonl"):
                self.assertEqual(
                    (out1 / name).read_bytes(), (out2 / name).read_bytes(),
                    f"{name} differs between two conversions",
                )


class TestV2EndToEndStub(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.out = converted_out(Path(self._tmp.name))
        self.q_path = self.out / "questions_32k.csv"
        self.c_path = self.out / "shared_contexts_32k.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def make_benchmark(self, backend, **kwargs):
        return PersonaMemV1(backend=backend, size="32k", benchmark="v2",
                            questions_path=self.q_path, contexts_path=self.c_path, **kwargs)

    def test_cot_runs_and_excludes_persona(self):
        backend = StubBackend(answer_response="(a)")
        bench = self.make_benchmark(backend)
        summary = bench.evaluate_cot(limit=3)
        self.assertEqual(len(summary.results), 3)
        self.assertTrue(0.0 <= summary.accuracy <= 1.0)
        for r in summary.results:
            self.assertEqual(r.predicted, "a")
            self.assertIn(r.correct_answer, "abcd")
        # No ground-truth persona leakage into the cot prompt.
        user_prompt = backend.calls[0][0][1]["content"]
        self.assertNotIn(PERSONA_MARKER, user_prompt)
        self.assertNotIn("Test Person", user_prompt)
        self.assertIn("User:", user_prompt)
        self.assertIn("Assistant:", user_prompt)

    def test_cot_opt_runs_with_cache(self):
        backend = StubBackend(answer_response="(b)")
        bench = self.make_benchmark(backend)
        summary = bench.evaluate_cot_opt(limit=3)
        self.assertEqual(len(summary.results), 3)
        for r in summary.results:
            self.assertEqual(r.predicted, "b")

        # Second run reuses the cached state walk: only answer calls are re-inferred.
        calls_after_first = backend.call_count
        bench.evaluate_cot_opt(limit=3)
        self.assertEqual(backend.call_count - calls_after_first, 3)

    def test_method_dispatch(self):
        bench = self.make_benchmark(StubBackend(answer_response="(a)"))
        summary = bench.evaluate("cot", limit=2)
        self.assertEqual(len(summary.results), 2)


class TestV2CliValidation(unittest.TestCase):
    def test_size_1M_rejected_for_v2(self):
        from main import parse_args

        with self.assertRaises(SystemExit):
            parse_args(["--benchmark", "v2", "--size", "1M", "--backend", "stub"])

    def test_valid_v2_size_parses(self):
        from main import parse_args

        args = parse_args(["--benchmark", "v2", "--size", "128k", "--backend", "stub"])
        self.assertEqual((args.benchmark, args.size), ("v2", "128k"))

    def test_data_download_rejects_1M_for_v2(self):
        from data_download import main as dd_main

        with self.assertRaises(SystemExit):
            dd_main(["--benchmark", "v2", "--size", "1M"])


if __name__ == "__main__":
    unittest.main()
