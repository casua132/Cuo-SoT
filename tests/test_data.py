"""Tests for benchmark data loading and slicing (uses the real 32k files)."""

import unittest

from data import BenchmarkData, dialogue_messages
from utils import BENCHMARK_DIR

DATA_PRESENT = (BENCHMARK_DIR / "questions_32k.csv").exists()


@unittest.skipUnless(DATA_PRESENT, "benchmark 32k data not downloaded")
class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = BenchmarkData(size="32k")
        cls.questions = cls.data.load_questions(limit=20)

    def test_questions_loaded(self):
        self.assertEqual(len(self.questions), 20)
        q = self.questions[0]
        self.assertIn(q.correct_answer, "abcd")
        self.assertEqual(len(q.all_options), 4)
        self.assertGreater(q.end_index, 0)
        self.assertTrue(q.query)

    def test_all_options_nonempty(self):
        for q in self.questions:
            for opt in q.all_options:
                self.assertTrue(opt.strip())

    def test_contexts_loaded(self):
        contexts = self.data.contexts()
        self.assertGreaterEqual(len(contexts), 1)
        for msgs in contexts.values():
            self.assertTrue(msgs)
            self.assertEqual(msgs[0]["role"], "system")

    def test_get_context_messages_slices(self):
        q = self.questions[0]
        msgs = self.data.get_context_messages(q.shared_context_id, q.end_index)
        full = self.data.contexts()[q.shared_context_id]
        self.assertEqual(len(msgs), q.end_index)
        self.assertEqual(msgs, full[: q.end_index])

    def test_shared_context_reuse(self):
        contexts = self.data.contexts()
        for q in self.questions:
            self.assertIn(q.shared_context_id, contexts)

    def test_dialogue_messages_filters_persona(self):
        contexts = self.data.contexts()
        sid = next(iter(contexts))
        raw = self.data.get_context_messages(sid, 10)
        self.assertTrue(any(m["role"] == "system" for m in raw))  # persona present
        dialogue = dialogue_messages(raw)
        self.assertTrue(all(m["role"] in ("user", "assistant") for m in dialogue))
        self.assertFalse(any(m["role"] == "system" for m in dialogue))


if __name__ == "__main__":
    unittest.main()
