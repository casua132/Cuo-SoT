"""End-to-end pipeline tests with a deterministic stub backend (no model, no network).

These verify that both methods run on real benchmark data, that the cot_opt
state-walk processes turns in order, and that the per-context cache actually
reuses state-walk inference across questions.
"""

import re
import unittest

from backend import StubBackend
from benchmark.personaMem import PersonaMemV1
from state import empty_state, format_user_state
from utils import BENCHMARK_DIR

DATA_PRESENT = (BENCHMARK_DIR / "questions_32k.csv").exists()

LIMIT = 10


@unittest.skipUnless(DATA_PRESENT, "benchmark 32k data not downloaded")
class TestPipelineStub(unittest.TestCase):
    def make_benchmark(self, backend, **kwargs):
        return PersonaMemV1(backend=backend, size="32k", **kwargs)

    # ------------------------------------------------------------------ basic runs
    def test_cot_runs_end_to_end(self):
        backend = StubBackend(answer_response="(a)")
        bench = self.make_benchmark(backend)
        summary = bench.evaluate_cot(limit=LIMIT)
        self.assertEqual(len(summary.results), LIMIT)
        self.assertTrue(0.0 <= summary.accuracy <= 1.0)
        for r in summary.results:
            self.assertEqual(r.predicted, "a")
            self.assertIn(r.correct_answer, "abcd")

    def test_cot_opt_runs_end_to_end(self):
        backend = StubBackend(answer_response="(b)")
        bench = self.make_benchmark(backend)
        summary = bench.evaluate_cot_opt(limit=LIMIT)
        self.assertEqual(len(summary.results), LIMIT)
        self.assertTrue(0.0 <= summary.accuracy <= 1.0)
        for r in summary.results:
            self.assertEqual(r.predicted, "b")

    def test_accuracy_matches_fixed_answer(self):
        backend = StubBackend(answer_response="(a)")
        bench = self.make_benchmark(backend)
        summary = bench.evaluate_cot(limit=LIMIT)
        questions = bench.data.load_questions(limit=LIMIT)
        expected = sum(1 for q in questions if q.correct_answer == "a") / len(questions)
        self.assertAlmostEqual(summary.accuracy, expected)

    def test_method_dispatch_and_aliases(self):
        backend = StubBackend(answer_response="(a)")
        bench = self.make_benchmark(backend)
        summary = bench.evaluate("cot", limit=3)
        self.assertEqual(len(summary.results), 3)
        # backward-compatible aliases return accuracy
        self.assertEqual(bench.cot(limit=3), summary.accuracy)

    # ------------------------------------------------------------------ state walk
    def test_cot_opt_state_walk_orders_turns(self):
        """The final state's objective must reflect the last user turn before the query."""

        def response_fn(messages):
            system = messages[0]["content"]
            if "psychological expert" not in system:
                return "(a)"
            user = messages[1]["content"]
            if "User message:" in user:
                marker = user.split("User message:", 1)[1].strip()[:40]
            elif "Updated user profile:" in user:
                marker = "PROFILE"
            else:
                marker = "NONE"
            state = empty_state()
            state["objective"] = marker
            return format_user_state(state)

        backend = StubBackend(response_fn=response_fn)
        bench = self.make_benchmark(backend)
        bench.evaluate_cot_opt(limit=1)

        # Last call is the cot_opt answer call; it embeds the final rendered state
        # under the "User Implicit State:" input section (the template also contains a
        # schema line with the same label, so search only inside that section).
        user_prompt = backend.calls[-1][0][1]["content"]
        state_section = user_prompt.split("User Implicit State:", 1)[1]
        m = re.search(r"\*\*objective\*\*:\s*([^\n]*)", state_section)
        self.assertTrue(m, "objective not found in the rendered state block")
        final_objective = m.group(1).strip()

        question = bench.data.load_questions(limit=1)[0]
        msgs = bench.data.get_context_messages(question.shared_context_id, question.end_index)
        user_msgs = [msg for msg in msgs if msg["role"] == "user"]
        last_user = re.sub(r"^\s*(?:user|assistant|system)\s*:\s*", "",
                           user_msgs[-1]["content"], count=1, flags=re.I).strip()
        # parse_user_state strips trailing whitespace from values, so normalize the
        # expected marker the same way.
        self.assertEqual(final_objective, last_user[:40].strip())

    # ------------------------------------------------------------------ caching
    def test_cot_opt_cache_reuses_state_walk(self):
        backend = StubBackend(answer_response="(a)")
        bench = self.make_benchmark(backend)
        bench.evaluate_cot_opt(limit=LIMIT)
        calls_after_first = backend.call_count
        bench.evaluate_cot_opt(limit=LIMIT)
        calls_after_second = backend.call_count
        # Second run: only the LIMIT answer calls, no state-walk calls (cache hit).
        self.assertEqual(calls_after_second - calls_after_first, LIMIT)

    def test_cot_opt_without_cache_rebuilds(self):
        backend = StubBackend(answer_response="(a)")
        bench = self.make_benchmark(backend, cache=False)
        bench.evaluate_cot_opt(limit=LIMIT)
        questions = bench.data.load_questions(limit=LIMIT)
        expected = 0
        for q in questions:
            msgs = bench.data.get_context_messages(q.shared_context_id, q.end_index)
            expected += sum(1 for m in msgs if m["role"] == "user")
        expected += LIMIT  # answer calls
        self.assertEqual(backend.call_count, expected)

    def test_cot_opt_excludes_persona_by_default(self):
        backend = StubBackend(answer_response="(a)")
        bench = self.make_benchmark(backend)  # seed_persona defaults to False
        bench.evaluate_cot_opt(limit=LIMIT)
        questions = bench.data.load_questions(limit=LIMIT)
        contexts_seen = set()
        state_calls = 0
        for q in questions:
            if q.shared_context_id in contexts_seen:
                continue
            contexts_seen.add(q.shared_context_id)
            msgs = bench.data.contexts()[q.shared_context_id]
            state_calls += sum(1 for m in msgs if m["role"] == "user")
        self.assertEqual(backend.call_count, state_calls + LIMIT)

    def test_cot_opt_seed_persona_includes_persona(self):
        backend = StubBackend(answer_response="(a)")
        bench = self.make_benchmark(backend, seed_persona=True)  # ablation path
        bench.evaluate_cot_opt(limit=LIMIT)
        questions = bench.data.load_questions(limit=LIMIT)
        contexts_seen = set()
        state_calls = 0
        for q in questions:
            if q.shared_context_id in contexts_seen:
                continue
            contexts_seen.add(q.shared_context_id)
            msgs = bench.data.contexts()[q.shared_context_id]
            state_calls += sum(1 for m in msgs if m["role"] in ("user", "system"))
        self.assertEqual(backend.call_count, state_calls + LIMIT)

    def test_cot_prompt_excludes_persona_system_messages(self):
        backend = StubBackend(answer_response="(a)")
        bench = self.make_benchmark(backend)
        bench.evaluate_cot(limit=1)
        # The cot prompt must not contain the ground-truth persona profile.
        user_prompt = backend.calls[0][0][1]["content"]
        self.assertNotIn("Current user persona", user_prompt)
        self.assertNotIn("Gender Identity", user_prompt)
        # but it must contain the dialogue turns.
        self.assertIn("User:", user_prompt)
        self.assertIn("Assistant:", user_prompt)


class TestCliStub(unittest.TestCase):
    def test_cli_stub_dry_run(self):
        from main import main

        rc = main(["--method", "cot", "--backend", "stub", "--limit", "3", "--size", "32k"])
        self.assertEqual(rc, 0)

    def test_cli_writes_results_csv(self):
        import os
        import tempfile

        from main import main

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "results.csv")
            rc = main(["--method", "cot_opt", "--backend", "stub", "--limit", "3",
                       "--output", out])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out))
            with open(out) as f:
                content = f.read()
            self.assertIn("question_id", content)
            self.assertIn("predicted", content)


if __name__ == "__main__":
    unittest.main()
