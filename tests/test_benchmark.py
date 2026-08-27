"""End-to-end pipeline tests with a deterministic stub backend (no model, no network).

These verify that both methods run on real benchmark data, that the cot_opt
state-walk processes turns in order, and that the per-context cache actually
reuses state-walk inference across questions.
"""

import re
import unittest

from backend import StubBackend
from benchmark.personaMem import PersonaMemV1, _update_schedule
from data import dialogue_messages
from prompts import GREAT_EXP_SUMMARIZE_MARKER
from state import GREAT_EXP_MAX, empty_state, format_user_state
from utils import BENCHMARK_DIR

DATA_PRESENT = (BENCHMARK_DIR / "questions_32k.csv").exists()

LIMIT = 10


class TestUpdateSchedule(unittest.TestCase):
    """The state-update schedule fires on the 1st, (1+N)-th, (1+2N)-th, ... user turn."""

    def test_every_1_is_original(self):
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        self.assertEqual(_update_schedule(msgs, 1, False), [0, 2])

    def test_every_n_fires_on_first_and_then_every_n(self):
        msgs = [{"role": "user", "content": f"u{i}"} for i in range(7)]
        self.assertEqual(_update_schedule(msgs, 2, False), [0, 2, 4, 6])
        self.assertEqual(_update_schedule(msgs, 3, False), [0, 3, 6])

    def test_assistant_never_fires_and_system_only_under_seed_persona(self):
        msgs = [
            {"role": "system", "content": "s0"},
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
        ]
        # u1 (idx3) is the 2nd user turn (no fire); u2 (idx4) is the 3rd (fire).
        self.assertEqual(_update_schedule(msgs, 2, False), [1, 4])
        self.assertEqual(_update_schedule(msgs, 2, True), [0, 1, 4])

    def test_every_is_clamped_to_1(self):
        msgs = [{"role": "user", "content": f"u{i}"} for i in range(3)]
        self.assertEqual(_update_schedule(msgs, 0, False), [0, 1, 2])


class _BatchSpyBackend(StubBackend):
    """Deterministic stub that also records the size of every complete_batch call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sizes: list[int] = []

    def complete_batch(self, messages_list, max_new_tokens=None):
        self.batch_sizes.append(len(messages_list))
        return super().complete_batch(messages_list, max_new_tokens=max_new_tokens)


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

    def test_cot_opt_update_every_throttles_state_updates(self):
        """update_every=N recomputes the state every N user turns; the turns since
        the last update are injected as short-term context into the answer call."""

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
        bench = self.make_benchmark(backend, update_every=3)
        bench.evaluate_cot_opt(limit=1)

        q = bench.data.load_questions(limit=1)[0]
        msgs = bench.data.get_context_messages(q.shared_context_id, q.end_index)
        user_turns = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
        update_idxs = _update_schedule(msgs, 3, False)
        last_update = update_idxs[-1] if update_idxs else -1
        recent = dialogue_messages(msgs[last_update + 1:])

        # State-update calls are throttled to the scheduled update turns.
        state_calls = [c for c in backend.calls
                       if "psychological expert" in c[0][0]["content"]]
        self.assertEqual(len(state_calls), len(update_idxs))
        if len(user_turns) > 1:
            self.assertLess(len(state_calls), len(user_turns))

        # The stale state reflects the last update turn, not the last user turn.
        user_prompt = backend.calls[-1][0][1]["content"]
        if update_idxs:
            last_msg = msgs[last_update]
            last_marker = re.sub(r"^\s*(?:user|assistant|system)\s*:\s*", "",
                                 last_msg.get("content", ""), count=1, flags=re.I).strip()[:40]
            state_section = user_prompt.split("User Implicit State:", 1)[1]
            m = re.search(r"\*\*objective\*\*\s*:\s*([^\n]*)", state_section)
            self.assertTrue(m, "objective not found in the rendered state block")
            self.assertEqual(m.group(1).strip(), last_marker,
                             "state must hold the last UPDATE turn's snapshot")

        # Turns since the last update are injected as short-term memory.
        if recent:
            self.assertIn("# Recent Conversation (since the last state update)", user_prompt)
            for m in recent:
                content = re.sub(r"^\s*(?:user|assistant|system)\s*:\s*", "",
                                 m.get("content", ""), count=1, flags=re.I).strip()
                self.assertIn(content, user_prompt)
        else:
            self.assertNotIn("# Recent Conversation", user_prompt)

    def test_cot_opt_update_every_1_is_original(self):
        """update_every=1 reproduces the original behaviour: one update per user
        turn and no recent-context section in the answer prompt."""
        backend = StubBackend(answer_response="(a)")
        bench = self.make_benchmark(backend, update_every=1)
        bench.evaluate_cot_opt(limit=1)

        q = bench.data.load_questions(limit=1)[0]
        msgs = bench.data.get_context_messages(q.shared_context_id, q.end_index)
        user_turns = sum(1 for m in msgs if m.get("role") == "user")
        state_calls = [c for c in backend.calls
                       if "psychological expert" in c[0][0]["content"]]
        self.assertEqual(len(state_calls), user_turns)
        self.assertNotIn("# Recent Conversation", backend.calls[-1][0][1]["content"])

    def test_cot_opt_carries_forward_unchanged_fields(self):
        """A field the model marks 'unchanged' keeps its previous value in the state;
        the 'unchanged' sentinel must never leak into the stored state."""

        calls = {"n": 0}

        def response_fn(messages):
            system = messages[0]["content"]
            if "psychological expert" not in system:
                return "(a)"
            calls["n"] += 1
            state = empty_state()
            if calls["n"] == 1:
                state["name"] = "Kanoa"      # first turn establishes the name
            else:
                state["name"] = "unchanged"  # later turns: no evidence of change
            state["objective"] = f"turn{calls['n']}"
            return format_user_state(state)

        backend = StubBackend(response_fn=response_fn)
        bench = self.make_benchmark(backend)
        bench.evaluate_cot_opt(limit=1)

        # Last call is the cot_opt answer call; it embeds the final rendered state.
        user_prompt = backend.calls[-1][0][1]["content"]
        state_section = user_prompt.split("User Implicit State:", 1)[1]
        m = re.search(r"\*\*name\*\*\s*:\s*([^\n]*)", state_section)
        self.assertTrue(m, "name not found in the rendered state block")
        self.assertEqual(m.group(1).strip(), "Kanoa",
                         "'unchanged' must resolve to the previous value, not be stored")

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

    # ------------------------------------------------------------------ batching
    def test_cot_batched_matches_sequential(self):
        seq = StubBackend(answer_response="(a)")
        spy = _BatchSpyBackend(answer_response="(a)")
        s1 = self.make_benchmark(seq).evaluate_cot(limit=LIMIT)
        s2 = self.make_benchmark(spy).evaluate_cot(limit=LIMIT, batch_size=3)
        self.assertEqual([r.question_id for r in s1.results], [r.question_id for r in s2.results])
        self.assertEqual([r.predicted for r in s1.results], [r.predicted for r in s2.results])
        # Same number of logical inference calls, just delivered in batches.
        self.assertEqual(seq.call_count, spy.call_count)
        self.assertEqual(spy.call_count, LIMIT)
        self.assertTrue(spy.batch_sizes)
        self.assertEqual(max(spy.batch_sizes), 3)

    def test_cot_opt_batched_matches_sequential(self):
        seq = StubBackend(answer_response="(a)")
        spy = _BatchSpyBackend(answer_response="(a)")
        s1 = self.make_benchmark(seq).evaluate_cot_opt(limit=LIMIT)
        s2 = self.make_benchmark(spy).evaluate_cot_opt(limit=LIMIT, batch_size=2)
        self.assertEqual([r.question_id for r in s1.results], [r.question_id for r in s2.results])
        self.assertEqual([r.predicted for r in s1.results], [r.predicted for r in s2.results])
        # Identical logical inference count: cache walk + answer calls, batched.
        self.assertEqual(seq.call_count, spy.call_count)
        self.assertTrue(spy.batch_sizes)
        self.assertLessEqual(max(spy.batch_sizes), 2)

    def test_cot_opt_batched_cache_hit(self):
        spy = _BatchSpyBackend(answer_response="(a)")
        bench = self.make_benchmark(spy)
        bench.evaluate_cot_opt(limit=LIMIT, batch_size=2)
        calls_after_first = spy.call_count
        bench.evaluate_cot_opt(limit=LIMIT, batch_size=2)
        calls_after_second = spy.call_count
        # Second run reuses the cached walk: only the answer calls are re-inferred.
        self.assertEqual(calls_after_second - calls_after_first, LIMIT)

    def test_cot_opt_batched_without_cache_keeps_walk_sequential(self):
        spy = _BatchSpyBackend(answer_response="(a)")
        bench = self.make_benchmark(spy, cache=False)
        summary = bench.evaluate_cot_opt(limit=LIMIT, batch_size=2)
        self.assertEqual(len(summary.results), LIMIT)
        # cache off: the per-question state walks stay sequential, the answer
        # calls are batched.
        self.assertTrue(spy.batch_sizes)
        self.assertLessEqual(max(spy.batch_sizes), 2)

    # ------------------------------------------------------------------ Great_experience condensation
    def test_cot_opt_great_exp_summarization_fires(self):
        """An over-budget Great_experience is condensed by an extra call; no prompt
        ever embeds an uncondensed Great_experience above the budget, and the
        batched walk triggers the same calls as the sequential one."""

        def response_fn(messages):
            system = messages[0]["content"]
            if GREAT_EXP_SUMMARIZE_MARKER in system:
                return "condensed: traveled the world; built software; produced music"
            if "psychological expert" not in system:
                return "(a)"
            user = messages[1]["content"]
            prev_block = user.split("User previous state:", 1)[1].split("New information:", 1)[0]
            m = re.search(r"\*\*Great_experience\*\*\s*:\s*([^\n]*)", prev_block)
            prev_ge = m.group(1).strip() if m else ""
            new_ge = "x" * 300 if not prev_ge or prev_ge == "unknown" else prev_ge + "y" * 300
            state = empty_state()
            state["Great_experience"] = new_ge
            return format_user_state(state)

        seq = StubBackend(response_fn=response_fn)
        s1 = self.make_benchmark(seq).evaluate_cot_opt(limit=LIMIT)
        summarize_calls = [c for c in seq.calls if GREAT_EXP_SUMMARIZE_MARKER in c[0][0]["content"]]
        self.assertTrue(summarize_calls, "expected at least one Great_experience condensation call")

        # No prompt (state update or answer) embeds an uncondensed Great_experience.
        for messages, _ in seq.calls:
            if GREAT_EXP_SUMMARIZE_MARKER in messages[0]["content"]:
                continue  # the condensation call's input is the bloated value by design
            user = messages[1]["content"]
            section = (user.split("User previous state:", 1)[1]
                       if "User previous state:" in user
                       else user.split("User Implicit State:", 1)[1])
            m = re.search(r"\*\*Great_experience\*\*\s*:\s*([^\n]*)", section)
            if m:
                self.assertLessEqual(len(m.group(1)), GREAT_EXP_MAX,
                                     "an uncondensed Great_experience was embedded in a prompt")

        # The batched walk triggers the same condensation calls and results.
        spy = _BatchSpyBackend(response_fn=response_fn)
        s2 = self.make_benchmark(spy).evaluate_cot_opt(limit=LIMIT, batch_size=2)
        self.assertEqual([r.question_id for r in s1.results], [r.question_id for r in s2.results])
        self.assertEqual([r.predicted for r in s1.results], [r.predicted for r in s2.results])
        self.assertEqual(seq.call_count, spy.call_count)

    def test_cot_opt_default_stub_never_summarizes(self):
        """With Great_experience staying at 'unknown' the walk adds no condensation calls."""
        backend = StubBackend(answer_response="(a)")
        bench = self.make_benchmark(backend)
        bench.evaluate_cot_opt(limit=LIMIT)
        for messages, _ in backend.calls:
            self.assertNotIn(GREAT_EXP_SUMMARIZE_MARKER, messages[0]["content"])

    def test_batch_one_is_the_sequential_path(self):
        spy = _BatchSpyBackend(answer_response="(a)")
        bench = self.make_benchmark(spy)
        bench.evaluate_cot(limit=LIMIT, batch_size=1)
        # batch_size=1 never goes through complete_batch; complete() is used directly.
        self.assertEqual(spy.batch_sizes, [])
        self.assertEqual(spy.call_count, LIMIT)


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
