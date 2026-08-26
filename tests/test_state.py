"""Tests for the implicit-state schema, formatting and parsing."""

import unittest

from state import (
    GREAT_EXP_MAX,
    MAX_FIELD_LEN,
    MAX_STATE_LEN,
    STATE_FIELDS,
    UNKNOWN,
    clean_summary_response,
    empty_state,
    format_user_state,
    needs_great_exp_summary,
    parse_user_state,
)


class TestState(unittest.TestCase):
    def test_empty_state_has_all_fields(self):
        state = empty_state()
        self.assertEqual(set(state), set(STATE_FIELDS))
        self.assertTrue(all(v == UNKNOWN for v in state.values()))
        self.assertEqual(len(STATE_FIELDS), 12)

    def test_format_user_state_renders_all_fields(self):
        state = empty_state()
        state["name"] = "Kanoa"
        state["age"] = "32"
        text = format_user_state(state)
        self.assertIn("**name**: Kanoa", text)
        self.assertIn("**age**: 32", text)
        self.assertIn("**Great_experience**: unknown", text)
        # 12 fields, each opening and closing with **
        self.assertEqual(text.count("**"), 24)

    def test_round_trip(self):
        state = {
            "name": "Kanoa Manu",
            "age": "32",
            "gender": "male",
            "location": "Honolulu",
            "preference": "likes fusion music",
            "occupation": "software engineer",
            "interest": "music production",
            "emotion": "excited",
            "objective": "share a recent experience",
            "knowledge": "advanced",
            "Great_experience": "produced a track",
            "character": "creative",
        }
        self.assertEqual(parse_user_state(format_user_state(state)), state)

    def test_parse_multiline_values(self):
        text = """**name**: Kanoa Manu
**age**: 32
**location**: a cozy and warm room
where it is raining outside
**occupation**: software engineer"""
        state = parse_user_state(text)
        self.assertEqual(state["name"], "Kanoa Manu")
        self.assertEqual(state["age"], "32")
        self.assertIn("cozy and warm room", state["location"])
        self.assertIn("raining outside", state["location"])
        self.assertEqual(state["occupation"], "software engineer")
        self.assertEqual(state["interest"], UNKNOWN)  # missing -> unknown
        self.assertEqual(state["Great_experience"], UNKNOWN)

    def test_parse_stray_quote_field(self):
        text = '**Great_experience**" : produced an electronic track'
        state = parse_user_state(text)
        self.assertEqual(state["Great_experience"], "produced an electronic track")

    def test_parse_with_dividers_and_header(self):
        text = """******************************
User Implicit State:
**name**: Kanoa Manu
**emotion**: a little excited, but also a little shy
******************************
Selected Candidate Response Identifier: (c)
******************************"""
        state = parse_user_state(text)
        self.assertEqual(state["name"], "Kanoa Manu")
        self.assertEqual(state["emotion"], "a little excited, but also a little shy")
        self.assertEqual(state["preference"], UNKNOWN)

    def test_parse_empty_and_none(self):
        self.assertEqual(parse_user_state(None), empty_state())
        self.assertEqual(parse_user_state(""), empty_state())
        self.assertEqual(parse_user_state("no state here"), empty_state())

    def test_parse_truncates_overlong_field(self):
        # a runaway field must be capped so the embedded state stays bounded
        tail = "y" * 500
        long = "x" * (MAX_FIELD_LEN + 500)
        state = parse_user_state(f"**preference**: {long}{tail}")
        self.assertEqual(len(state["preference"]), MAX_FIELD_LEN)
        self.assertTrue(state["preference"].endswith(tail), "keeps the most recent portion")

    def test_parse_caps_total_state_length(self):
        # even if the model runs away, the TOTAL embedded state must stay bounded
        long = "y" * MAX_STATE_LEN
        state = parse_user_state(f"**preference**: {long} **interest**: {long}")
        total = sum(len(v) for v in state.values())
        self.assertLessEqual(total, MAX_STATE_LEN)
        # the newest text (tail) of the surviving fields survives the trim
        self.assertLess(len(state["preference"]), MAX_STATE_LEN)
        self.assertTrue(state["interest"].endswith("y"))

    def test_great_exp_is_preserved_by_total_cap(self):
        # Great_experience is the one accumulating field: the total cap must trim
        # other fields first and leave it whole.
        ge = "y" * 600
        state = parse_user_state(
            f"**preference**: {'x' * 2000} **interest**: {'z' * 2000} **Great_experience**: {ge}"
        )
        self.assertEqual(state["Great_experience"], ge)
        self.assertLessEqual(sum(len(v) for v in state.values()), MAX_STATE_LEN)
        self.assertLess(len(state["preference"]), MAX_STATE_LEN)
        self.assertLess(len(state["interest"]), MAX_STATE_LEN)

    def test_great_exp_summary_trigger_threshold(self):
        state = empty_state()
        state["Great_experience"] = "x" * GREAT_EXP_MAX
        self.assertFalse(needs_great_exp_summary(state))
        state["Great_experience"] += "more"
        self.assertTrue(needs_great_exp_summary(state))

    def test_great_exp_hard_cap_and_custom_limit(self):
        # A single runaway output is bounded at min(MAX_STATE_LEN, 2 * limit)
        # (tail kept) and still flagged for condensation; the budget is tunable.
        long = "q" * 5000
        state = parse_user_state(f"**Great_experience**: {long}", great_exp_max=1500)
        self.assertEqual(len(state["Great_experience"]), min(MAX_STATE_LEN, 2 * 1500))
        self.assertTrue(state["Great_experience"].endswith("q"))
        self.assertTrue(needs_great_exp_summary(state, 1500))
        state2 = parse_user_state(f"**Great_experience**: {long}", great_exp_max=500)
        self.assertEqual(len(state2["Great_experience"]), min(MAX_STATE_LEN, 2 * 500))

    def test_clean_summary_response(self):
        self.assertEqual(clean_summary_response(None), UNKNOWN)
        self.assertEqual(clean_summary_response(""), UNKNOWN)
        # strips a leading field marker / divider / header
        text = "******************\n**Great_experience**: traveled to Japan in 2023\n******************"
        self.assertEqual(clean_summary_response(text, 500), "traveled to Japan in 2023")
        # truncates to max_chars (keeps the tail / most recent)
        out = clean_summary_response("y" * 100, 50)
        self.assertEqual(len(out), 50)
        self.assertTrue(out.endswith("y"))
        # a full state-block echo is reduced to its Great_experience value
        block = format_user_state({
            "name": "K", "age": "1", "gender": "m", "location": "x",
            "preference": "y", "occupation": "z", "interest": "w", "emotion": "e",
            "objective": "o", "knowledge": "k", "Great_experience": "made music", "character": "c",
        })
        self.assertEqual(clean_summary_response(block, 500), "made music")

    def test_parse_unknown_extra_field_terminates_previous(self):
        text = """**name**: Kanoa
**favorite_color**: blue
**occupation**: engineer"""
        state = parse_user_state(text)
        self.assertEqual(state["name"], "Kanoa")
        self.assertEqual(state["occupation"], "engineer")


if __name__ == "__main__":
    unittest.main()
