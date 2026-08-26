"""Tests for the implicit-state schema, formatting and parsing."""

import unittest

from state import MAX_FIELD_LEN, STATE_FIELDS, UNKNOWN, empty_state, format_user_state, parse_user_state


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

    def test_parse_unknown_extra_field_terminates_previous(self):
        text = """**name**: Kanoa
**favorite_color**: blue
**occupation**: engineer"""
        state = parse_user_state(text)
        self.assertEqual(state["name"], "Kanoa")
        self.assertEqual(state["occupation"], "engineer")


if __name__ == "__main__":
    unittest.main()
