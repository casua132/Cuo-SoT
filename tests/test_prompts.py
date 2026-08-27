"""Tests for prompt-template loading/rendering and input-block formatting."""

import unittest

from prompts import (
    _TEMPLATE_PLACEHOLDERS,
    format_candidates,
    format_conversation,
    load_prompt,
    render,
    system_prompt,
)
from utils import PROMPT_DIR


class TestTemplates(unittest.TestCase):
    def test_all_template_files_exist(self):
        for name in _TEMPLATE_PLACEHOLDERS:
            self.assertTrue((PROMPT_DIR / f"{name}.md").exists(), f"{name}.md missing")
            self.assertTrue((PROMPT_DIR / f"{name}_sys.md").exists(), f"{name}_sys.md missing")

    def test_render_cot(self):
        out = render(
            "cot",
            conversations="User: hi\nAssistant: hello",
            user_query="what's up?",
            candidate_responses="(a) one\n(b) two",
        )
        self.assertIn("User: hi", out)
        self.assertIn("what's up?", out)
        self.assertIn("(a) one", out)
        self.assertIn("Kanoa Manu", out)  # example output preserved verbatim

    def test_render_missing_placeholder_raises(self):
        with self.assertRaises(ValueError):
            render("cot", conversations="x", user_query="y")

    def test_render_cot_opt(self):
        # recent_context="" is the update_every=1 case: byte-identical to the old
        # prompt (state, one blank line, then the query).
        out = render(
            "cot_opt",
            implicit_state="**name**: Kanoa",
            recent_context="",
            user_query="q",
            candidate_responses="(a) x",
        )
        self.assertIn("**name**: Kanoa", out)
        self.assertIn("q", out)
        self.assertNotIn("Recent Conversation", out)
        # a non-empty recent_context injects the short-term-memory block.
        out2 = render(
            "cot_opt",
            implicit_state="**name**: Kanoa",
            recent_context="\n\n# Recent Conversation (since the last state update)\n\nUser: hi\n\n",
            user_query="q",
            candidate_responses="(a) x",
        )
        self.assertIn("# Recent Conversation (since the last state update)", out2)
        self.assertIn("User: hi", out2)

    def test_render_intent_induce(self):
        out = render(
            "intent_induce",
            user_previous_state="**name**: unknown",
            new_information="User message:\nhello",
        )
        self.assertIn("**name**: unknown", out)
        self.assertIn("hello", out)

    def test_system_prompts(self):
        self.assertIn("response-selection", system_prompt("cot"))
        self.assertIn("response-selection", system_prompt("cot_opt"))
        self.assertIn("psychological expert", system_prompt("intent_induce"))

    def test_schema_consistent_across_templates(self):
        # every task template documents the same twelve fields
        for name in ("cot", "cot_opt", "intent_induce"):
            text = load_prompt(name)
            for field in ("name", "age", "gender", "location", "preference",
                          "occupation", "interest", "emotion", "objective",
                          "knowledge", "Great_experience", "character"):
                self.assertIn(f"**{field}**", text, f"{name}.md missing field {field}")


class TestFormatting(unittest.TestCase):
    def test_format_conversation_no_duplicate_labels(self):
        messages = [
            {"role": "system", "content": "Current user persona: Name: Kanoa"},
            {"role": "user", "content": "User: Hi there!"},
            {"role": "assistant", "content": "Assistant: Hello!"},
            {"role": "user", "content": "no label here"},  # missing prefix
        ]
        text = format_conversation(messages)
        self.assertEqual(
            text,
            "System: Current user persona: Name: Kanoa\n"
            "User: Hi there!\n"
            "Assistant: Hello!\n"
            "User: no label here",
        )

    def test_format_candidates_strips_and_relabels(self):
        options = ["(a) First", "Second", "(c) Third", "  (d) Fourth "]
        text = format_candidates(options)
        self.assertEqual(text, "(a) First\n(b) Second\n(c) Third\n(d) Fourth")

    def test_format_candidates_letter_order(self):
        options = ["x", "y", "z", "w"]
        text = format_candidates(options)
        self.assertEqual(text, "(a) x\n(b) y\n(c) z\n(d) w")


if __name__ == "__main__":
    unittest.main()
