"""Tests for answer and correct-answer extraction."""

import unittest

from parsing import extract_answer, normalize_answer


class TestExtractAnswer(unittest.TestCase):
    def test_parenthesized_letter(self):
        self.assertEqual(extract_answer("(c)"), "c")
        self.assertEqual(extract_answer("I choose (b)."), "b")
        self.assertEqual(extract_answer("(A)"), "a")
        self.assertEqual(extract_answer("（d）"), "d")  # full-width parens

    def test_identifier_label(self):
        self.assertEqual(extract_answer("Selected Candidate Response Identifier: (c)"), "c")
        self.assertEqual(extract_answer("Selected Candidate Response Identifier: b"), "b")
        self.assertEqual(extract_answer("identifier: (a)"), "a")

    def test_bare_letter_with_cue(self):
        self.assertEqual(extract_answer("The answer is D"), "d")
        self.assertEqual(extract_answer("the selected option is b"), "b")

    def test_bare_letter_fallback(self):
        self.assertEqual(extract_answer("b"), "b")
        self.assertEqual(extract_answer("I would go with c."), "c")

    def test_illegal_options_ignored(self):
        self.assertIsNone(extract_answer("(e)"))

    def test_empty_and_none(self):
        self.assertIsNone(extract_answer(None))
        self.assertIsNone(extract_answer(""))
        self.assertIsNone(extract_answer("no identifier here"))


class TestNormalizeAnswer(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_answer("(c)"), "c")
        self.assertEqual(normalize_answer("'(d)'"), "d")
        self.assertEqual(normalize_answer("b"), "b")
        self.assertEqual(normalize_answer("A"), "a")
        self.assertIsNone(normalize_answer(None))
        self.assertIsNone(normalize_answer(""))


if __name__ == "__main__":
    unittest.main()
