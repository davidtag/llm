"""Unit tests for split_pattern.py."""

import unittest

from llm.tokenizers.split_pattern import SplitPattern


class TestSplitPattern(unittest.TestCase):
    """Unit tests for SplitPattern."""

    def test_all_and_default(self) -> None:
        default_name = SplitPattern.default_pattern_name()
        all_names = SplitPattern.all_pattern_names()

        self.assertGreater(len(all_names), 0)
        self.assertIn(default_name, all_names)

    def test_get_pattern_raises(self) -> None:
        with self.assertRaises(ValueError):
            SplitPattern.get_pattern("==gibberrish==")

    def test_get_pattern(self) -> None:
        SplitPattern.get_pattern(SplitPattern.default_pattern_name())
