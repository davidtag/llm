"""Unit tests for stdtoken.pyx."""

import unittest

from llm.tokenizers.stdtoken import TokenPair


class TestTokenPair(unittest.TestCase):
    """Unit tests for TokenPair."""

    def test_getters(self) -> None:
        """Test getters for fields."""
        pair = TokenPair(100, 200)
        self.assertEqual(pair.first, 100)
        self.assertEqual(pair.second, 200)

    def test_setters_are_disabled(self) -> None:
        """Test that the datat type is immutable."""
        pair = TokenPair(100, 200)
        with self.assertRaises(AttributeError):
            pair.first = 300
        with self.assertRaises(AttributeError):
            pair.second = 300

    def test_eq(self) -> None:
        """Test the equality operator."""
        pair1 = TokenPair(100, 200)
        pair2 = TokenPair(100, 200)
        pair3 = TokenPair(200, 100)

        # Forward equality
        self.assertEqual(pair1, pair2)
        self.assertNotEqual(pair1, pair3)
        self.assertNotEqual(pair2, pair3)

        # Backward equality
        self.assertEqual(pair2, pair1)
        self.assertNotEqual(pair3, pair1)
        self.assertNotEqual(pair3, pair2)

        # equality is based on values, not id
        self.assertIsNot(pair1, pair2)

    def test_hash(self) -> None:
        """Test the hash implementations."""
        pair1 = TokenPair(100, 200)
        pair2 = TokenPair(100, 200)
        pair3 = TokenPair(200, 100)

        self.assertEqual(hash(pair1), 100 * 1_000_000 + 200)
        self.assertEqual(hash(pair2), 100 * 1_000_000 + 200)
        self.assertEqual(hash(pair3), 200 * 1_000_000 + 100)

    def test_set_hashability(self) -> None:
        """Test that TokenPair can be used in sets."""
        pair1 = TokenPair(100, 200)
        pair2 = TokenPair(100, 200)
        pair3 = TokenPair(200, 100)

        pair_set = {pair1, pair2, pair3}

        self.assertEqual(len(pair_set), 2)
        self.assertSetEqual(pair_set, {TokenPair(100, 200), TokenPair(200, 100)})

    def test_dict_hashability(self) -> None:
        """Test that TokenPair can be used as dict keys."""
        pair1 = TokenPair(100, 200)
        pair2 = TokenPair(100, 200)
        pair3 = TokenPair(200, 100)

        pair_dict = {}
        pair_dict[pair1] = 10
        pair_dict[pair2] = 20  # overrrides the first
        pair_dict[pair3] = 30

        self.assertDictEqual(
            pair_dict,
            {
                TokenPair(first=100, second=200): 20,
                TokenPair(first=200, second=100): 30,
            },
        )

    def test_str(self) -> None:
        """Test string serialization."""
        pair = TokenPair(100, 200)
        self.assertEqual(str(pair), "TokenPair(first=100, second=200)")
