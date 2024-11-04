"""Unit tests for stdtoken.pyx."""

import heapq
import unittest

from llm.tokenizers.cython.stdtoken import TokenPair, TokenPairNode


class TestTokenPair(unittest.TestCase):
    """Unit tests for TokenPair."""

    def test_getters(self) -> None:
        """Test getters for fields."""
        pair = TokenPair(100, 200)
        self.assertEqual(pair.first, 100)
        self.assertEqual(pair.second, 200)

    def test_setters_are_disabled(self) -> None:
        """Test that the data type is immutable."""
        pair = TokenPair(100, 200)
        with self.assertRaises(AttributeError):
            pair.first = 300  # type: ignore
        with self.assertRaises(AttributeError):
            pair.second = 300  # type: ignore

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

    def test_lt(self) -> None:
        """Test the lt operator."""
        pair1 = TokenPair(100, 200)
        pair2 = TokenPair(100, 200)
        pair3 = TokenPair(200, 100)

        self.assertFalse(pair1 < pair2)
        self.assertFalse(pair2 < pair1)

        self.assertTrue(pair1 < pair3)
        self.assertFalse(pair3 < pair1)

    def test_hash(self) -> None:
        """Test the hash implementations."""
        pair1 = TokenPair(100, 200)
        pair2 = TokenPair(100, 200)
        pair3 = TokenPair(200, 100)

        self.assertEqual(hash(pair1), 100 * 1_000_000 + 200)
        self.assertEqual(hash(pair2), 100 * 1_000_000 + 200)
        self.assertEqual(hash(pair3), 200 * 1_000_000 + 100)

    def test_hash_large_values(self) -> None:
        """Test the hash implementation for large token values."""
        pair = TokenPair(999_999, 999_999)
        self.assertEqual(hash(pair) + 1, 1_000_000**2)

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


class TestTokenPairNode(unittest.TestCase):
    """Unit tests for TokenPairNode."""

    def test_getters(self) -> None:
        """Test getters for fields."""
        node = TokenPairNode(100, 200, count=13)
        self.assertEqual(node.first, 100)
        self.assertEqual(node.second, 200)
        self.assertEqual(node.pair, TokenPair(100, 200))
        self.assertEqual(node.count, 13)
        self.assertEqual(node.deleted, False)

    def test_setters(self) -> None:
        """Test that certain fields are immutable while others aren't."""
        node = TokenPairNode(100, 200, count=13)
        with self.assertRaises(AttributeError):
            node.first = 300  # type: ignore
        with self.assertRaises(AttributeError):
            node.second = 300  # type: ignore

        self.assertEqual(node.count, 13)
        node.count = 40
        self.assertEqual(node.count, 40)

        self.assertEqual(node.deleted, False)
        node.deleted = True
        self.assertEqual(node.deleted, True)

    def test_eq(self) -> None:
        """Test the equality operator."""
        node1 = TokenPairNode(100, 200, count=13)
        node2 = TokenPairNode(100, 200, count=13)
        node3 = TokenPairNode(200, 100, count=13)
        node4 = TokenPairNode(200, 100, count=13, deleted=True)

        # Forward equality
        self.assertEqual(node1, node2)
        self.assertNotEqual(node1, node3)
        self.assertNotEqual(node1, node4)
        self.assertNotEqual(node2, node3)
        self.assertNotEqual(node2, node4)
        self.assertNotEqual(node3, node4)

        # Backward equality
        self.assertEqual(node2, node1)
        self.assertNotEqual(node3, node1)
        self.assertNotEqual(node4, node1)
        self.assertNotEqual(node3, node2)
        self.assertNotEqual(node4, node2)
        self.assertNotEqual(node4, node3)

        # equality is based on values, not id
        self.assertIsNot(node1, node2)

    def test_not_hashable(self) -> None:
        """Test that TokenPairNode is not hashable."""
        node = TokenPairNode(100, 200, count=13)
        with self.assertRaises(TypeError):
            _ = set([node])
        with self.assertRaises(TypeError):
            _ = {node: 1}  # type: ignore

    def test_lt_by_count(self) -> None:
        """Test that nodes with higher count are ordered first."""
        node1 = TokenPairNode(100, 200, count=13)
        node2 = TokenPairNode(400, 500, count=53)
        self.assertLess(node2, node1)  # higher count orders

    def test_lt_tiebreak_with_pair(self) -> None:
        """Test that nodes with equal count tie-break by pair values."""
        node1 = TokenPairNode(100, 200, count=13)
        node2 = TokenPairNode(400, 500, count=13)
        self.assertLess(node1, node2)  # higher count orders

    def test_sort(self) -> None:
        """Test ability to sort nodes."""
        node1 = TokenPairNode(100, 200, count=13)
        node2 = TokenPairNode(400, 500, count=13)
        node3 = TokenPairNode(400, 500, count=53)

        node_list = [node1, node2, node3]
        sorted_nodes = sorted(node_list)
        self.assertListEqual(sorted_nodes, [node3, node1, node2])

    def test_heapify(self) -> None:
        """Test the ability to store nodes in a heap."""
        node1 = TokenPairNode(100, 200, count=13)
        node2 = TokenPairNode(400, 500, count=13)
        node3 = TokenPairNode(400, 500, count=53)

        heap = [node1, node2, node3]
        heapq.heapify(heap)

        min_node_1 = heapq.heappop(heap)
        min_node_2 = heapq.heappop(heap)
        min_node_3 = heapq.heappop(heap)

        self.assertIs(min_node_1, node3)
        self.assertIs(min_node_2, node1)
        self.assertIs(min_node_3, node2)

    def test_str(self) -> None:
        """Test string serialization."""
        pair = TokenPairNode(100, 200, count=13, deleted=True)
        self.assertEqual(str(pair), "TokenPairNode(first=100, second=200, count=13, deleted=True)")
