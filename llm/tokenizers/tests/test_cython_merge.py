"""Unit tests for merge.pyx."""

from collections import defaultdict
import unittest

import heapq

import numpy as np

from llm.tokenizers.cython.frequencies import (
    get_pairwise_token_frequencies_cython_loop,
    get_pairwise_token_frequencies_and_heap_numpy,
)
from llm.tokenizers.cython.merge import (
    merge,
    merge_inplace,
    merge_inplace_and_update_frequencies,
    merge_inplace_and_update_frequencies_and_heap,
)
from llm.tokenizers.cython.pytoken import TokenDtype, MaskedTokenDtype
from llm.tokenizers.cython.stdtoken import TokenPair, TokenPairNode


class TestMerge(unittest.TestCase):
    """Unit tests for merge()."""

    def test_empty_input(self) -> None:
        """Test with an empty input."""
        in_tokens: list[int] = []
        out_tokens = merge(in_tokens, TokenPair(0, 0), 1)
        expected: list[int] = []
        self.assertListEqual(out_tokens, expected)

    def test_single_token(self) -> None:
        """Test with a single token input."""
        in_tokens = [13]
        out_tokens = merge(in_tokens, TokenPair(0, 0), 1)
        expected = [13]
        self.assertListEqual(out_tokens, expected)

    def test_merges_one_pair(self) -> None:
        """Test merging a single pair."""
        in_tokens = [0, 0]
        out_tokens = merge(in_tokens, TokenPair(0, 0), 1)
        expected = [1]
        self.assertListEqual(out_tokens, expected)

    def test_merges_from_left_to_right(self) -> None:
        """Test merging occurs from left-to-right when multiple merges are possible."""
        in_tokens = [0, 0, 0]
        out_tokens = merge(in_tokens, TokenPair(0, 0), 1)
        expected = [1, 0]
        self.assertListEqual(out_tokens, expected)

    def test_multiple_merge_targets(self) -> None:
        """Test merging occurs correctly when there are multiple merges."""
        in_tokens = [0, 0, 0, 1, 3, 0, 1, 0, 0]
        out_tokens = merge(in_tokens, TokenPair(0, 0), 9)
        expected = [9, 0, 1, 3, 0, 1, 9]
        self.assertListEqual(out_tokens, expected)

    def test_successive_merges(self) -> None:
        """Test correctness and that the same underlying array is used after successive merges."""
        in_tokens = [0, 0, 0, 1, 3, 0, 1, 0, 0]
        o1 = merge(in_tokens, TokenPair(0, 0), 9)  # [9, 0, 1, 3, 0, 1, 9]
        o2 = merge(o1, TokenPair(0, 1), 10)  # [9, 10, 3, 10, 9]
        o3 = merge(o2, TokenPair(10, 9), 11)  # [9, 10, 3, 11]
        out_tokens = o3
        expected = [9, 10, 3, 11]
        self.assertListEqual(out_tokens, expected)


class TestMergeInPlace(unittest.TestCase):
    """Unit tests for merge_inplace()."""

    def test_empty_input(self) -> None:
        """Test with an empty input."""
        in_tokens = np.array([], dtype=TokenDtype)
        out_tokens = merge_inplace(in_tokens, 0, 0, 1)
        expected = np.array([], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)

    def test_single_token(self) -> None:
        """Test with a single token input."""
        in_tokens = np.array([13], dtype=TokenDtype)
        out_tokens = merge_inplace(in_tokens, 0, 0, 1)
        expected = np.array([13], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)

    def test_merges_one_pair(self) -> None:
        """Test merging a single pair."""
        in_tokens = np.array([0, 0], dtype=TokenDtype)
        out_tokens = merge_inplace(in_tokens, 0, 0, 1)
        expected = np.array([1], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)

    def test_merges_from_left_to_right(self) -> None:
        """Test merging occurs from left-to-right when multiple merges are possible."""
        in_tokens = np.array([0, 0, 0], dtype=TokenDtype)
        out_tokens = merge_inplace(in_tokens, 0, 0, 1)
        expected = np.array([1, 0], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)

    def test_multiple_merge_targets(self) -> None:
        """Test merging occurs correctly when there are multiple merges."""
        in_tokens = np.array([0, 0, 0, 1, 3, 0, 1, 0, 0], dtype=TokenDtype)
        out_tokens = merge_inplace(in_tokens, 0, 0, 9)
        expected = np.array([9, 0, 1, 3, 0, 1, 9], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)

    def test_successive_merges(self) -> None:
        """Test correctness and that the same underlying array is used after successive merges."""
        in_tokens = np.array([0, 0, 0, 1, 3, 0, 1, 0, 0], dtype=TokenDtype)
        o1 = merge_inplace(in_tokens, 0, 0, 9)  # [9, 0, 1, 3, 0, 1, 9]
        o2 = merge_inplace(o1, 0, 1, 10)  # [9, 10, 3, 10, 9]
        o3 = merge_inplace(o2, 10, 9, 11)  # [9, 10, 3, 11]
        out_tokens = o3
        expected = np.array([9, 10, 3, 11], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)


class TestMergeInPlaceAndUpdateFrequencies(unittest.TestCase):
    """Unit tests for merge_inplace_and_update_frequencies()."""

    def test_empty_input(self) -> None:
        """Test with an empty input."""
        in_tokens = np.array([], dtype=TokenDtype)
        frequencies = defaultdict[TokenPair, int](int)
        out_tokens = merge_inplace_and_update_frequencies(
            in_tokens,
            0,
            0,
            1,
            expected_num_merges=0,
            frequencies=frequencies,
        )
        expected = np.array([], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        self.assertDictEqual(frequencies, {})

    def test_single_token(self) -> None:
        """Test with a single token input."""
        in_tokens = np.array([13], dtype=TokenDtype)
        frequencies = defaultdict[TokenPair, int](int)
        out_tokens = merge_inplace_and_update_frequencies(
            in_tokens,
            0,
            0,
            1,
            expected_num_merges=0,
            frequencies=frequencies,
        )
        expected = np.array([13], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        self.assertDictEqual(frequencies, {})

    def test_merges_one_pair(self) -> None:
        """Test merging a single pair."""
        in_tokens = np.array([0, 0], dtype=TokenDtype)
        frequencies = defaultdict(int)
        frequencies[TokenPair(0, 0)] = 1
        out_tokens = merge_inplace_and_update_frequencies(
            in_tokens,
            0,
            0,
            9,
            expected_num_merges=1,
            frequencies=frequencies,
        )
        expected = np.array([9], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        self.assertDictEqual(frequencies, {})

    def test_merges_from_left_to_right(self) -> None:
        """Test merging occurs from left-to-right when multiple merges are possible."""
        in_tokens = np.array([3, 3, 3], dtype=TokenDtype)
        frequencies = defaultdict(int)
        frequencies[TokenPair(3, 3)] = 2
        out_tokens = merge_inplace_and_update_frequencies(
            in_tokens,
            3,
            3,
            9,
            expected_num_merges=2,
            frequencies=frequencies,
        )
        expected = np.array([9, 3], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        self.assertDictEqual(frequencies, {TokenPair(9, 3): 1})

    def test_multiple_merge_targets(self) -> None:
        """Test merging occurs correctly when there are multiple merges."""
        in_tokens = np.array([7, 7, 7, 1, 3, 7, 1, 7, 7], dtype=TokenDtype)
        frequencies = defaultdict(int)
        frequencies[TokenPair(7, 7)] = 3
        frequencies[TokenPair(7, 1)] = 2
        frequencies[TokenPair(1, 3)] = 1
        frequencies[TokenPair(3, 7)] = 1
        frequencies[TokenPair(1, 7)] = 1
        out_tokens = merge_inplace_and_update_frequencies(
            in_tokens,
            7,
            7,
            9,
            expected_num_merges=3,  # there will only be 2, but from freq map we would guess 3
            frequencies=frequencies,
        )
        expected = np.array([9, 7, 1, 3, 7, 1, 9], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        expected_frequencies = {
            TokenPair(9, 7): 1,
            TokenPair(7, 1): 2,
            TokenPair(1, 3): 1,
            TokenPair(3, 7): 1,
            TokenPair(1, 9): 1,
        }
        cleaned_frequencies = {key: val for key, val in frequencies.items() if val != 0}
        self.assertDictEqual(cleaned_frequencies, expected_frequencies)

    def test_contiguous_merges(self) -> None:
        """Test correct frequency updates when there are successive merges."""
        in_tokens = np.array([3, 7, 7, 7, 7, 4], dtype=TokenDtype)
        frequencies = defaultdict(int)
        frequencies[TokenPair(3, 7)] = 1
        frequencies[TokenPair(7, 7)] = 3
        frequencies[TokenPair(7, 4)] = 1
        out_tokens = merge_inplace_and_update_frequencies(
            in_tokens,
            7,
            7,
            9,
            expected_num_merges=3,  # there will only be 2, but from freq map we would guess 3
            frequencies=frequencies,
        )
        expected = np.array([3, 9, 9, 4], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        expected_frequencies = {
            TokenPair(3, 9): 1,
            TokenPair(9, 9): 1,
            TokenPair(9, 4): 1,
        }
        cleaned_frequencies = {key: val for key, val in frequencies.items() if val != 0}
        self.assertDictEqual(cleaned_frequencies, expected_frequencies)

    def test_contiguous_merges_boundary(self) -> None:
        """Test correct frequency updates when there are successive merges."""
        in_tokens = np.array([7, 7, 7, 7], dtype=TokenDtype)
        frequencies = defaultdict(int)
        frequencies[TokenPair(7, 7)] = 3
        out_tokens = merge_inplace_and_update_frequencies(
            in_tokens,
            7,
            7,
            9,
            expected_num_merges=3,  # there will only be 2, but from freq map we would guess 3
            frequencies=frequencies,
        )
        expected = np.array([9, 9], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        expected_frequencies = {
            TokenPair(9, 9): 1,
        }
        self.assertDictEqual(frequencies, expected_frequencies)

    def test_successive_merges(self) -> None:
        """Test corretness after stacked merges."""
        in_tokens = np.array([7, 7, 7, 1, 3, 7, 1, 7, 7], dtype=TokenDtype)
        frequencies = defaultdict(int)
        frequencies[TokenPair(7, 7)] = 3
        frequencies[TokenPair(7, 1)] = 2
        frequencies[TokenPair(1, 3)] = 1
        frequencies[TokenPair(3, 7)] = 1
        frequencies[TokenPair(1, 7)] = 1
        o1 = merge_inplace_and_update_frequencies(  # [9, 7, 1, 3, 7, 1, 9]
            in_tokens,
            7,
            7,
            9,
            expected_num_merges=3,
            frequencies=frequencies,
        )
        o2 = merge_inplace_and_update_frequencies(  # [9, 10, 3, 10, 9]
            o1,
            7,
            1,
            10,
            expected_num_merges=2,
            frequencies=frequencies,
        )
        o3 = merge_inplace_and_update_frequencies(  # [9, 10, 3, 11]
            o2,
            10,
            9,
            11,
            expected_num_merges=1,
            frequencies=frequencies,
        )
        out_tokens = o3
        expected = np.array([9, 10, 3, 11], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        expected_frequencies = {
            TokenPair(9, 10): 1,
            TokenPair(10, 3): 1,
            TokenPair(3, 11): 1,
        }
        cleaned_frequencies = {key: val for key, val in frequencies.items() if val != 0}
        self.assertDictEqual(cleaned_frequencies, expected_frequencies)

    def test_random_sequence(self) -> None:
        """Test implementation parity with naive version on a random sequence."""
        tokens = np.random.randint(low=0, high=100, size=10_000).astype(TokenDtype)
        frequencies = get_pairwise_token_frequencies_cython_loop(memoryview(tokens))
        initial_length = len(frequencies)
        max_freq_pair = max(frequencies.keys(), key=lambda pair: frequencies[pair])
        out_tokens = merge_inplace_and_update_frequencies(
            tokens,
            max_freq_pair.first,
            max_freq_pair.second,
            101,
            expected_num_merges=frequencies[max_freq_pair],
            frequencies=frequencies,
        )
        final_length = len(frequencies)
        self.assertNotEqual(initial_length, final_length)
        cleaned_frequencies = {key: val for key, val in frequencies.items() if val != 0}

        post_frequencies = get_pairwise_token_frequencies_cython_loop(memoryview(out_tokens))
        self.assertDictEqual(cleaned_frequencies, post_frequencies)


class TestMergeInPlaceAndUpdateFrequenciesAndHeap(unittest.TestCase):
    """Unit tests for merge_inplace_and_update_frequencies_and_heap()."""

    def test_empty_input(self) -> None:
        """Test with an empty input."""
        in_tokens = np.array([], dtype=TokenDtype)
        heap: list[TokenPairNode] = []
        frequencies: dict[TokenPair, TokenPairNode] = {}
        out_tokens = merge_inplace_and_update_frequencies_and_heap(
            in_tokens,
            0,
            0,
            1,
            expected_num_merges=0,
            frequencies=frequencies,
            heap=heap,
        )
        expected = np.array([], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        self.assertDictEqual(frequencies, {})
        self.assertListEqual(heap, [])

    def test_single_token(self) -> None:
        """Test with a single token input."""
        in_tokens = np.array([13], dtype=TokenDtype)
        heap: list[TokenPairNode] = []
        frequencies: dict[TokenPair, TokenPairNode] = {}
        out_tokens = merge_inplace_and_update_frequencies_and_heap(
            in_tokens,
            0,
            0,
            1,
            expected_num_merges=0,
            frequencies=frequencies,
            heap=[],
        )
        expected = np.array([13], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        self.assertDictEqual(frequencies, {})
        self.assertListEqual(heap, [])

    def test_merges_one_pair(self) -> None:
        """Test merging a single pair."""
        in_tokens = np.array([7, 7], dtype=TokenDtype)
        heap = [
            TokenPairNode(7, 7, count=1),
        ]
        frequencies = {
            TokenPair(7, 7): heap[0],
        }
        out_tokens = merge_inplace_and_update_frequencies_and_heap(
            in_tokens,
            7,
            7,
            9,
            expected_num_merges=1,
            frequencies=frequencies,
            heap=heap,
        )
        expected = np.array([9], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        self.assertDictEqual(frequencies, {})
        self.assertListEqual(heap, [])

    def test_merges_with_mask(self) -> None:
        """Test merging in the presence of a mask."""
        in_tokens = np.array([7, 7, -1, 7, 7], dtype=MaskedTokenDtype).astype(TokenDtype)
        heap = [
            TokenPairNode(7, 7, count=2),
        ]
        frequencies = {
            TokenPair(7, 7): heap[0],
        }
        out_tokens = merge_inplace_and_update_frequencies_and_heap(
            in_tokens,
            7,
            7,
            9,
            expected_num_merges=2,
            frequencies=frequencies,
            heap=heap,
        )
        expected = np.array([9, -1, 9], dtype=MaskedTokenDtype).astype(TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        self.assertDictEqual(frequencies, {})
        self.assertListEqual(heap, [])

    def test_merges_from_left_to_right(self) -> None:
        """Test merging occurs from left-to-right when multiple merges are possible."""
        in_tokens = np.array([3, 3, 3], dtype=TokenDtype)
        heap = [
            TokenPairNode(3, 3, count=2),
        ]
        frequencies = {
            TokenPair(3, 3): heap[0],
        }
        out_tokens = merge_inplace_and_update_frequencies_and_heap(
            in_tokens,
            3,
            3,
            9,
            expected_num_merges=2,
            frequencies=frequencies,
            heap=heap,
        )
        expected = np.array([9, 3], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        self.assertEqual(len(heap), 2)
        self.assertEqual(heap[0], TokenPairNode(9, 3, count=1))
        self.assertTrue(heap[1].deleted)
        self.assertDictEqual(frequencies, {TokenPair(9, 3): heap[0]})

    def test_multiple_merge_targets(self) -> None:
        """Test merging occurs correctly when there are multiple merges."""
        in_tokens = np.array([7, 7, 7, 1, 3, 7, 1, 7, 7], dtype=TokenDtype)
        heap = [
            TokenPairNode(7, 7, count=3),
            TokenPairNode(7, 1, count=2),
            TokenPairNode(1, 3, count=1),
            TokenPairNode(3, 7, count=1),
            TokenPairNode(1, 7, count=1),
        ]
        frequencies = {}
        frequencies[TokenPair(7, 7)] = heap[0]
        frequencies[TokenPair(7, 1)] = heap[1]
        frequencies[TokenPair(1, 3)] = heap[2]
        frequencies[TokenPair(3, 7)] = heap[3]
        frequencies[TokenPair(1, 7)] = heap[4]
        heapq.heapify(heap)
        out_tokens = merge_inplace_and_update_frequencies_and_heap(
            in_tokens,
            7,
            7,
            9,
            expected_num_merges=3,
            frequencies=frequencies,
            heap=heap,
        )
        expected = np.array([9, 7, 1, 3, 7, 1, 9], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        cleaned_heap_nodes = sorted([node for node in heap if not node.deleted])
        self.assertListEqual(
            cleaned_heap_nodes,
            [
                TokenPairNode(7, 1, count=2),
                TokenPairNode(1, 3, count=1),
                TokenPairNode(1, 9, count=1),
                TokenPairNode(3, 7, count=1),
                TokenPairNode(9, 7, count=1),
            ],
        )
        expected_frequencies = {
            TokenPair(7, 1): cleaned_heap_nodes[0],
            TokenPair(1, 3): cleaned_heap_nodes[1],
            TokenPair(1, 9): cleaned_heap_nodes[2],
            TokenPair(3, 7): cleaned_heap_nodes[3],
            TokenPair(9, 7): cleaned_heap_nodes[4],
        }
        self.assertDictEqual(frequencies, expected_frequencies)

    def test_contiguous_merges(self) -> None:
        """Test correct frequency updates when there are successive merges."""
        in_tokens = np.array([3, 7, 7, 7, 7, 4], dtype=TokenDtype)
        heap = [
            TokenPairNode(3, 7, count=1),
            TokenPairNode(7, 7, count=3),
            TokenPairNode(7, 4, count=1),
        ]
        frequencies = {}
        frequencies[TokenPair(3, 7)] = heap[0]
        frequencies[TokenPair(7, 7)] = heap[1]
        frequencies[TokenPair(7, 4)] = heap[2]
        heapq.heapify(heap)
        out_tokens = merge_inplace_and_update_frequencies_and_heap(
            in_tokens,
            7,
            7,
            9,
            expected_num_merges=3,
            frequencies=frequencies,
            heap=heap,
        )
        expected = np.array([3, 9, 9, 4], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        cleaned_heap_nodes = sorted([node for node in heap if not node.deleted])
        self.assertListEqual(
            cleaned_heap_nodes,
            [
                TokenPairNode(3, 9, count=1),
                TokenPairNode(9, 4, count=1),
                TokenPairNode(9, 9, count=1),
            ],
        )
        expected_frequencies = {
            TokenPair(3, 9): cleaned_heap_nodes[0],
            TokenPair(9, 9): cleaned_heap_nodes[2],
            TokenPair(9, 4): cleaned_heap_nodes[1],
        }
        self.assertDictEqual(frequencies, expected_frequencies)

    def test_contiguous_merges_boundary(self) -> None:
        """Test correct frequency updates when there are successive merges."""
        in_tokens = np.array([7, 7, 7, 7], dtype=TokenDtype)
        heap = [
            TokenPairNode(7, 7, count=3),
        ]
        frequencies = {}
        frequencies[TokenPair(7, 7)] = heap[0]
        out_tokens = merge_inplace_and_update_frequencies_and_heap(
            in_tokens,
            7,
            7,
            9,
            expected_num_merges=3,
            frequencies=frequencies,
            heap=heap,
        )
        expected = np.array([9, 9], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        cleaned_heap_nodes = sorted([node for node in heap if not node.deleted])
        self.assertListEqual(
            cleaned_heap_nodes,
            [
                TokenPairNode(9, 9, count=1),
            ],
        )
        expected_frequencies = {
            TokenPair(9, 9): cleaned_heap_nodes[0],
        }
        self.assertDictEqual(frequencies, expected_frequencies)

    def test_random_sequence(self) -> None:
        """Test implementation parity with re-computing the heap."""
        in_tokens = np.random.randint(low=0, high=100, size=10_000).astype(TokenDtype)
        frequencies, heap = get_pairwise_token_frequencies_and_heap_numpy(in_tokens)
        max_freq_node = heap[0]
        out_tokens = merge_inplace_and_update_frequencies_and_heap(
            in_tokens,
            max_freq_node.first,
            max_freq_node.second,
            101,
            expected_num_merges=max_freq_node.count,
            frequencies=frequencies,
            heap=heap,
        )
        _, post_heap = get_pairwise_token_frequencies_and_heap_numpy(out_tokens)

        while len(post_heap) > 0:  # compare the heaps by traversing them
            target_node = heapq.heappop(post_heap)
            out_node = heapq.heappop(heap)
            while out_node.deleted:
                out_node = heapq.heappop(heap)
            self.assertEqual(target_node, out_node)

        self.assertEqual(len(heap), 0)
