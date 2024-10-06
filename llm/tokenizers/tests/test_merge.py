"""Unit tests for merge.pyx."""

from collections import defaultdict
import unittest

import heapq

import numpy as np
from numpy.typing import NDArray

from llm.tokenizers.frequencies import (
    get_pairwise_token_frequencies_sequential_pure_python,
    TokenPairNode,
    get_pairwise_token_frequencies_and_heap_numpy,
)
from llm.tokenizers.merge import (
    merge_inplace,
    merge_inplace_and_update_frequencies,
    merge_inplace_and_update_frequencies_and_heap,
)


TokenDtype = np.uint32
NumpyTokenSequence = NDArray[TokenDtype]


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
        frequencies = {}
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
        frequencies = {}
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
        frequencies[(0, 0)] = 1
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
        frequencies[(3, 3)] = 2
        out_tokens = merge_inplace_and_update_frequencies(
            in_tokens,
            3,
            3,
            9,
            expected_num_merges=1,
            frequencies=frequencies,
        )
        expected = np.array([9, 3], dtype=TokenDtype)
        np.testing.assert_array_equal(out_tokens, expected)
        self.assertIs(out_tokens.base, in_tokens)
        self.assertIs(in_tokens.base, None)
        self.assertDictEqual(frequencies, {(9, 3): 1})

    def test_multiple_merge_targets(self) -> None:
        """Test merging occurs correctly when there are multiple merges."""
        in_tokens = np.array([7, 7, 7, 1, 3, 7, 1, 7, 7], dtype=TokenDtype)
        frequencies = defaultdict(int)
        frequencies[(7, 7)] = 3
        frequencies[(7, 1)] = 2
        frequencies[(1, 3)] = 1
        frequencies[(3, 7)] = 1
        frequencies[(1, 7)] = 1
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
            (9, 7): 1,
            (7, 1): 2,
            (1, 3): 1,
            (3, 7): 1,
            (1, 9): 1,
        }
        self.assertDictEqual(frequencies, expected_frequencies)

    def test_contiguous_merges(self) -> None:
        """Test correct frequency updates when there are successive merges."""
        in_tokens = np.array([3, 7, 7, 7, 7, 4], dtype=TokenDtype)
        frequencies = defaultdict(int)
        frequencies[(3, 7)] = 1
        frequencies[(7, 7)] = 3
        frequencies[(7, 4)] = 1
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
            (3, 9): 1,
            (9, 9): 1,
            (9, 4): 1,
        }
        self.assertDictEqual(frequencies, expected_frequencies)

    def test_contiguous_merges_boundary(self) -> None:
        """Test correct frequency updates when there are successive merges."""
        in_tokens = np.array([7, 7, 7, 7], dtype=TokenDtype)
        frequencies = defaultdict(int)
        frequencies[(7, 7)] = 3
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
            (9, 9): 1,
        }
        self.assertDictEqual(frequencies, expected_frequencies)

    def test_successive_merges(self) -> None:
        """Test corretness after stacked merges."""
        in_tokens = np.array([7, 7, 7, 1, 3, 7, 1, 7, 7], dtype=TokenDtype)
        frequencies = defaultdict(int)
        frequencies[(7, 7)] = 3
        frequencies[(7, 1)] = 2
        frequencies[(1, 3)] = 1
        frequencies[(3, 7)] = 1
        frequencies[(1, 7)] = 1
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
            (9, 10): 1,
            (10, 3): 1,
            (3, 11): 1,
        }
        self.assertDictEqual(frequencies, expected_frequencies)

    def test_random_sequence(self) -> None:
        """Test implementation parity with naive version on a random sequence."""
        tokens = np.random.randint(low=0, high=1000, size=10_000).astype(TokenDtype)
        frequencies = get_pairwise_token_frequencies_sequential_pure_python(tokens)
        out_tokens = merge_inplace_and_update_frequencies(
            tokens,
            7,
            7,
            9,
            expected_num_merges=max(frequencies.values()),
            frequencies=frequencies,
        )
        post_frequencies = get_pairwise_token_frequencies_sequential_pure_python(out_tokens)
        self.assertDictEqual(post_frequencies, frequencies)


class TestMergeInPlaceAndUpdateFrequenciesAndHeap(unittest.TestCase):
    """Unit tests for merge_inplace_and_update_frequencies_and_heap()."""

    def test_empty_input(self) -> None:
        """Test with an empty input."""
        in_tokens = np.array([], dtype=TokenDtype)
        heap = []
        frequencies = {}
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
        heap = []
        frequencies = {}
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
        heap = [TokenPairNode(count=1, token_1=7, token_2=7)]
        frequencies = {
            (7, 7): heap[0],
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

    def test_merges_from_left_to_right(self) -> None:
        """Test merging occurs from left-to-right when multiple merges are possible."""
        in_tokens = np.array([3, 3, 3], dtype=TokenDtype)
        heap = [TokenPairNode(count=2, token_1=3, token_2=3)]
        frequencies = {
            (3, 3): heap[0],
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
        self.assertEqual(heap[0], TokenPairNode(count=1, token_1=9, token_2=3))
        self.assertTrue(heap[1].ignore)
        self.assertDictEqual(frequencies, {(9, 3): heap[0]})

    def test_multiple_merge_targets(self) -> None:
        """Test merging occurs correctly when there are multiple merges."""
        in_tokens = np.array([7, 7, 7, 1, 3, 7, 1, 7, 7], dtype=TokenDtype)
        heap = [
            TokenPairNode(count=3, token_1=7, token_2=7),
            TokenPairNode(count=2, token_1=7, token_2=1),
            TokenPairNode(count=1, token_1=1, token_2=3),
            TokenPairNode(count=1, token_1=3, token_2=7),
            TokenPairNode(count=1, token_1=1, token_2=7),
        ]
        frequencies = {}
        frequencies[(7, 7)] = heap[0]
        frequencies[(7, 1)] = heap[1]
        frequencies[(1, 3)] = heap[2]
        frequencies[(3, 7)] = heap[3]
        frequencies[(1, 7)] = heap[4]
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
        cleaned_heap_nodes = sorted([node for node in heap if not node.ignore])
        self.assertListEqual(
            cleaned_heap_nodes,
            [
                TokenPairNode(count=2, token_1=7, token_2=1),
                TokenPairNode(count=1, token_1=1, token_2=3),
                TokenPairNode(count=1, token_1=1, token_2=9),
                TokenPairNode(count=1, token_1=3, token_2=7),
                TokenPairNode(count=1, token_1=9, token_2=7),
            ],
        )
        expected_frequencies = {
            (7, 1): cleaned_heap_nodes[0],
            (1, 3): cleaned_heap_nodes[1],
            (1, 9): cleaned_heap_nodes[2],
            (3, 7): cleaned_heap_nodes[3],
            (9, 7): cleaned_heap_nodes[4],
        }
        self.assertDictEqual(frequencies, expected_frequencies)

    def test_contiguous_merges(self) -> None:
        """Test correct frequency updates when there are successive merges."""
        in_tokens = np.array([3, 7, 7, 7, 7, 4], dtype=TokenDtype)
        heap = [
            TokenPairNode(count=1, token_1=3, token_2=7),
            TokenPairNode(count=3, token_1=7, token_2=7),
            TokenPairNode(count=1, token_1=7, token_2=4),
        ]
        frequencies = {}
        frequencies[(3, 7)] = heap[0]
        frequencies[(7, 7)] = heap[1]
        frequencies[(7, 4)] = heap[2]
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
        cleaned_heap_nodes = sorted([node for node in heap if not node.ignore])
        self.assertListEqual(
            cleaned_heap_nodes,
            [
                TokenPairNode(count=1, token_1=3, token_2=9),
                TokenPairNode(count=1, token_1=9, token_2=4),
                TokenPairNode(count=1, token_1=9, token_2=9),
            ],
        )
        expected_frequencies = {
            (3, 9): cleaned_heap_nodes[0],
            (9, 9): cleaned_heap_nodes[2],
            (9, 4): cleaned_heap_nodes[1],
        }
        self.assertDictEqual(frequencies, expected_frequencies)

    def test_contiguous_merges_boundary(self) -> None:
        """Test correct frequency updates when there are successive merges."""
        in_tokens = np.array([7, 7, 7, 7], dtype=TokenDtype)
        heap = [
            TokenPairNode(count=3, token_1=7, token_2=7),
        ]
        frequencies = {}
        frequencies[(7, 7)] = heap[0]
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
        cleaned_heap_nodes = sorted([node for node in heap if not node.ignore])
        self.assertListEqual(
            cleaned_heap_nodes,
            [
                TokenPairNode(count=1, token_1=9, token_2=9),
            ],
        )
        expected_frequencies = {
            (9, 9): cleaned_heap_nodes[0],
        }
        self.assertDictEqual(frequencies, expected_frequencies)

    def test_random_sequence(self) -> None:
        """Test implementation parity with re-computing the heap."""
        in_tokens = np.random.randint(low=0, high=1000, size=10_000).astype(TokenDtype)
        frequencies, heap = get_pairwise_token_frequencies_and_heap_numpy(in_tokens)
        max_freq_node = heap[0]
        out_tokens = merge_inplace_and_update_frequencies_and_heap(
            in_tokens,
            max_freq_node.pair[0],
            max_freq_node.pair[1],
            1001,
            expected_num_merges=max_freq_node.count,
            frequencies=frequencies,
            heap=heap,
        )
        _, post_heap = get_pairwise_token_frequencies_and_heap_numpy(out_tokens)

        while len(post_heap) > 0:  # compare the heaps by traversing them
            target_node = heapq.heappop(post_heap)
            out_node = heapq.heappop(heap)
            while out_node.ignore:
                out_node = heapq.heappop(heap)
            self.assertEqual(target_node, out_node)

        self.assertEqual(len(heap), 0)
