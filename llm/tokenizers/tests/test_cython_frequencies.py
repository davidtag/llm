"""Unit tests for frequencies.pyx."""

import abc
import heapq
from typing import Callable, Mapping, Tuple
import unittest

import numpy as np

from llm.tokenizers.cython.frequencies import (
    get_pairwise_tokens,
    get_pairwise_token_frequencies_sequential_pure_python,
    get_pairwise_token_frequencies_from_list,
    get_pairwise_token_frequencies_cython_loop,
    get_pairwise_token_frequencies_numpy,
    get_pairwise_token_frequencies_numpy_maxonly,
    get_pairwise_token_frequencies_numpy_bitshift,
    get_pairwise_token_frequencies_numpy_bitshift_maxonly,
    get_pairwise_token_frequencies_and_heap_numpy,
    get_masked_pairwise_token_frequencies_and_heap_numpy,
)
from llm.tokenizers.cython.pytoken import TokenDtype, NumpyTokenSequence, MaskedTokenDtype
from llm.tokenizers.cython.stdtoken import TokenPair


class TestGetPairwiseTokens(unittest.TestCase):
    """Unit tests for get_pairwise_tokens()."""

    def test_empty_input(self) -> None:
        pairs = get_pairwise_tokens([])
        self.assertSetEqual(pairs, set())

    def test_single_token(self) -> None:
        pairs = get_pairwise_tokens([13])
        self.assertSetEqual(pairs, set())

    def test_two_tokens(self) -> None:
        pairs = get_pairwise_tokens([13, 14])
        self.assertSetEqual(pairs, {TokenPair(13, 14)})

    def test_three_tokens(self) -> None:
        pairs = get_pairwise_tokens([13, 14, 15])
        self.assertSetEqual(pairs, {TokenPair(13, 14), TokenPair(14, 15)})

    def test_repeated_tokens(self) -> None:
        pairs = get_pairwise_tokens([13, 13, 13, 13, 13, 13, 13, 13, 13, 13])
        self.assertSetEqual(pairs, {TokenPair(13, 13)})

    def test_large_token_values(self) -> None:
        pairs = get_pairwise_tokens([9_000, 5_000, 40_000, 62_000])
        self.assertSetEqual(
            pairs,
            {
                TokenPair(9_000, 5_000),
                TokenPair(5_000, 40_000),
                TokenPair(40_000, 62_000),
            },
        )


#########################################################################################
# Common                                                                                #
#########################################################################################


class FrequencyCommonBaseTest(abc.ABC):
    """Common tests for all implementations."""

    assertDictEqual: Callable

    @abc.abstractmethod
    def _call(self, tokens: NumpyTokenSequence) -> Mapping[Tuple[int, int], int]:
        raise NotImplementedError

    def _convert_token_pair_to_tuple(self, pair: TokenPair) -> Tuple[int, int]:
        return (pair.first, pair.second)

    def _convert_token_pair_dict_to_tuple_dict(
        self, freq: Mapping[TokenPair, int]
    ) -> Mapping[Tuple[int, int], int]:
        out = {self._convert_token_pair_to_tuple(key): val for key, val in freq.items()}
        return out

    def test_empty_input(self) -> None:
        """Test frequency on an empty input."""
        tokens = np.array([], dtype=TokenDtype)
        freq = self._call(tokens)
        self.assertDictEqual(freq, {})

    def test_single_token(self) -> None:
        """Test frequency on a single-token input."""
        tokens = np.array([0], dtype=TokenDtype)
        freq = self._call(tokens)
        self.assertDictEqual(freq, {})

    def test_two_tokens(self) -> None:
        """Test frequency on a 2-token input."""
        tokens = np.array([0, 0], dtype=TokenDtype)
        freq = self._call(tokens)
        self.assertDictEqual(freq, {(0, 0): 1})

    def test_repeated_tokens(self) -> None:
        """Test frequency on a 2-token input."""
        tokens = np.array([9, 9, 9, 9, 9, 9, 9, 9], dtype=TokenDtype)
        freq = self._call(tokens)
        self.assertDictEqual(freq, {(9, 9): 7})

    def test_large_token_values(self) -> None:
        """Test that large token values are handled correctly."""
        tokens = np.array([9_000, 5_000, 40_000, 62_000], dtype=TokenDtype)
        freq = self._call(tokens)
        self.assertDictEqual(
            freq,
            {
                (9_000, 5_000): 1,
                (5_000, 40_000): 1,
                (40_000, 62_000): 1,
            },
        )


#########################################################################################
# Helpers returning full frequencies                                                    #
#########################################################################################


class AllFrequencyBaseTest(FrequencyCommonBaseTest):
    """Common tests for all implementations returning full-frequency maps."""

    def test_small_sequence(self) -> None:
        """Test frequency on a small input sequence."""
        tokens = np.array([0, 0, 1, 0, 2, 1, 0], dtype=TokenDtype)
        freq = self._call(tokens)
        self.assertDictEqual(
            freq,
            {
                (0, 0): 1,
                (0, 1): 1,
                (1, 0): 2,
                (0, 2): 1,
                (2, 1): 1,
            },
        )

    def test_random_sequence(self) -> None:
        """Test implementation parity with naive version on a random sequence."""
        tokens = np.random.randint(low=0, high=1000, size=10_000).astype(TokenDtype)
        ground_truth = get_pairwise_token_frequencies_sequential_pure_python(tokens.tolist())
        freq = self._call(tokens)
        self.assertDictEqual(freq, ground_truth)


class TestSequentialPurePython(AllFrequencyBaseTest, unittest.TestCase):
    """Unit tests for get_pairwise_token_frequencies_sequential_pure_python()."""

    def _call(self, tokens: NumpyTokenSequence) -> Mapping[Tuple[int, int], int]:
        return get_pairwise_token_frequencies_sequential_pure_python(tokens.tolist())


class TestFromList(AllFrequencyBaseTest, unittest.TestCase):
    """Unit tests for get_pairwise_token_frequencies_from_list()."""

    def _call(self, tokens: NumpyTokenSequence) -> Mapping[Tuple[int, int], int]:
        freq = get_pairwise_token_frequencies_from_list(tokens.tolist())
        return self._convert_token_pair_dict_to_tuple_dict(freq)


class TestSequentialCython(AllFrequencyBaseTest, unittest.TestCase):
    """Unit tests for get_pairwise_token_frequencies_cython_loop()."""

    def _call(self, tokens: NumpyTokenSequence) -> Mapping[Tuple[int, int], int]:
        freq = get_pairwise_token_frequencies_cython_loop(memoryview(tokens))
        return self._convert_token_pair_dict_to_tuple_dict(freq)


class TestNumpy(AllFrequencyBaseTest, unittest.TestCase):
    """Unit tests for get_pairwise_token_frequencies_numpy()."""

    def _call(self, tokens: NumpyTokenSequence) -> Mapping[Tuple[int, int], int]:
        freq = get_pairwise_token_frequencies_numpy(tokens)
        return self._convert_token_pair_dict_to_tuple_dict(freq)


class TestNumpyBitShift(AllFrequencyBaseTest, unittest.TestCase):
    """Unit tests for get_pairwise_token_frequencies_numpy_bitshift()."""

    def _call(self, tokens: NumpyTokenSequence) -> Mapping[Tuple[int, int], int]:
        freq = get_pairwise_token_frequencies_numpy_bitshift(tokens)
        return self._convert_token_pair_dict_to_tuple_dict(freq)


class TestNumpyWithHeap(AllFrequencyBaseTest, unittest.TestCase):
    """Unit tests for get_pairwise_token_frequencies_and_heap_numpy()."""

    def _call(self, tokens: NumpyTokenSequence) -> Mapping[Tuple[int, int], int]:
        pair_to_node, _ = get_pairwise_token_frequencies_and_heap_numpy(tokens)
        freq = {}
        for pair, node in pair_to_node.items():
            assert node.pair == pair
            assert node.deleted is False
            freq[self._convert_token_pair_to_tuple(pair)] = node.count
        return freq

    def test_small_sequence(self) -> None:
        """Test frequency on a small input sequence."""
        tokens = np.array([0, 0, 1, 0, 2, 1, 0], dtype=TokenDtype)
        # Raw frequencies:
        # {
        #     (0, 0): 1,
        #     (0, 1): 1,
        #     (1, 0): 2,
        #     (0, 2): 1,
        #     (2, 1): 1,
        # }
        pair_to_node, heap = get_pairwise_token_frequencies_and_heap_numpy(tokens)
        self.assertEqual(len(pair_to_node), len(heap))

        # Test the min_node
        min_node = heapq.heappop(heap)
        self.assertEqual(min_node.count, 2)
        self.assertEqual(min_node.pair, TokenPair(1, 0))
        self.assertIs(pair_to_node[TokenPair(1, 0)], min_node)

        # Check for correct tie-break ordering among remaining nodes
        node_1 = heapq.heappop(heap)
        node_2 = heapq.heappop(heap)
        node_3 = heapq.heappop(heap)
        node_4 = heapq.heappop(heap)
        self.assertEqual(len(heap), 0)

        self.assertEqual(node_1.count, 1)
        self.assertEqual(node_2.count, 1)
        self.assertEqual(node_3.count, 1)
        self.assertEqual(node_4.count, 1)

        self.assertEqual(node_1.pair, TokenPair(0, 0))
        self.assertEqual(node_2.pair, TokenPair(0, 1))
        self.assertEqual(node_3.pair, TokenPair(0, 2))
        self.assertEqual(node_4.pair, TokenPair(2, 1))

        self.assertIs(pair_to_node[TokenPair(0, 0)], node_1)
        self.assertIs(pair_to_node[TokenPair(0, 1)], node_2)
        self.assertIs(pair_to_node[TokenPair(0, 2)], node_3)
        self.assertIs(pair_to_node[TokenPair(2, 1)], node_4)


class TestMaskedNumpyWithHeap(AllFrequencyBaseTest, unittest.TestCase):
    """Unit tests for get_masked_pairwise_token_frequencies_and_heap_numpy()."""

    def _call(self, tokens: NumpyTokenSequence) -> Mapping[Tuple[int, int], int]:
        pair_to_node, _ = get_masked_pairwise_token_frequencies_and_heap_numpy(
            tokens.astype(MaskedTokenDtype),
            np.array([], dtype=MaskedTokenDtype),
        )
        freq = {}
        for pair, node in pair_to_node.items():
            assert node.pair == pair
            assert node.deleted is False
            freq[self._convert_token_pair_to_tuple(pair)] = node.count
        return freq

    def test_small_sequence(self) -> None:
        """Test frequency on a small input sequence."""
        tokens_masked = np.array([0, 0, -1, 1, 0, -1, 2, 1, 0], dtype=MaskedTokenDtype)
        masked_positions = np.array([2, 5], dtype=MaskedTokenDtype)
        # Raw frequencies:
        # {
        #     (0, 0): 1,
        #     (1, 0): 2,
        #     (2, 1): 1,
        # }
        pair_to_node, heap = get_masked_pairwise_token_frequencies_and_heap_numpy(
            tokens_masked,
            masked_positions,
        )
        self.assertEqual(len(pair_to_node), len(heap))

        # Test the min_node
        min_node = heapq.heappop(heap)
        self.assertEqual(min_node.count, 2)
        self.assertEqual(min_node.pair, TokenPair(1, 0))
        self.assertIs(pair_to_node[TokenPair(1, 0)], min_node)

        # Check for correct tie-break ordering among remaining nodes
        node_1 = heapq.heappop(heap)
        node_2 = heapq.heappop(heap)
        self.assertEqual(len(heap), 0)

        self.assertEqual(node_1.count, 1)
        self.assertEqual(node_2.count, 1)

        self.assertEqual(node_1.pair, TokenPair(0, 0))
        self.assertEqual(node_2.pair, TokenPair(2, 1))

        self.assertIs(pair_to_node[TokenPair(0, 0)], node_1)
        self.assertIs(pair_to_node[TokenPair(2, 1)], node_2)


#########################################################################################
# Helpers returning max-only frequencies                                                #
#########################################################################################


class MaxOnlyFrequencyBaseTest(FrequencyCommonBaseTest):
    """Common tests for all implementations returning max-only frequency maps."""

    def test_small_sequence(self) -> None:
        """Test frequency on a small input sequence."""
        tokens = np.array([0, 0, 1, 0, 2, 1, 0], dtype=TokenDtype)
        freq = self._call(tokens)
        self.assertDictEqual(
            freq,
            {
                (1, 0): 2,
            },
        )

    def test_small_sequence_with_ties(self) -> None:
        """Test frequency on a small input sequence."""
        tokens = np.array([0, 0, 1, 0, 2, 1], dtype=TokenDtype)
        freq = self._call(tokens)
        self.assertDictEqual(
            freq,
            {
                (0, 0): 1,
                (0, 1): 1,
                (1, 0): 1,
                (0, 2): 1,
                (2, 1): 1,
            },
        )

    def test_random_sequence(self) -> None:
        """Test implementation parity with naive version on a random sequence."""
        tokens = np.random.randint(low=0, high=1000, size=10_000).astype(TokenDtype)
        all_pairs = get_pairwise_token_frequencies_sequential_pure_python(tokens.tolist())
        max_val = max(all_pairs.values())
        ground_truth = {key: val for key, val in all_pairs.items() if val == max_val}
        freq = self._call(tokens)
        self.assertDictEqual(freq, ground_truth)


class TestNumpyMaxOnly(MaxOnlyFrequencyBaseTest, unittest.TestCase):
    """Unit tests for get_pairwise_token_frequencies_numpy_maxonly()."""

    def _call(self, tokens: NumpyTokenSequence) -> Mapping[Tuple[int, int], int]:
        freq = get_pairwise_token_frequencies_numpy_maxonly(tokens)
        return self._convert_token_pair_dict_to_tuple_dict(freq)


class TestNumpyBitShiftMaxOnly(MaxOnlyFrequencyBaseTest, unittest.TestCase):
    """Unit tests for get_pairwise_token_frequencies_numpy_bitshift_maxonly()."""

    def _call(self, tokens: NumpyTokenSequence) -> Mapping[Tuple[int, int], int]:
        freq = get_pairwise_token_frequencies_numpy_bitshift_maxonly(tokens)
        return self._convert_token_pair_dict_to_tuple_dict(freq)
