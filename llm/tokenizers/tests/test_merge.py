"""Unit tests for merge.pyx."""

from collections import defaultdict
import unittest

import numpy as np
from numpy.typing import NDArray

from llm.tokenizers.merge import (
    merge_inplace,
    merge_inplace_and_update_frequencies,
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
