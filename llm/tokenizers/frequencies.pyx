"""Various library methods for computing pairwise statistics over token sequences."""

from llm.tokenizers.stdtoken cimport token_t, token_sequence_t, TokenPair, TokenPairNode, TOKEN_VALUE_UBOUND

from collections import defaultdict
import heapq
from typing import Tuple

import cython
import numpy as np

from llm.tokenizers.pytoken import NumpyTokenSequence


def get_pairwise_token_frequencies_sequential_pure_python(
    tokens: Sequence[int],
) -> defaultdict[Tuple[int, int], int]:
    """Compute the token frequencies using a sequential scan in pure-Python."""
    freq: defaultdict[Tuple[int, int], int] = defaultdict(int)

    for token_1, token_2 in zip(tokens[:-1], tokens[1:], strict=True):
        pair = (token_1, token_2)
        freq[pair] += 1

    return freq


@cython.boundscheck(False)
@cython.wraparound(False)
def get_pairwise_token_frequencies_sequential_cython(
    token_sequence_t tokens,
) -> defaultdict[TokenPair, int]:
    """Compute the token frequencies using a sequential scan in Cython."""
    freq: defaultdict[TokenPair, int] = defaultdict(int)

    cdef Py_ssize_t n = tokens.shape[0]
    cdef Py_ssize_t i
    cdef token_t token_1
    cdef token_t token_2

    for i in range(n - 1):
        token_1 = tokens[i]
        token_2 = tokens[i + 1]
        pair = TokenPair(token_1, token_2)
        freq[pair] += 1

    return freq



def get_pairwise_token_frequencies_numpy(
    tokens: NumpyTokenSequence,
) -> defaultdict[TokenPair, int]:
    """Compute the token frequencies using numpy broadcast."""
    freq: defaultdict[TokenPair, int] = defaultdict(int)

    # Determine all unique pairs using bit packing
    y = tokens[:-1] * TOKEN_VALUE_UBOUND + tokens[1:]  # TODO(dtag): Tokens over ~2k will overflow int32
    unique_values, counts = np.unique(y, return_counts=True)

    # Efficiently unpack them
    first_tokens = np.floor_divide(unique_values, TOKEN_VALUE_UBOUND)
    second_tokens = np.mod(unique_values, TOKEN_VALUE_UBOUND)

    # Package the frequencies
    for token_1, token_2, count in zip(first_tokens, second_tokens, counts, strict=True):
        pair = TokenPair(token_1, token_2)
        freq[pair] = count

    return freq


def get_pairwise_token_frequencies_numpy_maxonly(
    tokens: NumpyTokenSequence,
) -> defaultdict[TokenPair, int]:
    """Compute the token frequencies using numpy broadcast, filtered for max-freq only."""
    freq: defaultdict[TokenPair, int] = defaultdict(int)

    if tokens.shape[0] <= 1:
        return freq

    # Determine all unique pairs using bit packing
    y = tokens[:-1] * TOKEN_VALUE_UBOUND + tokens[1:]  # TODO(dtag): Tokens over ~2k will overflow int32
    unique_values, counts = np.unique(y, return_counts=True)

    # Determine the max count and all it's occurences
    max_count = np.max(counts)
    max_count_idxs = np.argwhere(counts == max_count).reshape(-1, copy=False)

    # Efficiently unpack them
    first_tokens = np.floor_divide(unique_values[max_count_idxs], TOKEN_VALUE_UBOUND)
    second_tokens = np.mod(unique_values[max_count_idxs], TOKEN_VALUE_UBOUND)

    # Package the frequencies
    for token_1, token_2 in zip(first_tokens, second_tokens, strict=True):
        pair = TokenPair(token_1, token_2)
        freq[pair] = max_count

    return freq


cdef int _BIT_SHIFT = 16
cdef token_t MASK_UPPER_BITS = (1 << _BIT_SHIFT) - 1


def get_pairwise_token_frequencies_numpy_bitshift(
    tokens: NumpyTokenSequence,
) -> defaultdict[TokenPair, int]:
    """Compute the token frequencies using numpy broadcast with bitshift.

    NOTE: this can be slighlty faster than the base numpy version but
    only allows for tokens up to 2^16 = 65536.
    """
    freq: defaultdict[TokenPair, int] = defaultdict(int)

    # Determine all unique pairs using bit packing
    # Upper half bits are the first token, with lower half bits the second token
    y = np.bitwise_or(np.left_shift(tokens[:-1], _BIT_SHIFT), tokens[1:])
    unique_values, counts = np.unique(y, return_counts=True)

    # Efficiently unpack them
    first_tokens = np.right_shift(unique_values, _BIT_SHIFT)
    second_tokens = np.bitwise_and(unique_values, MASK_UPPER_BITS)

    # Package the frequencies
    for token_1, token_2, count in zip(first_tokens, second_tokens, counts, strict=True):
        pair = TokenPair(token_1, token_2)
        freq[pair] = count

    return freq


def get_pairwise_token_frequencies_numpy_bitshift_maxonly(
    tokens: NumpyTokenSequence,
) -> defaultdict[TokenPair, int]:
    """Compute the token frequencies using numpy broadcast with bitshift, filtered for max-freq only.

    NOTE: this can be slighlty faster than the base numpy version but
    only allows for tokens up to 2^16 = 65536.
    """
    freq: defaultdict[TokenPair, int] = defaultdict(int)

    if tokens.shape[0] <= 1:
        return freq

    # Determine all unique pairs using bit packing
    # Upper half bits are the first token, with lower half bits the second token
    y = np.bitwise_or(np.left_shift(tokens[:-1], _BIT_SHIFT), tokens[1:])
    unique_values, counts = np.unique(y, return_counts=True)

    # Determine the max count and all it's occurences
    max_count = np.max(counts)
    max_count_idxs = np.argwhere(counts == max_count).reshape(-1, copy=False)

    # Efficiently unpack them
    first_tokens = np.right_shift(unique_values[max_count_idxs], _BIT_SHIFT)
    second_tokens = np.bitwise_and(unique_values[max_count_idxs], MASK_UPPER_BITS)

    # Package the frequencies
    for token_1, token_2 in zip(first_tokens, second_tokens, strict=True):
        pair = TokenPair(token_1, token_2)
        freq[pair] = max_count

    return freq


def get_pairwise_token_frequencies_and_heap_numpy(
    tokens: NumpyTokenSequence,
) -> Tuple[
        dict[TokenPair, TokenPairNode],
        list[TokenPairNode],
    ]:
    """Compute the token frequencies using numpy broadcast, and arrange them in a heap."""
    freq: dict[TokenPair, TokenPairNode] = {}
    heap: list[TokenPairNode] = []

    # Determine all unique pairs using bit packing
    y = tokens[:-1] * TOKEN_VALUE_UBOUND + tokens[1:]  # TODO(dtag): Tokens over ~2k will overflow int32
    unique_values, counts = np.unique(y, return_counts=True)

    # Efficiently unpack them
    first_tokens = np.floor_divide(unique_values, TOKEN_VALUE_UBOUND)
    second_tokens = np.mod(unique_values, TOKEN_VALUE_UBOUND)

    # Package the frequencies
    for token_1, token_2, count in zip(first_tokens, second_tokens, counts, strict=True):
        pair = TokenPair(token_1, token_2)
        node = TokenPairNode(token_1, token_2, count=count)
        heap.append(node)
        freq[pair] = node

    heapq.heapify(heap)

    return freq, heap
