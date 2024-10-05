"""Various library methods for computing pairwise statistics over token sequences."""
import cython
from libc.stdint cimport uint32_t

from collections import defaultdict
import dataclasses
import heapq
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


ctypedef uint32_t Token
ctypedef (Token, Token) TokenPair
ctypedef Token[::1] TokenSequenece  # typed memory view. contiguous C-order memory.
NumpyTokenSequence = NDArray[np.uint32]


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
    TokenSequenece tokens,
) -> defaultdict[TokenPair, int]:
    """Compute the token frequencies using a sequential scan in Cython."""
    freq: defaultdict[TokenPair, int] = defaultdict(int)

    cdef Py_ssize_t n = tokens.shape[0]
    cdef Py_ssize_t i
    cdef Token token_1, token_2
    cdef TokenPair pair

    for i in range(n - 1):
        token_1 = tokens[i]
        token_2 = tokens[i + 1]
        pair = (token_1, token_2)
        freq[pair] += 1

    return freq


cdef int _MAX_NUM_TOKENS = 1_000_000


def get_pairwise_token_frequencies_numpy(
    tokens: NumpyTokenSequence,
) -> defaultdict[TokenPair, int]:
    """Compute the token frequencies using numpy broadcast."""
    freq: defaultdict[TokenPair, int] = defaultdict(int)

    # Determine all unique pairs using bit packing
    y = tokens[:-1] * _MAX_NUM_TOKENS + tokens[1:]
    unique_values, counts = np.unique(y, return_counts=True)

    # Efficiently unpack them
    first_tokens = np.floor_divide(unique_values, _MAX_NUM_TOKENS)
    second_tokens = np.mod(unique_values, _MAX_NUM_TOKENS)

    # Package the frequencies
    for token_1, token_2, count in zip(first_tokens, second_tokens, counts, strict=True):
        pair = (token_1, token_2)
        freq[pair] = count

    return freq


def get_pairwise_token_frequencies_numpy_maxonly(
    tokens: NumpyTokenSequence,
) -> defaultdict[TokenPair, int]:
    """Compute the token frequencies using numpy broadcast, filtered for max-freq only."""
    freq: defaultdict[TokenPair, int] = defaultdict(int)

    # Determine all unique pairs using bit packing
    y = tokens[:-1] * _MAX_NUM_TOKENS + tokens[1:]
    unique_values, counts = np.unique(y, return_counts=True)

    # Determine the max count and all it's occurences
    max_count_idxs = np.argmax(counts, keepdims=True)
    max_count = counts[max_count_idxs[0]]

    # Efficiently unpack them
    first_tokens = np.floor_divide(unique_values[max_count_idxs], _MAX_NUM_TOKENS)
    second_tokens = np.mod(unique_values[max_count_idxs], _MAX_NUM_TOKENS)

    # Package the frequencies
    for token_1, token_2 in zip(first_tokens, second_tokens, strict=True):
        pair = (token_1, token_2)
        freq[pair] = max_count

    return freq


cdef int _BIT_SHIFT = 16
cdef Token MASK_UPPER_BITS = (1 << _BIT_SHIFT) - 1


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
        pair = (token_1, token_2)
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

    # Determine all unique pairs using bit packing
    # Upper half bits are the first token, with lower half bits the second token
    y = np.bitwise_or(np.left_shift(tokens[:-1], _BIT_SHIFT), tokens[1:])
    unique_values, counts = np.unique(y, return_counts=True)

    # Determine the max count and all it's occurences
    max_count_idxs = np.argmax(counts, keepdims=True)
    max_count = counts[max_count_idxs[0]]

    # Efficiently unpack them
    first_tokens = np.right_shift(unique_values[max_count_idxs], _BIT_SHIFT)
    second_tokens = np.bitwise_and(unique_values[max_count_idxs], MASK_UPPER_BITS)

    # Package the frequencies
    for token_1, token_2 in zip(first_tokens, second_tokens, strict=True):
        pair = (token_1, token_2)
        freq[pair] = max_count

    return freq


cdef class TokenPairNode:
    """A single node in a min-heap, representing a TokenPair and it's frequency in a TokenSequence.

    Implements `<` comparison to order by max-count, with tie-breaking by token values in ascending order.
    """

    cdef int count
    cdef TokenPair pair
    cdef bint ignore

    def __cinit__(
        self,
        int count,
        # note: I needed to accept these as separate args because when accepting a TokenPair, I
        # get a [-Wmaybe-uninitialized] compiler warning on the generated Cython code.
        Token token_1,
        Token token_2,
        bint ignore = False
    ):
        self.count = count
        self.pair = (token_1, token_2)
        self.ignore = ignore

    def __lt__(self, other: TokenPairNode) -> bool:
        self_order = (-self.count, self.pair[0], self.pair[1])
        other_order = (-other.count, other.pair[0], other.pair[1])
        return  self_order <  other_order


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
    y = tokens[:-1] * _MAX_NUM_TOKENS + tokens[1:]
    unique_values, counts = np.unique(y, return_counts=True)

    # Efficiently unpack them
    first_tokens = np.floor_divide(unique_values, _MAX_NUM_TOKENS)
    second_tokens = np.mod(unique_values, _MAX_NUM_TOKENS)

    # Package the frequencies
    for token_1, token_2, count in zip(first_tokens, second_tokens, counts, strict=True):
        pair = (token_1, token_2)
        heap_node = TokenPairNode(count=count, token_1=token_1, token_2=token_2)
        heap.append(heap_node)
        freq[pair] = heap_node

    heapq.heapify(heap)

    return freq, heap
