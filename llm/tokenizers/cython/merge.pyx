"""Various library methods for performing BPE merge operations."""

from llm.tokenizers.cython.stdtoken cimport token_t, token_sequence_t, TokenPair, TokenPairNode

import heapq

import cython
import numpy as np
from numpy.typing import NDArray

from llm.tokenizers.cython.pytoken import TokenDtype, NumpyTokenSequence


def merge(
    tokens: list[int],
    pair: TokenPair,
    replacement: int,
) -> list[int]:
    """Replace all occurences of `pair` with `replacement`.

    Returns:
        A new list with all possible replacements performed.
    """
    output_tokens = []

    cdef Py_ssize_t n = len(tokens)
    cdef Py_ssize_t read_index = 0

    while read_index < n:
        if (
            read_index < n - 1
            and tokens[read_index] == pair.first
            and tokens[read_index + 1] == pair.second
        ):
            output_tokens.append(replacement)
            read_index += 2
        else:
            output_tokens.append(tokens[read_index])
            read_index += 1

    return output_tokens


@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
cdef Py_ssize_t _c_merge_inplace(
    token_sequence_t tokens,
    token_t token_1,
    token_t token_2,
    token_t output_token,
):
    """Replace all occurences of `(token_1, token_2)` with `output_token` in-place.

    Returns: the new effective size of `tokens`
    """
    cdef Py_ssize_t n = tokens.shape[0]
    cdef Py_ssize_t read_index = 0
    cdef Py_ssize_t write_index = 0

    while read_index < n:
        if (
            read_index < n - 1
            and tokens[read_index] == token_1
            and tokens[read_index + 1] == token_2
        ):
            tokens[write_index] = output_token
            read_index += 2
        else:
            tokens[write_index] = tokens[read_index]
            read_index += 1

        write_index += 1

    return write_index


def merge_inplace(
    tokens: NumpyTokenSequence,
    token_t token_1,
    token_t token_2,
    token_t output_token,
) -> NumpyTokenSequence:
    """Replace all occurences of `(token_1, token_2)` with `output_token` in-place.

    Returns:
        A view into the original array with replaced tokens and updated length.
    """
    new_size = _c_merge_inplace(
        tokens,
        token_1,
        token_2,
        output_token,
    )
    return tokens[:new_size]


@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
cdef (Py_ssize_t, Py_ssize_t, Py_ssize_t) _c_merge_inplace_and_report_neighbors(
    token_sequence_t tokens,
    token_t token_1,
    token_t token_2,
    token_t output_token,
    token_sequence_t prefix_neighbors,
    token_sequence_t suffix_neighbors,
):
    """Replace all occurences of `(token_1, token_2)` with `output_token` in-place.

    Additionally, fills `prefix_neighbors` and `suffix_neighbors` with the tokens immediately
    before and after the merged tokens in the input sequence.

    Returns: the new effective size of `tokens` and the number of prefix and suffix tokens
    """
    cdef Py_ssize_t n = tokens.shape[0]
    cdef Py_ssize_t read_index = 0
    cdef Py_ssize_t write_index = 0

    cdef Py_ssize_t num_prefix = 0
    cdef Py_ssize_t num_suffix = 0
    cdef bint previous_tokens_merged = False

    while read_index < n:
        if (
            read_index < n - 1
            and tokens[read_index] == token_1
            and tokens[read_index + 1] == token_2
        ):
            # TODO(dtag): Can likely speed this up by special-casing first & last token to avoid
            # a bunch of conditional branches and jump instructions
            if read_index >= 1:  # there exists a prefix
                if (
                    previous_tokens_merged
                ):
                    assert tokens[write_index - 1] == output_token
                    prefix_neighbors[num_prefix] = output_token
                else:
                    assert tokens[write_index - 1] != output_token
                    prefix_neighbors[num_prefix] = tokens[read_index - 1]
                num_prefix += 1

            if read_index < n - 2:  # there exists a suffix
                if (
                    read_index < n - 3
                    and tokens[read_index + 2] == token_1
                    and tokens[read_index + 3] == token_2
                ):
                    suffix_neighbors[num_suffix] = output_token
                else:
                    suffix_neighbors[num_suffix] = tokens[read_index + 2]
                num_suffix += 1

            previous_tokens_merged = True
            tokens[write_index] = output_token
            read_index += 2
        else:
            previous_tokens_merged = False
            tokens[write_index] = tokens[read_index]
            read_index += 1

        write_index += 1

    return (write_index, num_prefix, num_suffix)


def merge_inplace_and_update_frequencies(
    tokens: NumpyTokenSequence,
    token_t token_1,
    token_t token_2,
    token_t output_token,
    int expected_num_merges,
    frequencies: defaultdict[TokenPair, int],
) -> NumpyTokenSequence:
    """Replace all occurences of `(token_1, token_2)` with `output_token` in-place.

    Additionally, update `frequencies` to reflect the merges performed

    Returns:
        A view into the original array with replaced tokens and updated length.
    """
    # TODO(dtag): static initialize these arrays and pass-in for better memory efficiency and latency
    prefix_neighbors = np.zeros(shape=(expected_num_merges,), dtype=TokenDtype)
    suffix_neighbors = np.zeros(shape=(expected_num_merges,), dtype=TokenDtype)

    new_size, num_prefix, num_suffix = _c_merge_inplace_and_report_neighbors(
        tokens,
        token_1,
        token_2,
        output_token,
        prefix_neighbors,
        suffix_neighbors
    )

    effective_num_merges = len(tokens) - new_size
    assert effective_num_merges <= expected_num_merges
    assert effective_num_merges == expected_num_merges or token_1 == token_2

    prefix_values, prefix_counts = np.unique(prefix_neighbors[:num_prefix], return_counts=True)
    suffix_values, suffix_counts = np.unique(suffix_neighbors[:num_suffix], return_counts=True)

    for prefix, count in zip(prefix_values, prefix_counts, strict=True):
        if prefix == output_token:  # successive merges. special-casing handled with suffix tokens
            pass
        else:
            frequencies[TokenPair(prefix, token_1)] -= count
            frequencies[TokenPair(prefix, output_token)] += count

    for suffix, count in zip(suffix_values, suffix_counts, strict=False):
        if suffix == output_token:  # successive merges
            frequencies[TokenPair(token_2, token_1)] -= count
            frequencies[TokenPair(output_token, output_token)] += count
        else:
            frequencies[TokenPair(token_2, suffix)] -= count
            frequencies[TokenPair(output_token, suffix)] += count

    pair = TokenPair(token_1, token_2)
    if pair in frequencies:
        del frequencies[pair]

    return tokens[:new_size]


def _update_freq_and_heap(
    frequencies: dict[TokenPair, TokenPairNode],
    heap: list[TokenPairNode],
    pair: TokenPair,
    int diff,
):
    """Update the count of `pair` in `frequencies` and `heap` by `diff`."""
    maybe_heap_elem = frequencies.get(pair, None)
    if maybe_heap_elem is not None:
        # This pair already exists. To minimize heap operations at the expense of some extra memory,
        # we leave the existing node but mark it deleted, and create a new one. Client code needs
        # to inspect the deleted field of pop()ed nodes before using.
        maybe_heap_elem.deleted = True

        new_count = maybe_heap_elem.count + diff
        if new_count == 0:
            del frequencies[pair]
        else:
            node = TokenPairNode(first=pair.first, second=pair.second, count=new_count)
            heapq.heappush(heap, node)
            frequencies[pair] = node

    else:
        # First time seeing this pair. Create a node for it.
        node = TokenPairNode(first=pair.first, second=pair.second, count=diff)
        heapq.heappush(heap, node)
        frequencies[pair] = node


# To support re-use of the method below for the masked input token sequence, without re-implementing it.
cdef token_t MASKED_TOKEN = <token_t>(-1)


def merge_inplace_and_update_frequencies_and_heap(
    tokens: NumpyTokenSequence,
    token_t token_1,
    token_t token_2,
    token_t output_token,
    int expected_num_merges,
    frequencies: dict[TokenPair, TokenPairNode],
    heap: list[TokenPairNode],
) -> NumpyTokenSequence:
    """Replace all occurences of `(token_1, token_2)` with `output_token` in-place.

    Additionally, update `frequencies` and `heap` to reflect the merges performed

    Returns:
        A view into the original array with replaced tokens and updated length.
    """
    # TODO(dtag): static initialize these arrays and pass-in for better memory efficiency and latency
    prefix_neighbors = np.zeros(shape=(expected_num_merges,), dtype=TokenDtype)
    suffix_neighbors = np.zeros(shape=(expected_num_merges,), dtype=TokenDtype)

    new_size, num_prefix, num_suffix = _c_merge_inplace_and_report_neighbors(
        tokens,
        token_1,
        token_2,
        output_token,
        prefix_neighbors,
        suffix_neighbors
    )

    effective_num_merges = len(tokens) - new_size
    assert effective_num_merges <= expected_num_merges
    assert effective_num_merges == expected_num_merges or token_1 == token_2

    prefix_values, prefix_counts = np.unique(prefix_neighbors[:num_prefix], return_counts=True)
    suffix_values, suffix_counts = np.unique(suffix_neighbors[:num_suffix], return_counts=True)

    pair = TokenPair(token_1, token_2)
    if len(heap) > 0:
        min_elem = heapq.heappop(heap)  # pop min element before further operations
        assert min_elem.pair == pair
        assert min_elem is frequencies[pair]
        del frequencies[pair]

    for prefix, count in zip(prefix_values, prefix_counts, strict=True):
        if prefix == output_token:  # successive merges. special-casing handled with suffix tokens
            pass
        elif prefix == MASKED_TOKEN:  # freq/heap ignore masked positions
            pass
        else:
            _update_freq_and_heap(frequencies, heap, TokenPair(prefix, token_1), -count)
            _update_freq_and_heap(frequencies, heap, TokenPair(prefix, output_token), +count)

    for suffix, count in zip(suffix_values, suffix_counts, strict=False):
        if suffix == output_token:  # successive merges
            _update_freq_and_heap(frequencies, heap, TokenPair(token_2, token_1), -count)
            _update_freq_and_heap(frequencies, heap, TokenPair(output_token, output_token), +count)
        elif suffix == MASKED_TOKEN:  # freq/heap ignore masked positions
            pass
        else:
            _update_freq_and_heap(frequencies, heap, TokenPair(token_2, suffix), -count)
            _update_freq_and_heap(frequencies, heap, TokenPair(output_token, suffix), +count)

    # Target pair may have been recreated if token_1==token_2
    if pair in frequencies:
        node = frequencies[pair]
        assert node.count < 0
        node.deleted = True
        del frequencies[pair]

    return tokens[:new_size]
