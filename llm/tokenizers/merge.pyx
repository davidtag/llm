"""Various library methods for performing BPE merge operations."""

from llm.tokenizers.stdtoken cimport token_t, token_sequence_t, TokenPair, TokenPairNode

import heapq

import cython
import numpy as np
from numpy.typing import NDArray

from llm.tokenizers.pytoken import TokenDtype, NumpyTokenSequence


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

    Returns: the new effective size of `tokens`
    """
    cdef Py_ssize_t n = tokens.shape[0]
    cdef Py_ssize_t read_index = 0
    cdef Py_ssize_t write_index = 0

    cdef Py_ssize_t num_merges = 0
    cdef Py_ssize_t num_prefix = 0
    cdef Py_ssize_t num_suffix = 0
    cdef bint previous_tokens_merged = False

    while read_index < n:
        if (
            read_index < n - 1
            and tokens[read_index] == token_1
            and tokens[read_index + 1] == token_2
        ):

            # TODO(dtag): Gan speed this up by special-casing first & last token to avoid
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
            num_merges += 1
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

    Returns: the new effective size of `tokens`
    """
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
            frequencies[(prefix, token_1)] -= count
            frequencies[(prefix, output_token)] += count

    for suffix, count in zip(suffix_values, suffix_counts, strict=False):
        if suffix == output_token:  # successive merges
            frequencies[(token_2, token_1)] -= count
            frequencies[(output_token, output_token)] += count
        else:
            frequencies[(token_2, suffix)] -= count
            frequencies[(output_token, suffix)] += count

    pair = (token_1, token_2)
    if pair in frequencies:
        del frequencies[pair]

    keys_to_delete = set()
    for key, val in frequencies.items():
        if val == 0:
            keys_to_delete.add(key)
    for key in keys_to_delete:
        del frequencies[key]

    return tokens[:new_size]


def _update_freq_and_heap(
    frequencies: dict[TokenPair, TokenPairNode],
    heap: list[TokenPairNode],
    pair: Tuple[int, int],  # TODO(dtag): Create a cdef class for TokenPair
    int diff,
):
    """Update the count of `pair` in `frequencies` and `heap` by `diff`."""
    maybe_heap_elem = frequencies.get(pair, None)
    if maybe_heap_elem is not None:
        # This pair already exists. To minimize heap operations at the expense of some extra memory,
        # we leave the existing node but mark it ignored, and create a new one. Client code needs
        # to inspect the deleted field of pop()ed nodes before using.
        maybe_heap_elem.deleted = True

        new_count = maybe_heap_elem.count + diff
        if new_count == 0:
            del frequencies[pair]
        else:
            heap_node = TokenPairNode(count=new_count, token_1=pair[0], token_2=pair[1])
            heapq.heappush(heap, heap_node)
            frequencies[pair] = heap_node

    else:
        # First time seeing this pair. Create a node for it.
        heap_node = TokenPairNode(count=diff, token_1=pair[0], token_2=pair[1])
        heapq.heappush(heap, heap_node)
        frequencies[pair] = heap_node


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

    Returns: the new effective size of `tokens`
    """
    # TODO(dtag): static initialize these initialize these and pass-in for better memory efficiency and latency
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

    pair = (token_1, token_2)
    if len(heap) > 0:
        min_elem = heapq.heappop(heap)  # pop min element before further operations
        assert min_elem.pair == pair
        assert min_elem is frequencies[pair]
        del frequencies[pair]

    for prefix, count in zip(prefix_values, prefix_counts, strict=True):
        if prefix == output_token:  # successive merges. special-casing handled with suffix tokens
            pass
        else:
            _update_freq_and_heap(frequencies, heap, (prefix, token_1), -count)
            _update_freq_and_heap(frequencies, heap, (prefix, output_token), +count)

    for suffix, count in zip(suffix_values, suffix_counts, strict=False):
        if suffix == output_token:  # successive merges
            _update_freq_and_heap(frequencies, heap, (token_2, token_1), -count)
            _update_freq_and_heap(frequencies, heap, (output_token, output_token), +count)
        else:
            _update_freq_and_heap(frequencies, heap, (token_2, suffix), -count)
            _update_freq_and_heap(frequencies, heap, (output_token, suffix), +count)

    # Target pair may have been recreated if token_1==token_2
    if pair in frequencies:
        node = frequencies[pair]
        assert node.count < 0
        node.deleted = True
        del frequencies[pair]

    return tokens[:new_size]
