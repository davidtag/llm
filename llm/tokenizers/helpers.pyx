# distutils: language = c++
import cython
from typing import Dict, Tuple

from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair

import collections
import dataclasses
import heapq

import numpy as np
from numpy.typing import NDArray



# #@cython.boundscheck(False)
# #@cython.wraparound(False)
# def get_pairwise_token_frequencies(
#     int[::1] input_sequence,
#     bool max_only,
# ) -> None:
# #) -> Dict[
# #         Tuple[int, int],
# #         int,
# #     ]:
#     """Compute the frequency of bigram tokens in the provided sequence."""
#     cdef unordered_map[pair[int, int], int] freq
#     cdef pair[int, int] bigram
#     cdef int count

#     cdef Py_ssize_t i
#     cdef Py_ssize_t n = input_sequence.shape[0]

#     for i in range(n - 1):
#         bigram.first = input_sequence[i]
#         bigram.second = input_sequence[i + 1]
#         if freq.find(bigram) == freq.end():
#             freq[bigram] = 1
#         else:
#             freq[bigram] += 1
#         #freq[next_pair] = freq.get(next_pair, 0) + 1


#     #out: Dict[Tuple[int, int], int] = {}
#     #max_count = max(freq.values())
#     #for key, val in freq.items():
#     #    if val == max_count:
#     #        out[key] = val

#     #return out



@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _c_merge_inplace(
    int[::1] input_sequence,
    int token_1,
    int token_2,
    int output_token,
):
    """Implements core logic of merge_inplace() using pure C-land operations.

    Returns: size of the output array
    """
    cdef Py_ssize_t n = input_sequence.shape[0]
    cdef Py_ssize_t read_index = 0
    cdef Py_ssize_t write_index = 0

    while read_index < n:
        if (
            read_index < n - 1
            and input_sequence[read_index] == token_1
            and input_sequence[read_index + 1] == token_2
        ):
            input_sequence[write_index] = output_token
            read_index += 2
        else:
            input_sequence[write_index] = input_sequence[read_index]
            read_index += 1

        write_index += 1

    return write_index


@cython.boundscheck(False)
@cython.wraparound(False)
def merge_inplace(
    input_sequence: NDArray[np.int32],
    pair: Tuple[np.int32, np.int32],
    output_token: np.int32,
) -> NDArray[np.int32]:
    """Replace all occurences of `pair` in `input_sequence` with `output_token`.

    Returns:
        A view into the original array with replaced tokens and updated length.
    """
    new_size = _c_merge_inplace(
        input_sequence,
        pair[0],
        pair[1],
        output_token,
    )
    return input_sequence[:new_size]


@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _c_merge_inplace_and_report_neighbors(
    int[::1] input_sequence,
    int token_1,
    int token_2,
    int output_token,
    int[::1] prefix_neighbors,
    int[::1] suffix_neighbors,
):
    """Implements core logic of merge_inplace() using pure C-land operations.

    Returns: size of the output array
    """
    #print(f"_c_merge_inplace_and_report_neighbors({len(input_sequence)=}, {token_1=}, {token_2=}, {output_token=})")
    cdef Py_ssize_t n = input_sequence.shape[0]
    cdef Py_ssize_t read_index = 0
    cdef Py_ssize_t write_index = 0

    cdef Py_ssize_t num_merges = 0
    cdef bool just_merged = False

    while read_index < n:
        if (
            read_index < n - 1
            and input_sequence[read_index] == token_1
            and input_sequence[read_index + 1] == token_2
        ):

            # TODO(dtag): Gan speed this up by special-casing first & last token to avoid
            # a bunch of conditional branches and jump instructions
            if read_index >= 1:  # there exists a prefix
                if (
                    #write_index > 0
                    #and input_sequence[write_index - 1] == output_token
                    just_merged
                ):
                    assert input_sequence[write_index - 1] == output_token
                    prefix_neighbors[num_merges] = output_token
                else:
                    assert input_sequence[write_index - 1] != output_token
                    prefix_neighbors[num_merges] = input_sequence[read_index - 1]
                # prefix_neighbors[num_merges] = input_sequence[read_index - 1]

            if read_index <= n - 3:  # there exists a suffix
                if (
                    read_index < n - 3
                    and input_sequence[read_index + 2] == token_1
                    and input_sequence[read_index + 3] == token_2
                ):
                    suffix_neighbors[num_merges] = output_token
                else:
                    suffix_neighbors[num_merges] = input_sequence[read_index + 2]
                # suffix_neighbors[num_merges] = input_sequence[read_index + 2]

            just_merged = True
            num_merges += 1
            input_sequence[write_index] = output_token
            read_index += 2
        else:
            just_merged = False
            input_sequence[write_index] = input_sequence[read_index]
            read_index += 1

        write_index += 1

    #print(f"_c_merge_inplace_and_report_neighbors::{num_merges=}")
    return write_index

@cython.boundscheck(False)
@cython.wraparound(False)
def merge_inplace_and_update_frequencies(
    input_sequence: NDArray[np.int32],
    pair: Tuple[np.int32, np.int32],
    output_token: np.int32,
    int num_merges,
    frequencies: defaultdict[int, int],
) -> NDArray[np.int32]:
    """Replace all occurences of `pair` in `input_sequence` with `output_token`.

    Returns:
        A view into the original array with replaced tokens and updated length.
    """
    #print(f"merge_inplace_and_update_frequencies({len(input_sequence)=}, {pair=}, {output_token=}, {num_merges=}, {len(frequencies)=})")

    prefix_neighbors = np.zeros(shape=(num_merges,), dtype=np.int32)
    suffix_neighbors = np.zeros(shape=(num_merges,), dtype=np.int32)

    new_size = _c_merge_inplace_and_report_neighbors(
        input_sequence,
        pair[0],
        pair[1],
        output_token,
        prefix_neighbors,
        suffix_neighbors
    )
    effective_num_merges = len(input_sequence) - new_size  # TODO(dtag): why is this different?
    assert effective_num_merges <= num_merges
    assert effective_num_merges == num_merges or pair[0] == pair[1]

    prefix_values, prefix_counts = np.unique(prefix_neighbors[:effective_num_merges], return_counts=True)
    suffix_values, suffix_counts = np.unique(suffix_neighbors[:effective_num_merges], return_counts=True)

    for prefix, count in zip(prefix_values, prefix_counts, strict=True):
        # frequencies[(prefix, pair[0])] -= count
        # frequencies[(prefix, output_token)] += count
        if prefix == output_token:
            pass
        else:
            frequencies[(prefix, pair[0])] -= count
            frequencies[(prefix, output_token)] += count

    for suffix, count in zip(suffix_values, suffix_counts, strict=False):
        # frequencies[(pair[1], suffix)] -= count
        # frequencies[(output_token, suffix)] += count
        if suffix == output_token:
            frequencies[(pair[1], pair[0])] -= count
            frequencies[(output_token, output_token)] += count
        else:
            frequencies[(pair[1], suffix)] -= count
            frequencies[(output_token, suffix)] += count

    del frequencies[pair]

    # keys_to_delete = set()
    # for key, val in frequencies.items():
    #     if val == 0:
    #         keys_to_delete.add(key)
    # for key in keys_to_delete:
    #     del frequencies[key]

    return input_sequence[:new_size]


# MAX_NUM_MERGES = 500_000
# prefix_neighbors = np.zeros(shape=(MAX_NUM_MERGES,), dtype=np.int32)
# suffix_neighbors = np.zeros(shape=(MAX_NUM_MERGES,), dtype=np.int32)

@cython.boundscheck(False)
@cython.wraparound(False)
def merge_inplace_and_update_frequencies_and_heap(
    input_sequence: NDArray[np.int32],
    pair: Tuple[np.int32, np.int32],
    output_token: np.int32,
    int num_merges,
    frequencies: defaultdict[int, int],
    heap: list[Tuple[int, Tuple[np.int32, np.int32]]],
) -> NDArray[np.int32]:
    """Replace all occurences of `pair` in `input_sequence` with `output_token`.

    Returns:
        A view into the original array with replaced tokens and updated length.
    """
    #print(f"merge_inplace_and_update_frequencies({len(input_sequence)=}, {pair=}, {output_token=}, {num_merges=}, {len(frequencies)=})")

    # TODO: static initialize these and pass-in for better memory efficiency and latency
    prefix_neighbors = np.zeros(shape=(num_merges,), dtype=np.int32)
    suffix_neighbors = np.zeros(shape=(num_merges,), dtype=np.int32)

    new_size = _c_merge_inplace_and_report_neighbors(
        input_sequence,
        pair[0],
        pair[1],
        output_token,
        prefix_neighbors,
        suffix_neighbors
    )
    effective_num_merges = len(input_sequence) - new_size  # TODO(dtag): why is this different?
    assert effective_num_merges <= num_merges
    # if effective_num_merges > num_merges:
    #     import pdb
    #     pdb.set_trace()
    assert effective_num_merges == num_merges or pair[0] == pair[1], f"{effective_num_merges=}, {num_merges=}, {pair=}"
    # if effective_num_merges != num_merges and pair[0] != pair[1]:
    #     print(f"{effective_num_merges=}, {num_merges=}, {pair=}")
    #     import pdb
    #     pdb.set_trace()

    prefix_values, prefix_counts = np.unique(prefix_neighbors[:effective_num_merges], return_counts=True)
    suffix_values, suffix_counts = np.unique(suffix_neighbors[:effective_num_merges], return_counts=True)


    min_elem = heapq.heappop(heap)
    assert min_elem is frequencies[pair]
    del frequencies[pair]

    for prefix, count in zip(prefix_values, prefix_counts, strict=True):
        if prefix == output_token:
            pass
        else:
            #frequencies[(prefix, pair[0])] -= count
            #frequencies[(prefix, output_token)] += count
            update_freq_and_heap(frequencies, heap, (prefix, pair[0]), -count)
            update_freq_and_heap(frequencies, heap, (prefix, output_token), +count)

    for suffix, count in zip(suffix_values, suffix_counts, strict=False):
        if suffix == output_token:
            #frequencies[(pair[1], pair[0])] -= count
            #frequencies[(output_token, output_token)] += count
            update_freq_and_heap(frequencies, heap, (pair[1], pair[0]), -count)
            update_freq_and_heap(frequencies, heap, (output_token, output_token), +count)
        else:
            #frequencies[(pair[1], suffix)] -= count
            #frequencies[(output_token, suffix)] += count
            update_freq_and_heap(frequencies, heap, (pair[1], suffix), -count)
            update_freq_and_heap(frequencies, heap, (output_token, suffix), +count)


    # heapq.heapify(heap)  # rebalance


    return input_sequence[:new_size]


@dataclasses.dataclass(order=True)
class TokenPairElement:

    neg_count: int
    pair: TokenPair
    ignore: bool = False


@cython.boundscheck(False)
@cython.wraparound(False)
def update_freq_and_heap(
    frequencies,
    heap,
    pair,
    count,
):
    maybe_heap_elem = frequencies.get(pair, None)
    if maybe_heap_elem is not None:
        maybe_heap_elem.ignore = True

        new_count = maybe_heap_elem.neg_count-count
        if new_count != 0:
            heap_elem = TokenPairElement(neg_count=maybe_heap_elem.neg_count-count, pair=pair)
            heapq.heappush(heap, heap_elem)
            frequencies[pair] = heap_elem
        else:
            del frequencies[pair]

    else:
        heap_elem = TokenPairElement(neg_count=-count, pair=pair)
        heapq.heappush(heap, heap_elem)
        frequencies[pair] = heap_elem
