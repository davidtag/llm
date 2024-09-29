import cython
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray


#@cython.boundscheck(False)
#@cython.wraparound(False)
def get_pairwise_frequencies(
    int[::1] input_sequence
) -> Dict[
        Tuple[int, int],
        int,
    ]:
    """Compute the frequency of bigram tokens in the provided sequence."""
    freq: Dict[Tuple[int, int], int] = {}

    cdef Py_ssize_t i
    cdef Py_ssize_t n = input_sequence.shape[0]

    for i in range(n - 1):
        byte_1 = input_sequence[i]
        byte_2 = input_sequence[i + 1]
        pair = (byte_1, byte_2)
        if i == 0:
            freq[pair] = freq.get(pair, 0) + 1

    return freq


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
