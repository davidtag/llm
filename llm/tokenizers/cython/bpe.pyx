"""Utilities for implementing the Byte-Pair Encoding (BPE) algorithm."""

import cython


@cython.boundscheck(False)
@cython.wraparound(False)
def decode_bytes(tokens: list[int], vocab: list[bytes]) -> bytes:
    """Decode a list of tokens into bytes."""

    cdef Py_ssize_t n = len(tokens)
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t token

    buffer = bytearray()

    for i in range(n):
        token = tokens[i]
        token_bytes = vocab[token]
        buffer.extend(token_bytes)

    return bytes(buffer)
