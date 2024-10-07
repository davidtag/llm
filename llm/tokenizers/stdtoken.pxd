"""Cython data types for handling tokenization."""

from libc.stdint cimport uint32_t, int32_t


ctypedef uint32_t token_t
ctypedef token_t[::1] token_sequence_t  # typed memoryview, in contiguous C-order memory


cdef uint32_t MAX_TOKEN_VALUE


cdef class TokenPair:
    cdef token_t first
    cdef token_t second
    cdef token_t _unique


cdef class TokenPairNode:
    cdef token_t first
    cdef token_t second
    cdef int32_t count
    cdef bint ignore
