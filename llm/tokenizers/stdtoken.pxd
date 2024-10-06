"""Data types for handling tokenization."""

from libc.stdint cimport uint32_t


ctypedef uint32_t Token
ctypedef Token[::1] TokenSequenece  # typed memory view. contiguous C-order memory.


cdef class TokenPair:

    cdef Token first
    cdef Token second
