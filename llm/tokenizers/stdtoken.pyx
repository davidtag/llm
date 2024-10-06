"""Data types for handling tokenization."""

cdef class TokenPair:
    """Stores a pair of tokens.

    Requires about 1/2 as much memory and runs in 1/2 the time when construction and
    storing these objects in a Python dict, as compared to a regular tuple.
    """

    def __cinit__(self, Token first, Token second):
        self.first = first
        self.second = second
