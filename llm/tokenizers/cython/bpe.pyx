"""Utilities for implementing the Byte-Pair Encoding (BPE) algorithm."""

import cython

from llm.tokenizers.cython.frequencies import get_pairwise_tokens
from llm.tokenizers.cython.merge import merge
from llm.tokenizers.cython.stdtoken import TokenPair


def encode_tokens(tokens: list[int], merge_dict: dict[TokenPair, int]) -> list[int]:
    """Encode a sequence of tokens according to a merge priority."""
    while len(tokens) > 1:
        all_pairs = get_pairwise_tokens(tokens)
        merge_pair = min(all_pairs, key=lambda p: merge_dict.get(p, float("inf")))  # token value = priority
        replacement = merge_dict.get(merge_pair, None)
        if replacement is None:
            break
        tokens = merge(tokens, merge_pair, replacement)
    return tokens


def encode_piece(piece: str, merge_dict: dict[TokenPair, int]) -> list[int]:
    """Encode text according to a merge priority."""
    piece_bytes = piece.encode("utf-8")
    piece_base_tokens = list(piece_bytes)
    piece_tokens = encode_tokens(piece_base_tokens, merge_dict=merge_dict)
    return piece_tokens


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
