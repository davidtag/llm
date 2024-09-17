"""Utilities for implementing the Byte-Pair Encoding (BPE) algorithm."""

from llm.tokenizers.cython.stdtoken import TokenPair

def encode_tokens(tokens: list[int], merge_dict: dict[TokenPair, int]) -> list[int]:
    """Encode a sequence of tokens according to a merge priority."""
    ...

def encode_piece(piece: str, merge_dict: dict[TokenPair, int]) -> list[int]:
    """Encode text according to a merge priority."""
    ...

def decode_bytes(tokens: list[int], vocab: list[bytes]) -> bytes:
    """Decode a list of tokens into bytes."""
    ...
