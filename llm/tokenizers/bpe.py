"""Utilities for implementing the Byte-Pair Encoding (BPE) algorithm."""

from dataclasses import dataclass
import unicodedata

from llm.tokenizers.cython import bpe as _bpe_internal
from llm.tokenizers.cython.stdtoken import TokenPair


MergeList = list[tuple[TokenPair, int]]
MergeDict = dict[TokenPair, int]
Vocabulary = list[bytes]
ReverseVocabulary = dict[bytes, int]
PieceCache = dict[str, list[int]]


@dataclass
class PieceCacheCounterValue:
    """Helper for train_piece_cache().

    Represents the tokenization of a piece (regex text match) and its frequency.
    """

    tokens: list[int]
    count: int


def convert_merge_list_to_merge_dict(merge_list: MergeList) -> MergeDict:
    """Convert an ordered list of merge rules to a merge priority lookup."""
    merge_dict = {token_pair: replacement_token for token_pair, replacement_token in merge_list}
    return merge_dict


def convert_merge_list_to_vocabulary(merge_list: MergeList) -> Vocabulary:
    """Convert an ordered list of merge rules to a token vocabulary."""
    vocab_size = 256 + len(merge_list)
    vocab: Vocabulary = [b"" for _ in range(vocab_size)]

    for i in range(256):  # base tokens : 1-byte code points
        vocab[i] = bytes([i])

    for token_pair, replacement_token in merge_list:  # tokens minted during training
        vocab[replacement_token] = vocab[token_pair.first] + vocab[token_pair.second]

    return vocab


def convert_vocabulary_to_reverse_vocabulary(vocab: Vocabulary) -> ReverseVocabulary:
    """Convert a token vocabulary into an encoding lookup for byte fragments."""
    reverse_vocab = {token_bytes: token for token, token_bytes in enumerate(vocab)}
    return reverse_vocab


def convert_vocabulary_to_piece_cache(vocab: Vocabulary) -> PieceCache:
    """Convert a token vocabulary into an encoding lookup for text fragments (pieces)."""
    piece_cache: PieceCache = {}
    for token, token_bytes in enumerate(vocab):
        try:
            token_str = token_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue  # ignore invalid byte sequences
        else:
            piece_cache[token_str] = [token]
    return piece_cache


def encode_piece(piece: str, merge_dict: MergeDict) -> list[int]:
    """Encode text according to a merge priority."""
    return _bpe_internal.encode_piece(piece, merge_dict)


def decode_bytes(tokens: list[int], vocab: Vocabulary) -> bytes:
    """Decode a list of tokens into bytes."""
    # Equivalent but slower:
    #   return b"".join(vocab[token] for token in tokens)
    return _bpe_internal.decode_bytes(tokens, vocab)


def _replace_control_characters(s: str) -> str:
    """Escape control characters in a string.

    Ref: https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/base.py#L44
    """
    chars: list[str] = []

    for ch in s:
        if unicodedata.category(ch)[0] == "C":
            chars.append(f"\\u{ord(ch):04x}")  # escape
        else:
            chars.append(ch)  # this character is ok

    return "".join(chars)


def render_bytes(b: bytes) -> str:
    """Convert a sequence of bytes to a string, escaping control characters."""
    s = b.decode("utf-8", errors="replace")
    s = _replace_control_characters(s)
    return s
