"""Various library methods for performing BPE merge operations."""

from collections import defaultdict

from llm.tokenizers.cython.pytoken import NumpyTokenSequence
from llm.tokenizers.cython.stdtoken import TokenPair, TokenPairNode

def merge(
    tokens: list[int],
    pair: TokenPair,
    replacement: int,
) -> list[int]:
    """Replace all occurences of `pair` with `replacement`.

    Returns:
        A new list with all possible replacements performed.
    """
    ...

def merge_inplace(
    tokens: NumpyTokenSequence,
    token_1: int,
    token_2: int,
    output_token: int,
) -> NumpyTokenSequence:
    """Replace all occurences of `(token_1, token_2)` with `output_token` in-place.

    Returns:
        A view into the original array with replaced tokens and updated length.
    """
    ...

def merge_inplace_and_update_frequencies(
    tokens: NumpyTokenSequence,
    token_1: int,
    token_2: int,
    output_token: int,
    expected_num_merges: int,
    frequencies: defaultdict[TokenPair, int],
) -> NumpyTokenSequence:
    """Replace all occurences of `(token_1, token_2)` with `output_token` in-place.

    Additionally, update `frequencies` to reflect the merges performed

    Returns:
        A view into the original array with replaced tokens and updated length.
    """
    ...

def merge_inplace_and_update_frequencies_and_heap(
    tokens: NumpyTokenSequence,
    token_1: int,
    token_2: int,
    output_token: int,
    expected_num_merges: int,
    frequencies: dict[TokenPair, TokenPairNode],
    heap: list[TokenPairNode],
) -> NumpyTokenSequence:
    """Replace all occurences of `(token_1, token_2)` with `output_token` in-place.

    Additionally, update `frequencies` and `heap` to reflect the merges performed

    Returns:
        A view into the original array with replaced tokens and updated length.
    """
    ...
