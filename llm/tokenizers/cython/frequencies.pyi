"""Various library methods for computing pairwise statistics over token sequences."""

from collections import defaultdict
from typing import Sequence

from llm.tokenizers.cython.stdtoken import TokenPair, TokenPairNode
from llm.tokenizers.cython.pytoken import NumpyTokenSequence, NumpyMaskedTokenSequence

def get_pairwise_token_frequencies_sequential_pure_python(
    tokens: Sequence[int],
) -> defaultdict[tuple[int, int], int]:
    """Compute the token frequencies using a sequential scan in pure-Python."""
    ...

def get_pairwise_tokens(
    tokens: list[int],
) -> set[TokenPair]:
    """Compute the unique neighboring token pairs."""
    ...

def get_pairwise_token_frequencies_from_list(
    tokens: list[int],
) -> defaultdict[TokenPair, int]:
    """Compute the token frequencies using a sequential scan in Cython."""
    ...

def get_pairwise_token_frequencies_cython_loop(
    tokens: memoryview,
) -> defaultdict[TokenPair, int]:
    """Compute the token frequencies using a sequential scan in Cython."""
    ...

def get_pairwise_token_frequencies_numpy(
    tokens: NumpyTokenSequence,
) -> defaultdict[TokenPair, int]:
    """Compute the token frequencies using numpy broadcast."""
    ...

def get_pairwise_token_frequencies_numpy_maxonly(
    tokens: NumpyTokenSequence,
) -> defaultdict[TokenPair, int]:
    """Compute the token frequencies using numpy broadcast, filtered for max-freq only."""
    ...

def get_pairwise_token_frequencies_numpy_bitshift(
    tokens: NumpyTokenSequence,
) -> defaultdict[TokenPair, int]:
    """Compute the token frequencies using numpy broadcast with bitshift.

    NOTE: this can be slighlty faster than the base numpy version but
    only allows for tokens up to 2^16 = 65536.
    """
    ...

def get_pairwise_token_frequencies_numpy_bitshift_maxonly(
    tokens: NumpyTokenSequence,
) -> defaultdict[TokenPair, int]:
    """Compute the token frequencies using numpy broadcast with bitshift, filtered for max-freq only.

    NOTE: this can be slighlty faster than the base numpy version but
    only allows for tokens up to 2^16 = 65536.
    """
    ...

def get_pairwise_token_frequencies_and_heap_numpy(
    tokens: NumpyTokenSequence,
) -> tuple[
    dict[TokenPair, TokenPairNode],
    list[TokenPairNode],
]:
    """Compute the token frequencies using numpy broadcast, and arrange them in a heap."""
    ...

def get_masked_pairwise_token_frequencies_and_heap_numpy(
    tokens_masked: NumpyMaskedTokenSequence,
    masked_positions: NumpyMaskedTokenSequence,
) -> tuple[
    dict[TokenPair, TokenPairNode],
    list[TokenPairNode],
]:
    """Compute the token frequencies using numpy broadcast, and arrange them in a heap."""
    ...
