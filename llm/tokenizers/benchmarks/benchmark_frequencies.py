"""Benchmarks for frequencies.pyx."""

from collections import defaultdict
from typing import Callable

import numpy as np

from llm.tokenizers.cython.frequencies import (
    get_pairwise_token_frequencies_sequential_pure_python,
    get_pairwise_token_frequencies_from_list,
    get_pairwise_token_frequencies_cython_loop,
    get_pairwise_token_frequencies_numpy,
    get_pairwise_token_frequencies_numpy_maxonly,
    get_pairwise_token_frequencies_numpy_bitshift,
    get_pairwise_token_frequencies_numpy_bitshift_maxonly,
    get_pairwise_token_frequencies_and_heap_numpy,
    get_masked_pairwise_token_frequencies_and_heap_numpy,
)
from llm.tokenizers.cython.stdtoken import TokenPair, TokenPairNode
from llm.tokenizers.cython.pytoken import TokenDtype, MaskedTokenDtype, NumpyTokenSequence
from llm.utils.profile import Profile


def get_pairwise_token_frequencies_from_list_wrapped(
    tokens: NumpyTokenSequence,
) -> defaultdict[TokenPair, int]:
    """Call the underlying method with a list conversion."""
    return get_pairwise_token_frequencies_from_list(tokens.tolist())


def get_masked_pairwise_token_frequencies_and_heap_numpy_wrapped(
    tokens: NumpyTokenSequence,
) -> tuple[dict[TokenPair, TokenPairNode], list[TokenPairNode]]:
    """Call the underlying method with no masked positions."""
    return get_masked_pairwise_token_frequencies_and_heap_numpy(
        tokens.astype(dtype=MaskedTokenDtype),
        masked_positions=np.array([], dtype=MaskedTokenDtype),
    )


def main() -> None:
    """Run the benchmark."""
    print("-- Running benchmark for frequencies.pyx ---------------------------------------")
    tokens_1mb = np.random.randint(low=0, high=256, size=1024 * 1024).astype(dtype=TokenDtype)
    num_runs = 10
    methods: list[Callable] = [
        get_pairwise_token_frequencies_sequential_pure_python,
        get_pairwise_token_frequencies_from_list_wrapped,
        get_pairwise_token_frequencies_cython_loop,
        get_pairwise_token_frequencies_numpy,
        get_pairwise_token_frequencies_numpy_maxonly,
        get_pairwise_token_frequencies_numpy_bitshift,
        get_pairwise_token_frequencies_numpy_bitshift_maxonly,
        get_pairwise_token_frequencies_and_heap_numpy,
        get_masked_pairwise_token_frequencies_and_heap_numpy_wrapped,
    ]
    for method in methods:
        with Profile() as prof:
            for _ in range(num_runs):
                method(tokens_1mb)
        prof.scale_by(num_runs)
        print(f"{method.__name__}: {prof.milliseconds_formatted}")


if __name__ == "__main__":
    main()
