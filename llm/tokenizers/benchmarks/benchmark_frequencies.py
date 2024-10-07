"""Benchmarks for frequencies.pyx."""

import numpy as np

from llm.tokenizers.benchmarks.profile import Profile
from llm.tokenizers.frequencies import (
    get_pairwise_token_frequencies_sequential_pure_python,
    get_pairwise_token_frequencies_sequential_cython,
    get_pairwise_token_frequencies_numpy,
    get_pairwise_token_frequencies_numpy_maxonly,
    get_pairwise_token_frequencies_numpy_bitshift,
    get_pairwise_token_frequencies_numpy_bitshift_maxonly,
    get_pairwise_token_frequencies_and_heap_numpy,
)
from llm.tokenizers.pytoken import TokenDtype


def main() -> None:
    """Run the benchmark."""
    print("-- Running benchmark for frequencies.pyx ---------------------------------------")
    tokens_1mb = np.random.randint(low=0, high=256, size=1024 * 1024).astype(dtype=TokenDtype)
    num_runs = 10
    methods = [
        get_pairwise_token_frequencies_sequential_pure_python,
        get_pairwise_token_frequencies_sequential_cython,
        get_pairwise_token_frequencies_numpy,
        get_pairwise_token_frequencies_numpy_maxonly,
        get_pairwise_token_frequencies_numpy_bitshift,
        get_pairwise_token_frequencies_numpy_bitshift_maxonly,
        get_pairwise_token_frequencies_and_heap_numpy,
    ]
    for method in methods:
        with Profile() as prof:
            for _ in range(num_runs):
                method(tokens_1mb)
        prof.scale_by(num_runs)
        print(f"{method.__name__}: {prof.milliseconds_formatted}")


if __name__ == "__main__":
    main()
