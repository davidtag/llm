"""Bechmarks for frequencies.pyx."""

import time

import numpy as np

from llm.tokenizers.frequencies import (
    get_pairwise_token_frequencies_sequential_pure_python,
    get_pairwise_token_frequencies_sequential_cython,
    get_pairwise_token_frequencies_numpy,
    get_pairwise_token_frequencies_numpy_maxonly,
    get_pairwise_token_frequencies_numpy_bitshift,
    get_pairwise_token_frequencies_numpy_bitshift_maxonly,
    get_pairwise_token_frequencies_and_heap_numpy,
)


class Profile:
    """Helper context manager for profiling code."""

    def __init__(self) -> None:
        self.start = 0
        self.end = 0
        self.duration = 0

    def __enter__(self):
        """Enter the context."""
        self.start = time.monotonic()
        return self

    def __exit__(self, type, value, traceback):
        """Exit the context."""
        self.end = time.monotonic()
        self.duration = self.end - self.start

    def avg(self, num_runs: int) -> None:
        self.duration /= num_runs

    @property
    def seconds(self) -> float:
        return self.duration

    @property
    def milliseconds(self) -> float:
        return self.seconds * 1000.0

    @property
    def milliseconds_formatted(self) -> str:
        return f"{self.milliseconds:.1f}ms"


def main() -> None:
    """Run the benchmark."""
    print("-- Running benchmark for frequencies.pyx ---------------------------------------")
    tokens_1mb = np.random.randint(low=0, high=256, size=1024 * 1024).astype(dtype=np.uint32)
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
        prof.avg(num_runs)
        print(f"{method.__name__}: {prof.milliseconds_formatted}")


if __name__ == "__main__":
    main()
