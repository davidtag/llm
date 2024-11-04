"""Benchmakrs for stdtoken.pyx."""

import gc
import os
import sys

import psutil

from llm.tokenizers.cython.stdtoken import TokenPair
from llm.utils.profile import Profile


NUM_ITEMS = 1_000_000


def _get_process_memory() -> float:
    process = psutil.Process(os.getpid())
    rss_memory = process.memory_info().rss
    rss_memory_mb = rss_memory / (1024 * 1024)
    return rss_memory_mb


def _construct_dict_of_tuple() -> dict[tuple[int, int], int]:
    d = {}
    for i in range(NUM_ITEMS):
        pair = (i, i + 1)
        d[pair] = 1
    return d


def _construct_dict_of_token_pair() -> dict[TokenPair, int]:
    d = {}
    for i in range(NUM_ITEMS):
        pair = TokenPair(i, i + 1)
        d[pair] = 1
    return d


def main() -> None:
    """Run the benchmark."""
    print("-- Running benchmark for stdtoken.pyx ---------------------------------------")

    print()
    print("- Primitive Type Benchmark ----------------------")
    base_int = 137
    tuple_pair = (137, 432)
    token_pair = TokenPair(137, 432)
    print(f"int: size={sys.getsizeof(base_int)} bytes")
    print(f"Tuple[int, int]: size={sys.getsizeof(tuple_pair)} bytes")
    print(f"TokenPair: size={sys.getsizeof(token_pair)} bytes")
    del base_int, tuple_pair, token_pair
    gc.collect()

    print()
    print("- Container Benchmark ---------------------------")
    num_runs = 10
    methods = [
        _construct_dict_of_tuple,
        _construct_dict_of_token_pair,
        _construct_dict_of_tuple,
        _construct_dict_of_token_pair,
    ]

    starting_memory = _get_process_memory()
    print(f"Starting Memory: {starting_memory:.1f} MB")

    for method in methods:
        method_start_memory = _get_process_memory()  # note: memory grows across loops

        with Profile() as prof:
            for _ in range(num_runs):
                d = method()

        assert isinstance(d, dict)

        prof.scale_by(num_runs)
        d_size = sys.getsizeof(d)
        method_memory_use = _get_process_memory() - method_start_memory
        print(
            f"{method.__name__:<30}: "
            f"{prof.milliseconds_formatted}  "
            f"{len(d)=:,}  "
            f"{d_size=:,}  "
            f"{method_memory_use=:.1f}MB"
        )

        del d, d_size, prof, method_start_memory, method_memory_use
        gc.collect()

    ending_memory = _get_process_memory()
    print(f"Ending Memory: {ending_memory:.1f} MB")


if __name__ == "__main__":
    main()
