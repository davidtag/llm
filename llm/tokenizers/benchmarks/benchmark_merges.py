"""Benchmarks for interplay of frequencies.pyx and merge.pyx."""

import heapq

import numpy as np

from llm.tokenizers.cython.frequencies import (
    get_pairwise_token_frequencies_numpy,
    get_pairwise_token_frequencies_numpy_maxonly,
    get_pairwise_token_frequencies_and_heap_numpy,
)
from llm.tokenizers.cython.merge import (
    merge_inplace,
    merge_inplace_and_update_frequencies,
    merge_inplace_and_update_frequencies_and_heap,
)
from llm.tokenizers.cython.pytoken import TokenDtype, NumpyTokenSequence
from llm.utils.profile import Profile


def _repeated_freq_and_merge(tokens: NumpyTokenSequence, num_merges: int) -> int:
    next_token = 256
    for _ in range(num_merges):
        frequencies = get_pairwise_token_frequencies_numpy_maxonly(tokens)
        pair = min(frequencies.keys())
        tokens = merge_inplace(
            tokens=tokens,
            token_1=pair.first,
            token_2=pair.second,
            output_token=next_token,
        )
        next_token += 1
    return len(tokens)


def _merge_and_update_freq_inplace(tokens: NumpyTokenSequence, num_merges: int) -> int:
    next_token = 256
    frequencies = get_pairwise_token_frequencies_numpy(tokens)
    for _ in range(num_merges):
        pair = min(
            frequencies.keys(),
            key=lambda pair, freq=frequencies: (-freq[pair], pair),  # type: ignore[misc]
        )
        tokens = merge_inplace_and_update_frequencies(
            tokens=tokens,
            token_1=pair.first,
            token_2=pair.second,
            output_token=next_token,
            expected_num_merges=frequencies[pair],
            frequencies=frequencies,
        )
        next_token += 1
    return len(tokens)


def _merge_and_update_freq_inplace_with_heap(tokens: NumpyTokenSequence, num_merges: int) -> int:
    next_token = 256
    frequencies, heap = get_pairwise_token_frequencies_and_heap_numpy(tokens)
    for _ in range(num_merges):
        node = heap[0]
        while node.deleted:
            heapq.heappop(heap)
            node = heap[0]
        tokens = merge_inplace_and_update_frequencies_and_heap(
            tokens=tokens,
            token_1=node.first,
            token_2=node.second,
            output_token=next_token,
            expected_num_merges=node.count,
            frequencies=frequencies,
            heap=heap,
        )
        next_token += 1
    return len(tokens)


def main() -> None:
    """Run the benchmark."""
    print("-- Running benchmark for interplay of frequencies.pyx and merge.pyx -----------------------------")
    tokens_1mb = np.random.randint(low=0, high=256, size=1024 * 1024).astype(dtype=TokenDtype)
    num_runs = 1
    num_merges = 500
    methods = [
        _repeated_freq_and_merge,
        _merge_and_update_freq_inplace,
        _merge_and_update_freq_inplace_with_heap,
    ]
    final_sizes = set()
    for method in methods:
        with Profile() as prof:
            for _ in range(num_runs):
                final_size = method(np.copy(tokens_1mb), num_merges=num_merges)
                final_sizes.add(final_size)
        prof.scale_by(num_runs)
        print(f"{method.__name__}: {prof.milliseconds_formatted}")
    assert len(final_sizes) == 1


if __name__ == "__main__":
    main()
