"""."""

from collections import defaultdict, OrderedDict
from concurrent.futures import Executor, ProcessPoolExecutor
import heapq
import os
import time
from typing import Dict, Optional, Sequence, Tuple

import dataclasses
import pprint
import numpy as np
from numpy.typing import NDArray

from llm.tokenizers.helpers import (
    merge_inplace,
    merge_inplace_and_update_frequencies,
    merge_inplace_and_update_frequencies_and_heap,
    TokenPairElement,
)

Token = np.int32
TokenPair = Tuple[Token, Token]
MergeList = OrderedDict[TokenPair, Token]
Vocab = Dict[Token, str]


MAX_TOKEN_VALUE = 1_000_000


def get_pairwise_frequencies_sequential(byte_sequence: Sequence[Token]) -> defaultdict[TokenPair, int]:
    """Compute the frequency of bigram tokens in the provided sequence."""
    freq: defaultdict[TokenPair, int] = defaultdict(int)

    # n = len(byte_sequence)
    # for i in range(n - 1):
    #     byte_1 = byte_sequence[i]
    #     byte_2 = byte_sequence[i + 1]
    #     pair = (byte_1, byte_2)
    #     freq[pair] = freq.get(pair, 0) + 1

    for c_1, c_2 in zip(byte_sequence[:-1], byte_sequence[1:]):
        pair = (c_1, c_2)
        freq[pair] += 1

    return freq


def _get_pairwise_frequencies(
    byte_sequence: Sequence[Token],
    executor: Optional[Executor] = None,
) -> Dict[TokenPair, int]:
    """Compute the frequency of bigram tokens in the provided sequence."""
    if executor is None:
        return get_pairwise_frequencies_sequential(byte_sequence)

    n = len(byte_sequence)

    num_chunks = executor._max_workers
    chunk_size = n // num_chunks

    futures = []
    for i in range(num_chunks):
        if i == num_chunks - 1:
            future = executor.submit(get_pairwise_frequencies_sequential, byte_sequence[i * chunk_size :])
        else:
            future = executor.submit(
                get_pairwise_frequencies_sequential, byte_sequence[i * chunk_size : (i + 1) * chunk_size]
            )
        futures.append(future)

    freq = {}

    for future in futures:
        this_freq = future.result()
        for key, val in this_freq.items():
            freq[key] = freq.get(key, 0) + val

    # TODO: calculate frequency at boundaries

    return freq


# def get_pairwise_token_frequencies(byte_sequence: NDArray[np.int32]) -> Dict[TokenPair, int]:
#     """Compute the frequency of bigram tokens in the provided sequence."""
#     freq: Dict[TokenPair, int] = {}

#     # y = byte_sequence[:-1] * MAX_TOKEN_VALUE + byte_sequence[1:]
#     # unique_values, frequencies = np.unique(y, return_counts=True)

#     # for pact_pair, count in zip(unique_values, frequencies, strict=True):
#     #     token_1 = pact_pair // MAX_TOKEN_VALUE
#     #     token_2 = pact_pair % MAX_TOKEN_VALUE
#     #     pair = (token_1, token_2)
#     #     freq[pair] = count

#     # shift = 16  # TODO(dtag): Only allows 65k tokens

#     # # Determine all unique pairs using bit packing
#     # # Lower `shift` bits are the second token, with upper bits the first token
#     # y = np.left_shift(byte_sequence[:-1], shift) + byte_sequence[1:]
#     # unique_values, frequencies = np.unique(y, return_counts=True)

#     # # Efficiently unpack them
#     # first_tokens = np.right_shift(unique_values, shift)
#     # mask = (1 << shift) - 1
#     # second_tokens = np.bitwise_and(unique_values, mask)

#     # # Package the frequencies
#     # for token_1, token_2, count in zip(first_tokens, second_tokens, frequencies, strict=True):
#     #     pair = (token_1, token_2)
#     #     freq[pair] = count

#     return freq


def get_pairwise_token_frequencies(
    token_sequence: NDArray[np.int32],
    max_only: bool = False,
) -> defaultdict[TokenPair, int]:
    """Compute the frequency of bigram tokens in the provided sequence.

    If `max_only=True`, only include the token pair with the highest frequency,
    including ties, in the output.
    """
    freq: defaultdict[TokenPair, int] = defaultdict(int)

    bit_shift = 16  # TODO(dtag): Only allows 65k tokens

    # Determine all unique pairs using bit packing
    # Lower `shift` bits are the second token, with upper bits the first token
    y = np.left_shift(token_sequence[:-1], bit_shift) + token_sequence[1:]
    unique_values, counts = np.unique(y, return_counts=True)

    # Efficiently unpack them
    # TODO(dtag): You can avoid doing this for all inputs in the case of max_only=True
    first_tokens = np.right_shift(unique_values, bit_shift)
    mask_upper_bits = (1 << bit_shift) - 1
    second_tokens = np.bitwise_and(unique_values, mask_upper_bits)

    # Package the frequencies
    if max_only:
        max_count = np.max(counts)
        for token_1, token_2, count in zip(first_tokens, second_tokens, counts, strict=True):
            if count == max_count:
                pair = (token_1, token_2)
                freq[pair] = count
    else:
        for token_1, token_2, count in zip(first_tokens, second_tokens, counts, strict=True):
            pair = (token_1, token_2)
            freq[pair] = count

    return freq


def get_pairwise_token_frequencies_and_heap(
    token_sequence: NDArray[np.int32],
) -> Tuple[Dict[TokenPair, TokenPairElement], list[TokenPairElement]]:
    """Compute the frequency of bigram tokens in the provided sequence.

    If `max_only=True`, only include the token pair with the highest frequency,
    including ties, in the output.
    """
    freq: Dict[TokenPair, TokenPairElement] = {}
    heap: list[TokenPairElement] = []

    bit_shift = 16  # TODO(dtag): Only allows 65k tokens

    # Determine all unique pairs using bit packing
    # Lower `shift` bits are the second token, with upper bits the first token
    y = np.left_shift(token_sequence[:-1], bit_shift) + token_sequence[1:]
    unique_values, counts = np.unique(y, return_counts=True)

    # Efficiently unpack them
    # TODO(dtag): You can avoid doing this for all inputs in the case of max_only=True
    first_tokens = np.right_shift(unique_values, bit_shift)
    mask_upper_bits = (1 << bit_shift) - 1
    second_tokens = np.bitwise_and(unique_values, mask_upper_bits)

    # Package the frequencies
    for token_1, token_2, count in zip(first_tokens, second_tokens, counts, strict=True):
        pair = (token_1, token_2)
        heap_elem = TokenPairElement(neg_count=-count, pair=pair)
        heap.append(heap_elem)
        freq[pair] = heap_elem

    heapq.heapify(heap)

    return freq, heap


# def merge_inplace(
#     byte_sequence: NDArray[np.int32],
#     pair: TokenPair,
#     output_token: Token,
# ) -> NDArray[np.int32]:
#     """Replace all occurences of `pair` in `input_sequence` with `output_token`."""
#     assert byte_sequence.ndim == 1
#     n = byte_sequence.shape[0]

#     read_index = 0
#     write_index = 0

#     while read_index < n:
#         if (
#             read_index < n - 1
#             and byte_sequence[read_index] == pair[0]
#             and byte_sequence[read_index + 1] == pair[1]
#         ):
#             byte_sequence[write_index] = output_token
#             read_index += 2
#         else:
#             byte_sequence[write_index] = byte_sequence[read_index]
#             read_index += 1

#         write_index += 1

#     return byte_sequence[:write_index]


class BasicTokenizer:
    """."""

    NUM_BASE_TOKENS = 256

    def __init__(self, merges: Optional[MergeList] = None) -> None:
        """Initialize the tokenizer."""
        self.merges = MergeList() if merges is None else merges
        self.merge_ranks = {pair: i for i, (pair, _) in enumerate(self.merges.items())}
        self.vocab = self.convert_merge_list_to_vocab(self.merges)

    def train(
        self,
        text: bytes,
        num_merges: int,
        verbose: bool = False,
    ) -> MergeList:
        """."""
        print("--------------")
        byte_sequence = np.array(list(text), dtype=np.int32)
        print("----")

        merges = MergeList()
        next_token = np.int32(self.NUM_BASE_TOKENS)

        for i in range(num_merges):
            # We only compress pairs occuring more than once, which requires at least 3 input elements
            if len(byte_sequence) < 3:
                break

            # Decide what to merge, if we can. We only merge pairs occuring atleast twice
            time_0 = time.monotonic()
            frequencies = get_pairwise_token_frequencies(byte_sequence, max_only=True)
            time_1 = time.monotonic()
            next_pair = min(frequencies.keys())  # tie-break equal-frequency pairs using pair indices
            if frequencies[next_pair] < 2:
                break
            time_2 = time.monotonic()

            # Merge it
            byte_sequence = merge_inplace(byte_sequence, pair=next_pair, output_token=next_token)
            time_3 = time.monotonic()
            merges[next_pair] = next_token
            self.vocab[next_token] = self.vocab[next_pair[0]] + self.vocab[next_pair[1]]
            next_token += 1
            time_4 = time.monotonic()

            if verbose:
                t_freq = time_1 - time_0
                t_max = time_2 - time_1
                t_merge = time_3 - time_2
                t_total = time_4 - time_0
                print(
                    f" {i + 1:6}/{num_merges} : "
                    f"Stats=[{t_freq=:.3f},{t_max=:.3f},{t_merge=:.3f},{t_total=:.3f}] "
                    "Merged: "
                    f"[{self.decode([next_pair[0]])}][{self.decode([next_pair[1]])}] -> [{next_token}]. "
                    f"freq={frequencies[next_pair]}. len={len(byte_sequence)}"
                )

        return merges

    def train2(
        self,
        text: bytes,
        num_merges: int,
        verbose: bool = False,
    ) -> MergeList:
        """."""
        print("--------------")
        byte_sequence = np.array(list(text), dtype=np.int32)
        print("----")

        merges = MergeList()
        next_token = np.int32(self.NUM_BASE_TOKENS)

        frequencies = get_pairwise_token_frequencies(byte_sequence, max_only=False)
        # frequencies = get_pairwise_frequencies_sequential(byte_sequence)

        for i in range(num_merges):
            time_0 = time.monotonic()

            # print(f"{len(frequencies)=}")
            time_1 = time.monotonic()

            # Decide what to merge, if we can. We only merge pairs occuring atleast twice
            next_pair = max(
                frequencies.keys(),
                key=lambda pair, freq=frequencies: (freq[pair], -pair[0], -pair[1]),
            )
            target_merges = frequencies[next_pair]
            # print(f"{target_merges=}")
            assert target_merges >= 0
            if target_merges < 2:
                break
            time_2 = time.monotonic()

            # Merge it
            byte_sequence = merge_inplace_and_update_frequencies(
                input_sequence=byte_sequence,
                pair=next_pair,
                output_token=next_token,
                num_merges=target_merges,
                frequencies=frequencies,
            )
            time_3 = time.monotonic()

            # Update vocab
            merges[next_pair] = next_token
            self.vocab[next_token] = self.vocab[next_pair[0]] + self.vocab[next_pair[1]]
            next_token += 1
            time_4 = time.monotonic()

            if verbose:
                t_freq = time_1 - time_0
                t_max = time_2 - time_1
                t_merge = time_3 - time_2
                t_total = time_4 - time_0
                print(
                    f" {i + 1:6}/{num_merges} : "
                    f"Stats=[{t_freq=:.3f},{t_max=:.3f},{t_merge=:.3f},{t_total=:.3f}] "
                    "Merged: "
                    f"[{self.decode([next_pair[0]])}][{self.decode([next_pair[1]])}] -> [{next_token}]. "
                    f"freq={target_merges}. len={len(byte_sequence)}"
                )

        return merges

    def train3(
        self,
        text: bytes,
        num_merges: int,
        verbose: bool = False,
    ) -> MergeList:
        """."""
        print("--------------")
        byte_sequence = np.array(list(text), dtype=np.int32)
        print("----")

        merges = MergeList()
        next_token = np.int32(self.NUM_BASE_TOKENS)

        frequencies, heap = get_pairwise_token_frequencies_and_heap(byte_sequence)
        print(len(frequencies), len(heap), heap[0])

        for i in range(num_merges):
            time_0 = time.monotonic()

            if len(heap) / len(frequencies) > 2:
                new_heap = [item for item in heap if not item.ignore]
                heapq.heapify(new_heap)
                heap = new_heap
                assert len(heap) == len(new_heap)
                print("!!CLEANNING")
            # print(f"{len(frequencies)=}")
            time_1 = time.monotonic()

            # Decide what to merge, if we can. We only merge pairs occuring atleast twicem
            # print(len(heap))
            min_elem = heap[0]
            while min_elem.ignore:
                heapq.heappop(heap)
                min_elem = heap[0]
            target_merges = -min_elem.neg_count
            next_pair = min_elem.pair
            assert min_elem is frequencies[next_pair]
            assert target_merges >= 0
            if target_merges < 2:
                break
            time_2 = time.monotonic()

            # Merge it
            # if i == 5451:
            #     print(f"{target_merges=}")
            #     freq_check = get_pairwise_frequencies_sequential(byte_sequence)
            #     print(f"{freq_check[next_pair]=}")
            #     import pdb

            #     pdb.set_trace()
            # freq_check = get_pairwise_token_frequencies(byte_sequence, max_only=True)
            # assert freq_check[next_pair] == target_merges

            byte_sequence = merge_inplace_and_update_frequencies_and_heap(
                input_sequence=byte_sequence,
                pair=next_pair,
                output_token=next_token,
                num_merges=target_merges,
                frequencies=frequencies,
                heap=heap,
            )
            time_3 = time.monotonic()

            # Update vocab
            merges[next_pair] = next_token
            self.vocab[next_token] = self.vocab[next_pair[0]] + self.vocab[next_pair[1]]
            next_token += 1
            time_4 = time.monotonic()

            if verbose:
                t_freq = time_1 - time_0
                t_max = time_2 - time_1
                t_merge = time_3 - time_2
                t_total = time_4 - time_0
                print(
                    f" {i + 1:6}/{num_merges} : "
                    f"Stats=[{t_freq=:.3f},{t_max=:.3f},{t_merge=:.3f},{t_total=:.3f}] "
                    "Merged: "
                    # f"[{self.decode([next_pair[0]])}][{self.decode([next_pair[1]])}] -> [{next_token}]. "
                    f"[{next_pair[0]}][{next_pair[1]}] -> [{next_token}]. "
                    f"freq={target_merges}. len={len(byte_sequence)}"
                )

        return merges

    @staticmethod
    def merge(input_sequence: Sequence[Token], pair: TokenPair, output_token: Token) -> Sequence[Token]:
        """Replace all occurences of `pair` in `input_sequence` with `output_token`."""
        output_sequence = []

        n = len(input_sequence)
        i = 0
        num_merges = 0

        while i < n:
            if i < n - 1 and input_sequence[i] == pair[0] and input_sequence[i + 1] == pair[1]:
                output_sequence.append(output_token)
                i += 2
                num_merges += 1
            else:
                output_sequence.append(input_sequence[i])
                i += 1

        n_out = len(output_sequence)
        assert n_out + num_merges == n

        return output_sequence

    @classmethod
    def convert_merge_list_to_vocab(cls, merges: MergeList) -> Vocab:
        """Convert a sequence of merges to a decoding vocabulary."""
        vocab: Vocab = {}

        for i in range(cls.NUM_BASE_TOKENS):
            vocab[i] = chr(i)

        for pair, token in merges.items():
            vocab[token] = vocab[pair[0]] + vocab[pair[1]]

        return vocab

    def visualize_frequencies(self, freq: Dict[TokenPair, int]) -> None:
        """."""
        freq_sorted = sorted([(count, pair) for pair, count in freq.items()], reverse=True)
        for count, pair in freq_sorted:
            # char_0 = chr(pair[0])
            # char_1 = chr(pair[1])
            char_0 = pair[0]
            char_1 = pair[1]
            print(f"[{char_0}][{char_1}]: {count}")

    def encode(self, text: str) -> NDArray[np.int32]:
        """Convert text to a series of tokens."""
        # current_sequence = list(text.encode("utf-8"))
        current_sequence = np.array(list(text), dtype=np.int32)
        while True:
            frequencies = get_pairwise_token_frequencies(current_sequence)
            pair = min(
                frequencies.keys(),
                # Among all possible merge candidates in the current sequence,
                # pick the one with the lowest rank in the merge list
                key=lambda pair: self.merge_ranks.get(pair, float("inf")),
            )
            if pair not in self.merges:
                break  # nothing left to merge
            pair_arr = np.array(pair, dtype=np.int32)
            current_sequence = merge_inplace(
                input_sequence=current_sequence,
                pair=pair_arr,
                output_token=self.merges[pair],
            )
            current_sequence = np.asarray(current_sequence, dtype=np.int32, copy=False)
        return current_sequence

    def encode2(self, text: str) -> NDArray[np.int32]:
        """Convert text to a series of tokens."""
        current_sequence = np.array(list(text), dtype=np.int32)
        frequencies = get_pairwise_token_frequencies(current_sequence, max_only=False)

        for token_pair, replacement in self.merges.items():
            target_merges = frequencies.get(token_pair, 0)
            if target_merges == 0:
                continue

            current_sequence = merge_inplace_and_update_frequencies(
                input_sequence=current_sequence,
                pair=token_pair,
                output_token=replacement,
                num_merges=target_merges,
                frequencies=frequencies,
            )
        return current_sequence

    def decode(self, tokens: Sequence[Token]) -> str:
        """Convert a series of tokens to text."""
        output = "".join(self.vocab[token] for token in tokens)
        return output


def main():
    """."""
    text = "Here's some sample text, and here's some more of it"
    with open("data/blob/t8.shakespeare.txt", mode="rb") as f:
        text = f.read()
        print(len(text))
    tokenizer = BasicTokenizer()
    merges = tokenizer.train3(text, num_merges=1200, verbose=True)
    # pprint.pprint(merges)

    trained_tokenizer = BasicTokenizer(merges=merges)
    # pprint.pprint(trained_tokenizer.vocab)
    # sample_text = "Some sample text".encode("utf-8")
    for i in range(10):
        print("Encoding...")
        start = time.monotonic()
        out = trained_tokenizer.encode2(text)
        end = time.monotonic()
        duration = end - start
        print(f"Done encoding: {len(text)} -> {len(out)}. {duration:.3f}s")
    # decoded = trained_tokenizer.decode(encoded)
    # print(decoded)

    # def is_identity(text):
    #     return trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    # print(is_identity(r"  gibberish here is more gibberish 's here's |222\$%%|"))


if __name__ == "__main__":
    main()
