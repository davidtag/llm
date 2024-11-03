"""Implementation of a basic BPE tokenizer."""

import heapq
import pprint
import time

import numpy as np
import tiktoken

from llm.tokenizers.benchmarks.profile import Profile
from llm.tokenizers.cython.frequencies import (
    get_pairwise_token_frequencies_numpy,
    get_pairwise_token_frequencies_and_heap_numpy,
)
from llm.tokenizers.cython.merge import (
    merge_inplace_and_update_frequencies,
    merge_inplace_and_update_frequencies_and_heap,
)
from llm.tokenizers.cython.stdtoken import TokenPair
from llm.tokenizers.cython.pytoken import TokenDtype, NumpyTokenSequence


TRAIN_FILE = "data/blob/t8.shakespeare.txt"


_BYTE = 1
_KB = 1024 * _BYTE
_MB = 1024 * _KB


MergeList = list[tuple[TokenPair, int]]
Vocabulary = list[bytes]


def _load_file_bytes(file_path: str = TRAIN_FILE) -> bytes:
    with open(file_path, mode="rb") as f:
        data = f.read()
    print(f"Read {len(data):,} bytes")
    return data


def _split_train_and_test(data: bytes, val_length: int = 1 * _MB) -> tuple[bytes, bytes]:
    assert len(data) > val_length
    train = data[:-val_length]
    val = data[-val_length:]
    print(f"Split data into train ({len(train):,} bytes) and val ({len(val):,} bytes)")
    return train, val


def _convert_bytes_to_token_sequence(data: bytes) -> NumpyTokenSequence:
    out = np.array(list(data), dtype=TokenDtype)
    assert out.shape == (len(data),)
    return out


def _train(text: bytes, num_merges: int, verbose: bool = True) -> MergeList:
    train_start = time.monotonic()

    merges: MergeList = []
    next_token = 256

    if len(text) == 0:
        return merges

    tokens = _convert_bytes_to_token_sequence(text)
    frequencies, heap = get_pairwise_token_frequencies_and_heap_numpy(tokens)

    for i in range(num_merges):
        iter_start = time.monotonic()
        node = heap[0]
        while node.deleted:
            heapq.heappop(heap)
            node = heap[0]
        if node.count < 2:
            break
        tokens = merge_inplace_and_update_frequencies_and_heap(
            tokens=tokens,
            token_1=node.first,
            token_2=node.second,
            output_token=next_token,
            expected_num_merges=node.count,
            frequencies=frequencies,
            heap=heap,
        )
        merges.append((node.pair, next_token))
        next_token += 1
        if verbose:
            iter_end = time.monotonic()
            iter_duration_ms = 1000 * (iter_end - iter_start)
            elapsed = iter_end - train_start
            print(
                f" {i + 1:6}/{num_merges} : [{next_token}] <- "
                f"[{node.first:>5}][{node.second:>5}]"
                f": iter={iter_duration_ms:>4.1f}ms, {elapsed=:.3f}s"
            )

    return merges


def _convert_merge_list_to_vocab(merges: MergeList) -> Vocabulary:
    vocab_size = 256 + len(merges)
    vocab: Vocabulary = [b"" for _ in range(vocab_size)]

    for i in range(256):  # base vocab : 1-byte code points
        vocab[i] = chr(i).encode("utf-8")

    for pair, output_token in merges:  # merges
        vocab[output_token] = vocab[pair.first] + vocab[pair.second]

    return vocab


def _encode(data: bytes, merges: MergeList) -> NumpyTokenSequence:
    tokens = _convert_bytes_to_token_sequence(data)
    frequencies = get_pairwise_token_frequencies_numpy(tokens)

    for pair, output_token in merges:
        pair_freq = frequencies.get(pair, 0)
        if pair_freq == 0:
            continue
        assert pair_freq > 0

        tokens = merge_inplace_and_update_frequencies(
            tokens=tokens,
            token_1=pair.first,
            token_2=pair.second,
            output_token=output_token,
            expected_num_merges=pair_freq,
            frequencies=frequencies,
        )

    return tokens


def _decode(tokens: NumpyTokenSequence, vocab: Vocabulary) -> bytes:
    # TODO(dtag): Speed this up in Cython by pre-allocating the correct byte size and
    # using C loops instead of Python loops.
    return b"".join([vocab[token] for token in tokens])


def main() -> None:
    """Entrypoint."""
    print()
    print("-- Basic Tokenizer Example ----------------------------------------------------------------------")
    data = _load_file_bytes(TRAIN_FILE)
    train_data, val_data = _split_train_and_test(data, val_length=2 * _MB)

    print("Training....")
    merges = _train(train_data, num_merges=1300)
    vocab = _convert_merge_list_to_vocab(merges)
    pprint.pprint(vocab)

    print("Encoding...")
    with Profile() as prof:
        tokens = _encode(val_data, merges).tolist()
    print(f"  val_data={len(val_data):,} -> tokens={len(tokens):,}: elapsed={prof.milliseconds_formatted}")

    print("Deconding...")
    with Profile() as prof:
        val_out = _decode(tokens, vocab).decode()
    print(f"  tokens={len(tokens):,} -> val_out={len(val_out):,}: elapsed={prof.milliseconds_formatted}")
    assert len(val_out) == len(val_data)
    assert val_out == val_data

    print("Encodig with tiktoken...")
    enc = tiktoken.get_encoding("gpt2")
    val_text = val_data.decode("utf-8")
    with Profile() as prof:
        tokens = enc.encode(val_text)
    print(f"  val_text={len(val_text):,} -> tokens={len(tokens):,}: elapsed={prof.milliseconds_formatted}")

    print("Decoding with tiktoken...")
    with Profile() as prof:
        val_out = enc.decode(tokens)
    print(f"  tokens={len(tokens):,} -> val_out={len(val_out):,}: elapsed={prof.milliseconds_formatted}")
    assert len(val_out) == len(val_text)
    assert val_out == val_text


if __name__ == "__main__":
    main()
