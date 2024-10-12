"""Implementation of a basic BPE tokenizer."""

import heapq
import time

import regex
import numpy as np
import tiktoken

from llm.tokenizers.benchmarks.profile import Profile
from llm.tokenizers.frequencies import (
    get_pairwise_token_frequencies_sequential_pure_python,
)
from llm.tokenizers.merge import (
    merge_inplace_and_update_frequencies_and_heap,
)
from llm.tokenizers.stdtoken import TokenPair, TokenPairNode
from llm.tokenizers.pytoken import TokenDtype, NumpyTokenSequence, MaskedTokenDtype, NumpyMaskedTokenSequence


TRAIN_FILE = "data/blob/t8.shakespeare.txt"


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


_BYTE = 1
_KB = 1024 * _BYTE
_MB = 1024 * _KB


MergeList = list[tuple[TokenPair, int]]
MergeDict = dict[tuple[int, int], int]  # TODO(dtag): Use TokenPair as key
Vocabulary = list[bytes]
ReverseVocabulary = dict[bytes, int]


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


TOKEN_VALUE_UBOUND = 1_000_000


def get_masked_pairwise_token_frequencies_and_heap_numpy(
    tokens_masked: NumpyMaskedTokenSequence,
    masked_positions: NumpyMaskedTokenSequence,
) -> tuple[
    dict[TokenPair, TokenPairNode],
    list[TokenPairNode],
]:
    """Compute the token frequencies using numpy broadcast, and arrange them in a heap."""
    freq: dict[TokenPair, TokenPairNode] = {}
    heap: list[TokenPairNode] = []

    if len(tokens_masked) == 0:
        return freq, heap

    # Determine all unique pairs using bit packing
    buffer = np.zeros(len(tokens_masked) - 1, dtype=np.int64)
    np.add(buffer, tokens_masked[:-1], out=buffer)
    np.multiply(buffer, TOKEN_VALUE_UBOUND, out=buffer)
    np.add(buffer, tokens_masked[1:], out=buffer)
    buffer[masked_positions] = -1
    buffer[masked_positions - 1] = -1
    unique_values, counts = np.unique(buffer, return_counts=True)

    # Efficiently unpack them
    first_tokens = np.floor_divide(unique_values, TOKEN_VALUE_UBOUND)
    second_tokens = np.mod(unique_values, TOKEN_VALUE_UBOUND)

    # Package the frequencies
    for token_1, token_2, count in zip(first_tokens, second_tokens, counts, strict=True):
        assert token_2 > 0
        if token_1 < 0:
            continue
        pair = TokenPair(token_1, token_2)
        node = TokenPairNode(token_1, token_2, count=count)
        heap.append(node)
        freq[pair] = node

    heapq.heapify(heap)

    return freq, heap


def _prepare_masked_token_sequence(
    text: str,
    pattern: regex.Pattern,
) -> tuple[
    NumpyMaskedTokenSequence,
    NumpyMaskedTokenSequence,
]:
    tokens_masked: list[int] = []
    masked_positions: list[int] = []

    for match in pattern.finditer(text, concurrent=False):
        # Extract the pattern-matched piece in the original text
        span = match.span()
        piece_str = text[span[0] : span[1]]

        # Get the byte-level tokens for this piece
        piece_bytes = piece_str.encode("utf-8")
        piece_tokens = list(piece_bytes)
        tokens_masked.extend(piece_tokens)

        # Append a mask at the end of each piece
        masked_positions.append(len(tokens_masked))
        tokens_masked.append(-1)

    # Package into array and remove terminal mask
    tokens_masked_npy = np.array(tokens_masked[:-1], dtype=MaskedTokenDtype)
    masked_positions_npy = np.array(masked_positions[:-1], dtype=MaskedTokenDtype)

    return tokens_masked_npy, masked_positions_npy


def _train(text: str, pattern: regex.Pattern, num_merges: int, verbose: bool = True) -> MergeList:
    train_start = time.monotonic()

    merges: MergeList = []
    next_token = 256

    if len(text) == 0:
        return merges

    tokens_masked, mask_positions = _prepare_masked_token_sequence(text, pattern=pattern)
    frequencies, heap = get_masked_pairwise_token_frequencies_and_heap_numpy(tokens_masked, mask_positions)
    tokens = np.array(tokens_masked, dtype=TokenDtype)  # TODO(dtag): Handle underflow

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
                f": iter={iter_duration_ms:>4.1f}ms, {elapsed=:.3f}s, freq={node.count}"
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


def _convert_vocabulary_to_reverse_vocabulary(vocab: Vocabulary) -> ReverseVocabulary:
    reverse_vocab = {token_bytes: token for token, token_bytes in enumerate(vocab)}
    return reverse_vocab


def _convert_merges_list_to_merges_dict(merges: MergeList) -> MergeDict:
    merges_dict = {
        (token_pair.first, token_pair.second): replacement_token for token_pair, replacement_token in merges
    }
    return merges_dict


def _merge(
    tokens: list[int],
    pair: tuple[int, int],
    replacement: int,
) -> list[int]:
    output_tokens = []
    n = len(tokens)
    i = 0

    while i < n:
        if i < n - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            output_tokens.append(replacement)
            i += 2
        else:
            output_tokens.append(tokens[i])
            i += 1

    return output_tokens


def _bpe(tokens: list[int], merges_dict: MergeDict) -> list[int]:
    while len(tokens) > 2:
        freq = get_pairwise_token_frequencies_sequential_pure_python(tokens)
        merge_pair = min(freq.keys(), key=lambda p: merges_dict.get(p, float("inf")))
        if merge_pair not in merges_dict:
            break
        replacement = merges_dict[merge_pair]
        tokens = _merge(tokens, merge_pair, replacement)
    return tokens


def _encode(
    text: str, pattern: regex.Pattern, merges_dict: MergeDict, reverse_vocab: ReverseVocabulary
) -> list[int]:
    tokens: list[int] = []
    bpe_cache: dict[bytes, list[int]] = {}  # for a serving system, can be an LRU cache

    for match in pattern.finditer(text, concurrent=False):
        # Extract the pattern-matched piece in the original text
        span = match.span()
        piece_str = text[span[0] : span[1]]

        # Get the byte-level tokens for this piece
        piece_bytes = piece_str.encode(
            "utf-8"
        )  # TODO(dtag): We can skip this if we cache the str -> tokens instead
        maybe_token = reverse_vocab.get(piece_bytes, None)  # TODO(dtag): SortedDict might be faster
        if maybe_token is not None:
            tokens.append(maybe_token)
        else:
            # Note: caching reduces runtime from 1800ms to 750ms on our val set
            maybe_cached_tokens = bpe_cache.get(piece_bytes, None)
            if maybe_cached_tokens is not None:
                tokens.extend(maybe_cached_tokens)
            else:
                piece_tokens = list(piece_bytes)
                merged_piece_tokens = _bpe(piece_tokens, merges_dict=merges_dict)
                tokens.extend(merged_piece_tokens)
                bpe_cache[piece_bytes] = merged_piece_tokens

    return tokens


def _decode_bytes(tokens: list[int], vocab: Vocabulary) -> bytes:
    # TODO(dtag): Speed this up in Cython by pre-allocating the correct byte size and
    # using C loops instead of Python loops.
    return b"".join([vocab[token] for token in tokens])


def _decode(tokens: list[int], vocab: Vocabulary, errors: str = "replace") -> str:
    return _decode_bytes(tokens, vocab=vocab).decode("utf-8", errors=errors)


def main():
    """Entrypoint."""
    print()
    print("-- Regex Tokenizer Example ----------------------------------------------------------------------")
    data = _load_file_bytes(TRAIN_FILE)
    train_data, val_data = _split_train_and_test(data, val_length=2 * _MB)

    # train_text = "This is my fancy string, where is your's, man? ✐✐. Hi there."
    train_text = train_data.decode("utf-8")
    val_text = val_data.decode("utf-8")

    pattern = regex.compile(GPT4_SPLIT_PATTERN)

    print("Training....")
    merges = _train(train_text, pattern=pattern, num_merges=1000)
    vocab = _convert_merge_list_to_vocab(merges)
    reverse_vocab = _convert_vocabulary_to_reverse_vocabulary(vocab)
    merges_dict = _convert_merges_list_to_merges_dict(merges)
    # pprint.pprint(vocab)

    # TODO(dtag): Run encode on the train set and in addition to the merges, store the top #merges most
    # common byte patterns and their decoded token sequence

    print("Encoding...")
    with Profile() as prof:
        tokens = _encode(val_text, pattern=pattern, merges_dict=merges_dict, reverse_vocab=reverse_vocab)
    print(f"  val_text={len(val_text):,} -> tokens={len(tokens):,}: elapsed={prof.milliseconds_formatted}")

    print("Deconding...")
    with Profile() as prof:
        val_out = _decode(tokens, vocab=vocab)
    print(f"  tokens={len(tokens):,} -> val_out={len(val_out):,}: elapsed={prof.milliseconds_formatted}")
    assert len(val_out) == len(val_text)
    assert val_out == val_text

    print("Encodig with tiktoken...")
    enc = tiktoken.get_encoding("cl100k_base")
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
