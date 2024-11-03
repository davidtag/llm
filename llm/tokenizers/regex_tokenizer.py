"""Implementation of a BPE tokenizer with regex-based preprocessing."""

from __future__ import annotations

import base64
import copy
import json
import heapq
from os import PathLike
from pathlib import Path
import time

import numpy as np
import regex

from llm.tokenizers import bpe
from llm.tokenizers.cython.frequencies import get_masked_pairwise_token_frequencies_and_heap_numpy
from llm.tokenizers.cython.merge import merge_inplace_and_update_frequencies_and_heap
from llm.tokenizers.cython.pytoken import TokenDtype, MaskedTokenDtype, NumpyMaskedTokenSequence
from llm.tokenizers.cython.stdtoken import TokenPair


class RegexTokenizer:
    """Implementation of a BPE tokenizer with regex-based preprocessing."""

    def __init__(
        self,
        merge_list: bpe.MergeList,
        split_pattern: str,
    ) -> None:
        """Initialize the tokenizer."""
        self.merge_list = merge_list
        self.split_pattern = split_pattern
        self.pattern = regex.compile(split_pattern)
        self.merge_dict = bpe.convert_merge_list_to_merge_dict(self.merge_list)
        self.vocab = bpe.convert_merge_list_to_vocabulary(self.merge_list)
        self.trained_cache = bpe.convert_vocabulary_to_piece_cache(self.vocab)
        self.runtime_cache: bpe.PieceCache = {}  # TODO(dtag): Convert to LRU

    @property
    def vocab_size(self) -> int:
        """The size of the tokenizer vocabulary."""
        return len(self.vocab)

    ######################################
    # Encoding
    ######################################

    def encode(self, text: str, use_cache: bool = True) -> list[int]:
        """Encode a string into tokens."""
        tokens: list[int] = []

        for match in self.pattern.finditer(text, concurrent=False):
            # Extract the pattern-matched piece in the original text
            span = match.span()
            piece_str = text[span[0] : span[1]]

            # If already cached during training, directly get tokens
            maybe_tokens = self.trained_cache.get(piece_str, None)
            if maybe_tokens is not None:
                tokens.extend(maybe_tokens)
                continue

            # If already cached during runtime, directly get tokens
            if use_cache:
                maybe_tokens = self.runtime_cache.get(piece_str, None)
                if maybe_tokens is not None:
                    tokens.extend(maybe_tokens)
                    continue

            # Otherwise, compute it
            piece_tokens = bpe.encode_piece(piece_str, merge_dict=self.merge_dict)
            tokens.extend(piece_tokens)
            if use_cache:
                self.runtime_cache[piece_str] = piece_tokens

        return tokens

    ######################################
    # Decoding
    ######################################

    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decode a list of tokens into bytes."""
        return bpe.decode_bytes(tokens, vocab=self.vocab)

    def decode(self, tokens: list[int], errors: str = "replace") -> str:
        """Decode a list of tokens into a string."""
        return self.decode_bytes(tokens).decode("utf-8", errors=errors)

    ######################################
    # Training
    ######################################

    @classmethod
    def train(
        cls,
        text: str,
        split_pattern: str,
        num_merges: int,
        verbose: bool = True,
    ) -> RegexTokenizer:
        """Run the Byte-Pair Encoding (BPE) algorithm to generate an ordered list of merge rules."""
        train_start = time.monotonic()
        if verbose:
            print(f"-- Training {cls.__name__} --------------------------------------------")

        if num_merges < 0:
            raise ValueError("`num_merges` must be non-negative")

        if len(text) == 0:
            return RegexTokenizer(merge_list=[], split_pattern=split_pattern)

        pattern = regex.compile(split_pattern)
        merge_list: bpe.MergeList = []
        next_token = 256

        # Prepare initial data structures to speed up merge operations
        tokens_masked, masked_positions = cls._prepare_masked_token_sequence(text, pattern=pattern)
        frequencies, heap = get_masked_pairwise_token_frequencies_and_heap_numpy(
            tokens_masked=tokens_masked,
            masked_positions=masked_positions,
        )
        tokens = np.array(tokens_masked, dtype=TokenDtype)  # underflow masks (-1) by design. handled in merge
        if verbose:
            elapsed = time.monotonic() - train_start
            print(f" {0:6}/{num_merges} : {"Initializion":<22} : {elapsed=:.3f}s")

        # Perform merge operations
        for i in range(num_merges):
            iter_start = time.monotonic()
            node = heap[0]  # represents max-frequency pair
            while node.deleted:
                heapq.heappop(heap)
                node = heap[0]
            if node.count < 2:
                break  # no merge candidates left
            tokens = merge_inplace_and_update_frequencies_and_heap(
                tokens=tokens,
                token_1=node.first,
                token_2=node.second,
                output_token=next_token,
                expected_num_merges=node.count,
                frequencies=frequencies,
                heap=heap,
            )
            if verbose:
                iter_end = time.monotonic()
                iter_duration_ms = 1000 * (iter_end - iter_start)
                elapsed = iter_end - train_start
                print(
                    f" {i + 1:6}/{num_merges} : [{next_token}] <- "
                    f"[{node.first:>5}][{node.second:>5}]"
                    f": iter={iter_duration_ms:>4.1f}ms, {elapsed=:.3f}s, freq={node.count}"
                )
            merge_list.append((node.pair, next_token))
            next_token += 1

        return RegexTokenizer(merge_list=merge_list, split_pattern=split_pattern)

    def train_piece_cache(
        self,
        text: str,
        num_extra_pieces: int = 10,
        verbose: bool = True,
    ) -> RegexTokenizer:
        """Augment the trained_cache of a tokenizer using some additional training text.

        This essentially warms up the runtime_cache based on some training data. The training data can be
        the same or different from the data used to train the BPE merges.
        """
        if num_extra_pieces < 0:
            raise ValueError("`num_extra_pieces` must be non-negative")

        train_start = time.monotonic()
        if verbose:
            print(f"-- Training {self.__class__.__name__} : piece cache -------------------------------")

        piece_counter: dict[str, bpe.PieceCacheCounterValue] = {}

        for match in self.pattern.finditer(text, concurrent=False):
            # Extract the pattern-matched piece in the original text
            span = match.span()
            piece_str = text[span[0] : span[1]]

            # If we've seen this while training the merges, there's nothing to do, it
            # will always be in the output
            if piece_str in self.trained_cache:
                continue

            # If we have already computed and cached this piece, augment it's count
            if piece_str in piece_counter:
                piece_counter[piece_str].count += 1
                continue

            # Otherwise, compute the correct encoding and cache it
            piece_tokens = bpe.encode_piece(piece_str, merge_dict=self.merge_dict)
            piece_counter[piece_str] = bpe.PieceCacheCounterValue(tokens=piece_tokens, count=1)

        top_pieces = sorted(
            piece_counter.keys(),
            key=lambda piece, counter=piece_counter: counter[piece].count,  # type: ignore[misc]
            reverse=True,
        )[:num_extra_pieces]
        top_pieces_cache = {piece: piece_counter[piece].tokens for piece in top_pieces}
        effective_num_extra_pieces = len(top_pieces_cache)

        new_trained_cached = copy.deepcopy(self.trained_cache)
        new_trained_cached.update(top_pieces_cache)
        assert len(new_trained_cached) == len(self.trained_cache) + len(top_pieces_cache)  # no duplicates

        new_tokenizer = RegexTokenizer(merge_list=self.merge_list, split_pattern=self.split_pattern)
        new_tokenizer.trained_cache = new_trained_cached

        if verbose:
            training_end = time.monotonic()
            elapsed = training_end - train_start
            print(f"  DONE: {effective_num_extra_pieces=}, {elapsed=:.3f}s")

        return new_tokenizer

    ######################################
    # Persistence to disk
    ######################################

    def save(self, directory: PathLike) -> None:
        """Store all tokenizer artifacts to disk."""
        base_dir = Path(directory)
        base_dir.mkdir(parents=False, exist_ok=True)

        self._save_pattern(file=Path(base_dir, "pattern.txt"))
        self._save_merges(file=Path(base_dir, "merges.txt"))
        self._save_vocab(file=Path(base_dir, "vocab.txt"))
        self._save_cache(file=Path(base_dir, "cache.txt"))

    @classmethod
    def load(cls, directory: PathLike) -> RegexTokenizer:
        """Instantiate a tokenizer from saved artifacts."""
        base_dir = Path(directory)
        if not base_dir.exists() or not base_dir.is_dir():
            raise ValueError("Model directory not found")

        split_pattern = cls._load_pattern(file=Path(base_dir, "pattern.txt"))
        merge_list = cls._load_merges(file=Path(base_dir, "merges.txt"))
        piece_cache = cls._load_cache(file=Path(base_dir, "cache.txt"))
        tokenizer = RegexTokenizer(merge_list=merge_list, split_pattern=split_pattern)
        tokenizer.trained_cache.update(piece_cache)
        return tokenizer

    ######################################
    # Private Methods: for persistence
    ######################################

    def _save_pattern(self, file: PathLike) -> None:
        with open(file, mode="w", encoding="utf-8") as f:
            f.write(self.split_pattern)
            f.write("\n")  # for shell readability with cat

    @staticmethod
    def _load_pattern(file: PathLike) -> str:
        with open(file, mode="r", encoding="utf-8") as f:
            split_pattern = f.read()[:-1]
        return split_pattern

    def _save_merges(self, file: PathLike) -> None:
        with open(file, mode="w", encoding="utf-8") as f:
            for pair, replacement in self.merge_list:
                f.write(f"{pair.first},{pair.second}:{replacement}\n")

    @staticmethod
    def _load_merges(file: PathLike) -> bpe.MergeList:
        merge_list = bpe.MergeList()
        with open(file, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                pair_str, replacement_str = line.strip().split(":")
                first, second = pair_str.split(",")
                token_pair = TokenPair(first=int(first), second=int(second))
                replacement = int(replacement_str)
                merge_list.append((token_pair, replacement))
        return merge_list

    def _save_vocab(self, file: PathLike) -> None:
        with open(file, mode="w", encoding="utf-8") as f:
            for token, token_bytes in enumerate(self.vocab):
                f.write(f"[{token}] : [{bpe.render_bytes(token_bytes)}]\n")

    def _save_cache(self, file: PathLike) -> None:
        with open(file, mode="w", encoding="utf-8") as f:
            for piece_str, piece_tokens in self.trained_cache.items():
                if len(piece_tokens) > 1:  # len==1 is implicilty captured by the vocabulary
                    f.write(f"{base64.b64encode(piece_str.encode("utf-8")).decode("utf-8")}:{piece_tokens}\n")

    @staticmethod
    def _load_cache(file: PathLike) -> bpe.PieceCache:
        piece_cache = bpe.PieceCache()
        with open(file, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                b64_str, piece_tokens = line.strip().split(":")
                piece_str = base64.b64decode(b64_str.encode("utf-8")).decode("utf-8")
                piece_cache[piece_str] = json.loads(piece_tokens)
        return piece_cache

    ######################################
    # Private Methods: for training
    ######################################

    @staticmethod
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
