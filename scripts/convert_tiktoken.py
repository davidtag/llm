"""Convert popular tiktoken (i.e., OpenAI) encoders to this repo's format."""

from pathlib import Path

import tiktoken

from llm.data.loaders import load_text_file
from llm.data.registry import TokenizerRegistry, TextDataRegistry
from llm.tokenizers import bpe
from llm.tokenizers import RegexTokenizer
from llm.tokenizers.bpe import convert_reverse_vocabulary_to_vocabulary, MergeList, MergeDict, Vocabulary
from llm.tokenizers.cython import bpe as _bpe_internal
from llm.tokenizers.cython.stdtoken import TokenPair
from llm.utils.profile import Profile


ENCODINGS_TO_CONVERT = [
    "gpt2",
    "r50k_base",
    "cl100k_base",
    "o200k_base",
]


def _normalize_token(token: int, vocab: Vocabulary) -> int:
    if token < 256:
        token_bytes = vocab[token]
        return ord(token_bytes)
    return token


def _convert_mergeable_ranks_to_merge_list(mergeable_ranks: dict[bytes, int]) -> MergeList:
    merge_list: MergeList = []
    merge_dict: MergeDict = {}

    vocab = convert_reverse_vocabulary_to_vocabulary(mergeable_ranks)

    for token, token_bytes in enumerate(vocab):
        # The 0...255 tokens are a permutation of the base bytes
        # These tokens are not the result of a merge rule
        if token < 256:
            assert len(token_bytes) == 1
            continue
        assert token >= 256

        # For the other tokens, token_bytes is a concatenation of the representation of
        # 2 preceding tokens (i.e., those with values < token)
        valid_pairs = []
        for split_point in range(1, len(token_bytes)):
            part_1 = token_bytes[:split_point]
            part_2 = token_bytes[split_point:]
            maybe_token_1 = mergeable_ranks.get(part_1, None)
            maybe_token_2 = mergeable_ranks.get(part_2, None)
            if (
                maybe_token_1 is None
                or maybe_token_2 is None
                or maybe_token_1 >= token
                or maybe_token_2 >= token
            ):
                continue
            token_1 = _normalize_token(maybe_token_1, vocab=vocab)
            token_2 = _normalize_token(maybe_token_2, vocab=vocab)
            valid_pairs.append(TokenPair(first=token_1, second=token_2))

        if len(valid_pairs) == 0:
            raise RuntimeError(f"Unable to convert {token=}: {token_bytes!r}")

        elif len(valid_pairs) == 1:
            pair = valid_pairs[0]
            merge_list.append((pair, token))
            merge_dict[pair] = token

        else:
            # Among possible merges, pick the one that preserves the encoding
            matched = False
            for pair in valid_pairs:
                merge_list.append((pair, token))
                merge_dict[pair] = token

                out_tokens = _bpe_internal.encode_tokens(list(token_bytes), merge_dict=merge_dict)
                if out_tokens == [token]:
                    matched = True
                    break
                else:
                    merge_list.pop()
                    del merge_dict[pair]
            if not matched:
                raise RuntimeError(f"Unable to convert {token=}: {token_bytes!r}")

    return merge_list


def _convert_tiktoken_encoder_to_regex_tokenizer(encoder: tiktoken.Encoding) -> RegexTokenizer:
    merge_list = _convert_mergeable_ranks_to_merge_list(encoder._mergeable_ranks)
    split_pattern = encoder._pat_str
    return RegexTokenizer(merge_list=merge_list, split_pattern=split_pattern)


def _convert(name: str) -> None:
    print(f"-- Converting encoding={name}")
    encoder = tiktoken.get_encoding(name)
    tokenizer = _convert_tiktoken_encoder_to_regex_tokenizer(encoder)

    # Initialize the checkpoint directory
    tokenizer_registry = TokenizerRegistry()
    checkpoint_dir = Path(tokenizer_registry.checkpoint_dir, name)
    checkpoint_dir.mkdir(exist_ok=True)

    # Save the tokenizer
    tokenizer.save(checkpoint_dir)

    # Save the reference vocabularty for manual inspection and comparison
    vocab = convert_reverse_vocabulary_to_vocabulary(encoder._mergeable_ranks)
    with open(Path(checkpoint_dir, "vocab.ref.txt"), mode="w", encoding="utf-8") as f:
        for token, token_bytes in enumerate(vocab):
            f.write(f"[{token}] : [{bpe.render_bytes(token_bytes)}]\n")


def _load_tokenizer(name: str) -> RegexTokenizer:
    tokenizer_registry = TokenizerRegistry()
    checkpoint_dir = Path(tokenizer_registry.checkpoint_dir, name)
    if not checkpoint_dir.exists():
        raise RuntimeError(f"Checkpoint dir {checkpoint_dir} doesn't exist. Did you train the tokenizer?")
    tokenizer = RegexTokenizer.load(checkpoint_dir)
    return tokenizer


def _test(name: str) -> None:
    print(f"-- Testing encoding={name}")
    encoder = tiktoken.get_encoding(name)
    tokenizer = _load_tokenizer(name)

    text = "Hello World! my name is AMAZING  123 LOL (안녕하세요!) joined123 😉"
    ref_tokens = encoder.encode(text)
    these_tokens = tokenizer.encode(text)
    assert len(ref_tokens) == len(these_tokens)  # tokens are the same up to a permutation of base tokens
    assert encoder.decode(ref_tokens) == text
    assert tokenizer.decode(these_tokens) == text


def _profile(name: str) -> None:
    print(f"-- Profiling encoding={name}")
    encoder = tiktoken.get_encoding(name)
    tokenizer = _load_tokenizer(name)

    registry = TextDataRegistry()
    text = load_text_file(registry.raw_text_file)
    size = len(text) / 1024 / 1024
    print(f"Loaded text of {size:.1f} MB")

    with Profile() as prof:
        ref_tokens = encoder.encode(text)
    print(f"tiktoken.Encoding: {prof.milliseconds:>5,.0f} ms ({size / prof.seconds:.1f} MB/s)")

    with Profile() as prof:
        these_tokens = tokenizer.encode(text, use_cache=False)
    print(f"RegexTokenizer   : {prof.milliseconds:>5,.0f} ms ({size / prof.seconds:.1f} MB/s)")

    assert len(ref_tokens) == len(these_tokens)  # tokens are the same up to a permutation of base tokens


def main() -> None:
    """Entrypoint."""
    for name in ENCODINGS_TO_CONVERT:
        _convert(name)
        _test(name)
        _profile(name)


if __name__ == "__main__":
    main()
