"""Convert popular tiktoken (i.e., OpenAI) encoders to this repo's format."""

from pathlib import Path

import tiktoken

from llm.data.registry import TokenizerRegistry
from llm.tokenizers import bpe
from llm.tokenizers import RegexTokenizer
from llm.tokenizers.bpe import convert_reverse_vocabulary_to_vocabulary, MergeList, MergeDict, Vocabulary
from llm.tokenizers.cython import bpe as _bpe_internal
from llm.tokenizers.cython.stdtoken import TokenPair


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
    encoder = tiktoken.get_encoding(name)
    tokenizer = _convert_tiktoken_encoder_to_regex_tokenizer(encoder)

    # Initialize the checkpoint directory
    tokenizer_registry = TokenizerRegistry()
    checkpoint_dir = Path(tokenizer_registry.checkpoint_dir, name)
    checkpoint_dir.mkdir(exist_ok=True)

    tokenizer.save(checkpoint_dir)

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
    print(
        f"Loaded tokenizer '{name}' with "
        f"vocab_size={tokenizer.vocab_size:,} and cache_size={len(tokenizer.trained_cache):,}"
    )
    return tokenizer


def _test(name: str) -> None:
    encoder = tiktoken.get_encoding(name)
    tokenizer = _load_tokenizer(name)

    text = "Hello World! my name is AMAZING  123 LOL (안녕하세요!) joined123 😉"
    print(encoder.encode(text))
    print(tokenizer.encode(text))


def main() -> None:
    """Entrypoint."""
    _convert("cl100k_base")
    _test("cl100k_base")


if __name__ == "__main__":
    main()
