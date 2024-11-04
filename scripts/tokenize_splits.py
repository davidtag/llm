"""Tokenize and save the train/val/test text splits."""

import argparse
from pathlib import Path

from llm.data.loaders import load_text_file, write_token_file
from llm.data.registry import DataRegistry, TokenRegistry, TokenizerRegistry
from llm.tokenizers import RegexTokenizer


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


def _tokenize_and_save(tokenizer: RegexTokenizer, src: Path, dst: Path) -> None:
    text = load_text_file(src)
    tokens = tokenizer.encode(text)
    n_text = len(text)
    n_tokens = len(tokens)
    compression = 1 - n_tokens / n_text
    print(
        f"Encoded {n_text:,} characters to {n_tokens:,} tokens. "
        f"compression={100 * compression:.1f}%. "
        f"Saving to: {dst}"
    )
    write_token_file(dst, tokens)


def main(args: argparse.Namespace) -> None:
    """Entrypoint."""
    tokenizer = _load_tokenizer(args.name)

    data_registry = DataRegistry()

    token_registry = TokenRegistry(tokenizer_name=args.name)
    token_registry.token_dir.mkdir(parents=True, exist_ok=True)

    _tokenize_and_save(tokenizer, src=data_registry.train_text_file, dst=token_registry.train_token_file)
    _tokenize_and_save(tokenizer, src=data_registry.val_text_file, dst=token_registry.val_token_file)
    _tokenize_and_save(tokenizer, src=data_registry.test_text_file, dst=token_registry.test_token_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize and save the train/val/test text splits.")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        default="",
        help="The name of the saved tokenizer checkpoint",
    )
    args = parser.parse_args()

    main(args)
