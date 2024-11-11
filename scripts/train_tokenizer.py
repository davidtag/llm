"""Train the tokenizer."""

import argparse
from pathlib import Path
import time

from llm.data.loaders import load_text_file
from llm.data.registry import TextDataRegistry, TokenizerRegistry
from llm.tokenizers import RegexTokenizer
from llm.tokenizers import SplitPattern


def _validate(tokenizer: RegexTokenizer) -> None:
    print("-- Validating ----------------------------------------------------")
    sample_unicode = "hello world!!!? (안녕하세요!) joined123 😉"
    print(f"{sample_unicode=}")
    tokens = tokenizer.encode(sample_unicode)
    print(f"{tokens=}")
    recreated_input = tokenizer.decode(tokens)
    print(f"{recreated_input=}")
    assert sample_unicode == recreated_input


def main(args: argparse.Namespace) -> None:
    """Entrypoint."""
    # Initialize the checkpoint directory
    tokenizer_registry = TokenizerRegistry()
    tokenizer_registry.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if len(args.name) > 0:
        checkpoint_dir = Path(tokenizer_registry.checkpoint_dir, args.name)
        if checkpoint_dir.exists() and not args.overwrite:
            raise RuntimeError(
                f"Checkpoint dir {checkpoint_dir} already exists. Change name or provide -y flag."
            )
    else:
        checkpoint_dir = Path(tokenizer_registry.checkpoint_dir, f"{int(time.time())}")
    checkpoint_dir.mkdir(exist_ok=True)

    # Load the training data
    data_registry = TextDataRegistry()
    train_text = load_text_file(data_registry.train_text_file)

    # Train the tokenizer
    tokenizer = RegexTokenizer.train(
        text=train_text,
        split_pattern=SplitPattern.get_pattern(args.split_pattern),
        num_merges=args.num_merges,
        verbose=args.verbose,
    )

    # Augment piece cache
    new_tokenizer = tokenizer.train_piece_cache(
        text=train_text,
        num_extra_pieces=args.cache_size,
        verbose=args.verbose,
    )

    # Validate encode-decode consistency
    _validate(new_tokenizer)

    # Save the checkpoint
    new_tokenizer.save(checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE text encoder with regex split pattern.")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=False,
        default="",
        help="A name for the emitted checkpoint",
    )
    parser.add_argument(
        "-y",
        "--overwrite",
        action="store_true",
        help="Overwrite an existing checkpoint if it already exists",
    )
    parser.add_argument(
        "-s",
        "--split_pattern",
        type=str,
        default=SplitPattern.default_pattern_name(),
        required=False,
        choices=SplitPattern.all_pattern_names(),
        help="The regex pattern used to split the text",
    )
    parser.add_argument(
        "-m",
        "--num_merges",
        type=int,
        default=10_000,
        required=False,
        help="The number of merges to perform",
    )
    parser.add_argument(
        "-c",
        "--cache_size",
        type=int,
        default=10_000,
        required=False,
        help="The number of extra text pieces to cache beyond the base merges",
    )
    parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Disable verbose output",
    )
    args = parser.parse_args()

    main(args)
