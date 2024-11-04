"""Generate text from a trained Transformer model."""

import argparse
from pathlib import Path
import sys

import numpy as np

from llm.data.registry import ModelRegistry, TokenizerRegistry
from llm.models.transformer import Transformer
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


def _load_model_for_eval(
    checkpoint_name: str,
    checkpoint_iter: str,
) -> Transformer:
    model_registry = ModelRegistry()
    checkpoint_path = Path(
        model_registry.checkpoint_dir,
        checkpoint_name,
        f"checkpoint.{checkpoint_iter}.pkl",
    )

    model = Transformer.load_for_eval(model_file=checkpoint_path)
    print(f"Loaded model {checkpoint_name}:{checkpoint_iter} with n_params={model.n_params:,}")
    return model


def main(args: argparse.Namespace) -> None:
    """Entrypoint."""
    tokenizer = _load_tokenizer(args.tokenizer_name)
    model = _load_model_for_eval(
        checkpoint_name=args.checkpoint_name,
        checkpoint_iter=args.checkpoint_iter,
    )

    print("Generating text...")
    print("=================================================================================")
    start_str = "ACT I. Scene I.\n\n"
    start_tokens = tokenizer.encode(start_str)
    start_sequence = np.array(start_tokens)

    print(start_str)
    for token in model.generate_stream(start_sequence, max_tokens=500):
        text = tokenizer.decode([token])
        sys.stdout.write(text)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    print("=================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer model.")
    parser.add_argument(
        "-t",
        "--tokenizer_name",
        type=str,
        required=True,
        default="",
        help="The name of the saved tokenizer checkpoint",
    )
    parser.add_argument(
        "-n",
        "--checkpoint_name",
        type=str,
        required=True,
        default="",
        help="The name of a previously saved model checkpoint",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_iter",
        type=int,
        required=True,
        default=0,
        help="The iteration number of the previously saved model checkpoint",
    )
    args = parser.parse_args()

    main(args)
