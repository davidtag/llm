"""Train a Transformer model."""

import argparse
from pathlib import Path
import time

import numpy as np

from llm.data.loaders import load_token_file
from llm.data.registry import ModelRegistry, TokenRegistry, TokenizerRegistry
from llm.loss.cross_entropy import CrossEntropyLoss
from llm.models.transformer import Transformer
from llm.optimizers.adam import Adam
from llm.tokenizers import RegexTokenizer


def _initialize_checkpoint_dir(args: argparse.Namespace) -> Path:
    model_registry = ModelRegistry()
    model_registry.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if len(args.name) > 0:
        if args.name == args.starting_checkpoint_name:
            raise RuntimeError("The starting checkpoint and final checkpoint names must be different")
        checkpoint_dir = Path(model_registry.checkpoint_dir, args.name)
        if checkpoint_dir.exists() and not args.overwrite:
            raise RuntimeError(
                f"Checkpoint dir {checkpoint_dir} already exists. Change name or provide -y flag."
            )
    else:
        checkpoint_dir = Path(model_registry.checkpoint_dir, f"{int(time.time())}")
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


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


def _load_tokens(name: str) -> np.ndarray:
    token_registry = TokenRegistry(tokenizer_name=name)
    train_tokens = load_token_file(token_registry.train_token_file)
    print(f"Loaded {len(train_tokens):,} train tokens")
    return np.array(train_tokens)


def _initialize_model_for_training(
    vocab_size: int,
    lr: float,
) -> Transformer:
    optimizer = Adam(lr=lr)
    model = Transformer(
        vocab_size=vocab_size,
        context_size=128,
        n_blocks=6,
        d_model=64,
        d_k=8,
        d_v=8,
        h=8,
        d_ff=256,
        optimizer=optimizer,
    )
    print(f"Initialized model with n_params={model.n_params:,}")
    return model


def _load_model_for_training(
    starting_checkpoint_name: str,
    starting_checkpoint_iter: str,
    lr: float,
) -> Transformer:
    optimizer = Adam(lr=lr)

    model_registry = ModelRegistry()
    checkpoint_path = Path(
        model_registry.checkpoint_dir,
        starting_checkpoint_name,
        f"checkpoint.{starting_checkpoint_iter}.pkl",
    )

    model = Transformer.load_for_training(model_file=checkpoint_path, optimizer=optimizer)
    print(
        f"Loaded model checkpoint {starting_checkpoint_name}:{starting_checkpoint_iter} "
        f"with n_params={model.n_params:,}"
    )
    return model


def _train_model(
    model: Transformer,
    training_sequence: np.ndarray,
    checkpoint_dir: Path,
    batch_size: int = 32,
    num_batches: int = 20,
    checkpoint_freq: int = 1_000,
) -> Transformer:
    loss_fn = CrossEntropyLoss()

    def get_batch() -> tuple[np.ndarray, np.ndarray]:
        idxs = np.random.randint(low=0, high=len(training_sequence) - model.context_size, size=batch_size)
        data = np.stack([training_sequence[i : i + model.context_size] for i in idxs])
        targets = np.stack([training_sequence[i + 1 : i + 1 + model.context_size] for i in idxs])
        return data, targets

    print("-- Training --------------------------------------------------------------")
    for i in range(num_batches):
        data, targets = get_batch()

        # Forward Pass
        logits = model.forward(data)
        loss = loss_fn.forward(logits, targets)

        # Backward Pass
        dlogits = loss_fn.backward()
        model.backward(dlogits)
        model.step()

        # Report batch loss and per-token perplexity
        perplexity = np.exp(loss)
        print(f"  {i + 1:6}/{num_batches}: {loss=:6.3f}  {perplexity=:7,.0f}")

        # Save Checkpoints
        if i % checkpoint_freq == 0 or i == num_batches - 1:
            path = Path(checkpoint_dir, f"checkpoint.{i + 1}.pkl")
            model.save(path)

    return model


def main(args: argparse.Namespace) -> None:
    """Entrypoint."""
    checkpoint_dir = _initialize_checkpoint_dir(args)
    tokenizer = _load_tokenizer(args.tokenizer_name)
    train_tokens = _load_tokens(args.tokenizer_name)
    if len(args.starting_checkpoint_name) > 0:
        if args.starting_checkpoint_iter <= 0:
            raise RuntimeError(
                "When providing a starting checkpoint, the iter number must also be provided and be positive"
            )
        model = _load_model_for_training(
            starting_checkpoint_name=args.starting_checkpoint_name,
            starting_checkpoint_iter=args.starting_checkpoint_iter,
            lr=args.learning_rate,
        )
    else:
        model = _initialize_model_for_training(
            vocab_size=tokenizer.vocab_size,
            lr=args.learning_rate,
        )
    _train_model(
        model=model,
        training_sequence=train_tokens,
        checkpoint_dir=checkpoint_dir,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Transformer model.")
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
        "--starting_checkpoint_name",
        type=str,
        required=False,
        default="",
        help="The name of a previously saved model checkpoint",
    )
    parser.add_argument(
        "-c",
        "--starting_checkpoint_iter",
        type=int,
        required=False,
        default=0,
        help="The iteration number of the previously saved model checkpoint",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        required=False,
        default=0.001,
        help="The learning rate for the Adam optimizer",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        required=False,
        default=32,
        help="The batch size",
    )
    parser.add_argument(
        "-nb",
        "--num_batches",
        type=int,
        required=False,
        default=50_000,
        help="The number of batches to process during training",
    )
    args = parser.parse_args()

    main(args)
