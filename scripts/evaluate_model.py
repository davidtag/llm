"""Evaluate the performance of a trained model on a split."""

import argparse
from pathlib import Path

import numpy as np

from llm.data.loaders import load_token_file
from llm.data.registry import TokenRegistry, ModelRegistry
from llm.loss.cross_entropy import CrossEntropyLoss
from llm.models.transformer import Transformer


def _load_tokens(name: str, split: str) -> np.ndarray:
    token_registry = TokenRegistry(tokenizer_name=name)
    if split == "train":
        tokens = load_token_file(token_registry.train_token_file)
    elif split == "val":
        tokens = load_token_file(token_registry.val_token_file)
    elif split == "test":
        tokens = load_token_file(token_registry.test_token_file)
    else:
        raise ValueError(f"Unrecognize split value: {split}")
    print(f"Loaded {len(tokens):,} {split} tokens")
    return np.array(tokens)


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


def _evaluate_model(
    model: Transformer,
    eval_tokens: np.ndarray,
    batch_size: int,
    num_batches: int,
) -> None:
    loss_fn = CrossEntropyLoss(enable_grad=False)

    num_predictions = 0
    total_loss = 0

    def get_batch() -> tuple[np.ndarray, np.ndarray]:
        idxs = np.random.randint(low=0, high=len(eval_tokens) - model.context_size, size=batch_size)
        data = np.stack([eval_tokens[i : i + model.context_size] for i in idxs])
        targets = np.stack([eval_tokens[i + 1 : i + 1 + model.context_size] for i in idxs])
        return data, targets

    print("-- Evaluating --------------------------------------------------------------")

    for i in range(num_batches):
        data, targets = get_batch()

        logits = model.forward(data)
        batch_loss = loss_fn.forward(logits, targets)

        n = targets.size
        num_predictions += n
        total_loss += batch_loss * n

        avg_loss = total_loss / num_predictions
        perplexity = np.exp(avg_loss)

        print(f"  {i + 1:6}/{num_batches}: {batch_loss=:6.3f}  |  {avg_loss=:6.3f}  {perplexity=:7,.0f}")

    print()
    print(f"Evaluation on {num_predictions:,} predictions: {avg_loss=:6.3f}  {perplexity=:7,.0f}")


def main(args: argparse.Namespace) -> None:
    """Entrypoint."""
    tokens = _load_tokens(name=args.tokenizer_name, split=args.split)
    model = _load_model_for_eval(
        checkpoint_name=args.checkpoint_name,
        checkpoint_iter=args.checkpoint_iter,
    )

    _evaluate_model(
        model=model,
        eval_tokens=tokens,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the performance of a trained model on a split.")
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
        help="The name of the saved model checkpoint",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_iter",
        type=int,
        required=True,
        default=0,
        help="The iteration number of the saved model checkpoint",
    )
    parser.add_argument(
        "-s",
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="The data split used for evaluation",
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
        default=16,
        help="The number of batches to process for evaluation",
    )
    args = parser.parse_args()

    main(args)
