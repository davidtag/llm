from os import PathLike
from pathlib import Path
import time

import numpy as np

from llm.loss.cross_entropy import CrossEntropyLoss
from llm.models.transformer import Transformer
from llm.optimizers.adam import Adam
from llm.tokenizers.regex_tokenizer import RegexTokenizer


DATA_FILE = Path("data/blob/t8.shakespeare.txt")
TOKENIZER_MODEL = Path("out/")
CHECKPOINTS_DIR = Path("checkpoints/")
# STARTING_CHECKPOINT = Path(CHECKPOINTS_DIR, "v1", "checkpoint.49000.1730125678383.pkl")
# STARTING_CHECKPOINT = Path(CHECKPOINTS_DIR, "checkpoint.3000.1730152270336.pkl")
# v0: lr=0.001 for 100 iters. context_size=32. batch_size=32.
# v1: lr=0.001 for 50000 iters. context_size=32. batch_size=32.
# v2: start form v1.49000 and lr=0.0005. context_size=32. batch_size=32.
# v3: lr=0.001 for 37000 iters. context_size=128. batch_size=16.
# v4: start from v3.37000 and lr=0.0002. context_size=128. batch_size=16.
# v5: start from v4.33000 and lr=0.0001. context_size=128. batch_size=32.
STARTING_CHECKPOINT = Path(CHECKPOINTS_DIR, "v5", "checkpoint.7000.1730563202097.pkl")


def _load_text_file(file_path: PathLike) -> str:
    with open(file_path, mode="r", encoding="utf-8") as f:
        data = f.read()
    return data


def _load_train_test_split(file_path: PathLike, val_length: int = 2 * 1024 * 1024) -> tuple[str, str]:
    data = _load_text_file(file_path)
    train = data[:-val_length]
    val = data[-val_length:]
    return train, val


def _train_model(
    training_sequence,
    vocab_size: int,
    num_epochs: int = 20,
    context_size: int = 16,
    batch_size: int = 32,
) -> Transformer:
    optimizer = Adam(lr=0.0001)
    model = Transformer(
        vocab_size=vocab_size,
        context_size=context_size,
        n_blocks=6,
        d_model=64,
        d_k=8,
        d_v=8,
        h=8,
        d_ff=256,
        optimizer=optimizer,
    )
    if STARTING_CHECKPOINT is not None:
        model.load(STARTING_CHECKPOINT)
    loss_fn = CrossEntropyLoss()

    def get_batch() -> tuple[np.ndarray, np.ndarray]:
        idxs = np.random.randint(low=0, high=len(training_sequence) - context_size, size=batch_size)
        data = np.stack([training_sequence[i : i + context_size] for i in idxs])
        targets = np.stack([training_sequence[i + 1 : i + 1 + context_size] for i in idxs])
        return data, targets

    for i in range(num_epochs):
        data, targets = get_batch()

        # Forward Pass
        logits = model.forward(data)
        loss = loss_fn.forward(logits, targets)

        # Backward Pass
        dlogits = loss_fn.backward()
        model.backward(dlogits)
        model.step()

        print(f"  {i + 1:6}/{num_epochs}: {loss:.3f}")

        # Save Checkpoints
        if i % 1000 == 0 or i == num_epochs - 1:
            now = int(time.time() * 1000)
            path = Path(CHECKPOINTS_DIR, f"checkpoint.{i}.{now}.pkl")
            model.save(path)

    return model


def main():
    # Load the training / val text
    train_text, val_text = _load_train_test_split(DATA_FILE)
    print(f"Loaded data: {len(train_text)=:,} chars, {len(val_text)=:,} chars")

    # Tokenize the training / val text using a pre-trained tokenizer
    tokenizer = RegexTokenizer.load(TOKENIZER_MODEL)
    train_tokens = np.array(tokenizer.encode(train_text, use_cache=False))
    val_tokens = np.array(tokenizer.encode(val_text, use_cache=False))
    print(f"Tokenizes data: {len(train_tokens)=:,} chars, {len(val_tokens)=:,} chars")

    # Train the Transformer model
    print("Training transformer...")
    model = _train_model(
        training_sequence=train_tokens,
        vocab_size=tokenizer.vocab_size,
        num_epochs=0,
        context_size=128,
        batch_size=32,
    )

    # Prediction
    print("Generating text...")
    print("=================================================================================")
    # start_str = " The house was on fire"
    # start_str = " From fairest creatures we desire"
    # start_str = " In fair Verona, where we lay our scene,"
    # start_str = " ROMEO married JULIET, therefore she is his"
    start_str = "ACT I. Scene I.\nRome. A public place.\n\nEnter ROMEO.\n\n"
    # start_str = " I went to Verona, which is in the country of"
    start_sequence = np.array(tokenizer.encode(start_str))
    for _ in range(1):
        output_tokens = model.generate(start_sequence, max_tokens=500)
        output_str = tokenizer.decode(output_tokens.tolist())
        complete_str = start_str + output_str
        print(complete_str)
    print("=================================================================================")


if __name__ == "__main__":
    main()
