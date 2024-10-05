"""."""

import time

import tiktoken


def encode_shakespeare_with_gpt2() -> None:
    """."""
    print("Reading file...")
    with open("data/blob/t8.shakespeare.txt", mode="r", encoding="utf-8") as f:
        text = f.read()

    print("Loading encoder...")
    enc = tiktoken.get_encoding("gpt2")

    for i in range(100):
        print("Encoding...")
        start = time.monotonic()
        out = enc.encode(text)
        end = time.monotonic()
        duration = end - start
        print(f"Done encoding: {len(text)} -> {len(out)}. {duration:.3f}s")


if __name__ == "__main__":
    encode_shakespeare_with_gpt2()
