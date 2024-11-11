"""Helpers for data loading."""

from os import PathLike
import pickle

import numpy as np


def write_text_file(file_path: PathLike, text: str) -> None:
    """Write a UTF-8 encoded text file from a string in memory."""
    with open(file_path, mode="w", encoding="utf-8") as f:
        f.write(text)


def load_text_file(file_path: PathLike) -> str:
    """Load a UTF-8 encoded text file into a string in memory."""
    with open(file_path, mode="r", encoding="utf-8") as f:
        text = f.read()
    return text


def write_token_file(file_path: PathLike, tokens: list[int]) -> None:
    """Write a line-separated token file."""
    with open(file_path, mode="w", encoding="utf-8") as f:
        for token in tokens:
            f.write(f"{token}\n")


def load_token_file(file_path: PathLike) -> list[int]:
    """Load a line-separated token file."""
    with open(file_path, mode="r", encoding="utf-8") as f:
        data = f.read()
    return list(map(int, data.splitlines()))


def load_cifar_10_split(file_path: PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Load a CIFAR-10 data split."""
    with open(file_path, "rb") as f:
        split = pickle.load(f, encoding="bytes")

    raw_data = split[b"data"]
    assert raw_data.shape == (10_000, 3_072)  # stored with channel as first dimension
    data = raw_data.reshape(10_000, 3, 32, 32)
    data = np.transpose(data, axes=(0, 2, 3, 1))
    assert data.shape == (10_000, 32, 32, 3)

    targets = np.array(split[b"labels"], dtype=np.int32)
    assert targets.shape == (10_000,)

    return data, targets
