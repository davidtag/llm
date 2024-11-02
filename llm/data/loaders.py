"""Helpers for data loading."""

from os import PathLike


def write_text_file(file_path: PathLike, text: str) -> None:
    """Write a UTF-8 encoded text file from a string in memory."""
    with open(file_path, mode="w", encoding="utf-8") as f:
        f.write(text)


def load_text_file(file_path: PathLike) -> str:
    """Load a UTF-8 encoded text file into a string in memory."""
    with open(file_path, mode="r", encoding="utf-8") as f:
        text = f.read()
    return text
