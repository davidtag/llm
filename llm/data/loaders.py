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
