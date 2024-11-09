"""Unit tests for loaders.py"""

from pathlib import Path
import tempfile
import unittest

from llm.data.loaders import (
    write_text_file,
    load_text_file,
    write_token_file,
    load_token_file,
)


class TestTextFile(unittest.TestCase):
    """Unit tests for writing/loading from text file."""

    def test_basic(self) -> None:
        data = "some string"

        with tempfile.NamedTemporaryFile() as f:
            path = Path(f.name)
            write_text_file(path, data)

            data2 = load_text_file(path)
            self.assertEqual(data2, data)


class TestTokenFile(unittest.TestCase):
    """Unit tests for writing/loading from a token file."""

    def test_basic(self) -> None:
        data = [137, 259, 143378]

        with tempfile.NamedTemporaryFile() as f:
            path = Path(f.name)
            write_token_file(path, data)

            data2 = load_token_file(path)
            self.assertEqual(data2, data)
