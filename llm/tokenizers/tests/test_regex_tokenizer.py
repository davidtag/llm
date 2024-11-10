"""Unit tests for regex_tokenizer.py.

These are largely interface tests. There are more involved functional tests in test_bpe.py.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from llm.tokenizers.cython.stdtoken import TokenPair
from llm.tokenizers.regex_tokenizer import RegexTokenizer
from llm.tokenizers.split_pattern import SplitPattern


class TestEmptyMerges(unittest.TestCase):
    """Unit tests for RegexTokenizer when there's an empty merge list."""

    def setUp(self) -> None:
        self.tokenizer = RegexTokenizer(
            merge_list=[],
            split_pattern=SplitPattern.get_pattern("gpt-4"),
        )

    def test_vocab_size(self) -> None:
        self.assertEqual(self.tokenizer.vocab_size, 256)

    def test_empty(self) -> None:
        tokens = self.tokenizer.encode("")
        self.assertEqual(tokens, [])

        text_bytes = self.tokenizer.decode_bytes([])
        self.assertEqual(text_bytes, b"")

        text = self.tokenizer.decode([])
        self.assertEqual(text, "")

    def test_encode_single_split(self) -> None:
        tokens = self.tokenizer.encode("aa")
        self.assertEqual(tokens, [97, 97])

        text_bytes = self.tokenizer.decode_bytes([97, 97])
        self.assertEqual(text_bytes, b"aa")

        text = self.tokenizer.decode([97, 97])
        self.assertEqual(text, "aa")

    def test_multiple_splits(self) -> None:
        tokens = self.tokenizer.encode("a b")
        self.assertEqual(tokens, [97, 32, 98])

        text_bytes = self.tokenizer.decode_bytes([97, 32, 98])
        self.assertEqual(text_bytes, b"a b")

        text = self.tokenizer.decode([97, 32, 98])
        self.assertEqual(text, "a b")


class TestSingleMerge(unittest.TestCase):
    """Unit tests for RegexTokenizer when there's a single merge rule."""

    def setUp(self) -> None:
        self.tokenizer = RegexTokenizer(
            merge_list=[(TokenPair(32, 32), 256)],
            split_pattern=SplitPattern.get_pattern("gpt-4"),
        )

    def test_vocab_size(self) -> None:
        self.assertEqual(self.tokenizer.vocab_size, 257)

    def test_empty(self) -> None:
        tokens = self.tokenizer.encode("")
        self.assertEqual(tokens, [])

        text_bytes = self.tokenizer.decode_bytes([])
        self.assertEqual(text_bytes, b"")

        text = self.tokenizer.decode([])
        self.assertEqual(text, "")

    def test_encode_single_split(self) -> None:
        tokens = self.tokenizer.encode("aa")
        self.assertEqual(tokens, [97, 97])

        text_bytes = self.tokenizer.decode_bytes([97, 97])
        self.assertEqual(text_bytes, b"aa")

        text = self.tokenizer.decode([97, 97])
        self.assertEqual(text, "aa")

    def test_multiple_splits(self) -> None:
        tokens = self.tokenizer.encode("a b")
        self.assertEqual(tokens, [97, 32, 98])

        text_bytes = self.tokenizer.decode_bytes([97, 32, 98])
        self.assertEqual(text_bytes, b"a b")

        text = self.tokenizer.decode([97, 32, 98])
        self.assertEqual(text, "a b")

    def test_merge(self) -> None:
        tokens = self.tokenizer.encode("a   b")
        self.assertEqual(tokens, [97, 256, 32, 98])

        text_bytes = self.tokenizer.decode_bytes([97, 256, 32, 98])
        self.assertEqual(text_bytes, b"a   b")

        text = self.tokenizer.decode([97, 256, 32, 98])
        self.assertEqual(text, "a   b")


class TestSaveAndLoad(unittest.TestCase):
    """Unit tests for saving & loading tokenizer state."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(delete=False)
        self.tokenizer = RegexTokenizer(
            merge_list=[
                (TokenPair(32, 32), 256),  # " "
                (TokenPair(97, 98), 257),  # "ab"
            ],
            split_pattern=SplitPattern.get_pattern("gpt-4"),
        )
        self.tokenizer.trained_cache.update(
            {"abab": [257, 257]},
        )
        self.save_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_save(self) -> None:
        self.tokenizer.save(self.save_dir)

        vocab_path = Path(self.save_dir, "vocab.txt")
        with open(vocab_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 258)
        self.assertEqual(lines[-2].strip(), "[256] : [  ]")
        self.assertEqual(lines[-1].strip(), "[257] : [ab]")

        cache_path = Path(self.save_dir, "cache.txt")
        with open(cache_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].strip(), "YWJhYg==:[257, 257]")  # YWJhYg== is base64.b64encode(b"abab")

    def test_round_trip(self) -> None:
        self.tokenizer.save(self.save_dir)

        new_tokenizer = RegexTokenizer.load(self.save_dir)
        self.assertEqual(new_tokenizer.split_pattern, self.tokenizer.split_pattern)
        self.assertEqual(new_tokenizer.merge_list, self.tokenizer.merge_list)
        self.assertEqual(new_tokenizer.trained_cache, self.tokenizer.trained_cache)
        self.assertDictEqual(new_tokenizer.runtime_cache, {})


@patch("builtins.print")
class TestTrain(unittest.TestCase):
    """Unit tests for train."""

    def test_train(self, mock_print: MagicMock) -> None:
        """Test training merges."""
        tokenizer = RegexTokenizer.train(
            text=" aa   bbaa   ,",
            split_pattern=SplitPattern.get_pattern("gpt-4"),
            num_merges=100,
            verbose=True,
        )

        self.assertEqual(tokenizer.split_pattern, SplitPattern.get_pattern("gpt-4"))
        self.assertEqual(
            tokenizer.merge_list,
            [
                (TokenPair(first=32, second=32), 256),  # " "
                (TokenPair(first=97, second=97), 257),  # "aa"
            ],
        )

        mock_print.assert_called()

    def test_train_piece_cache(self, mock_print: MagicMock) -> None:
        """Test additional training of the piece cache."""
        tokenizer = RegexTokenizer(
            merge_list=[
                (TokenPair(32, 32), 256),  # " "
                (TokenPair(97, 98), 257),  # "ab"
            ],
            split_pattern=SplitPattern.get_pattern("gpt-4"),
        )

        new_tokenizer = tokenizer.train_piece_cache(
            text=" ababab ababab    ",
            num_extra_pieces=10,
            verbose=True,
        )

        self.assertEqual(new_tokenizer.split_pattern, SplitPattern.get_pattern("gpt-4"))
        self.assertEqual(new_tokenizer.merge_list, tokenizer.merge_list)
        self.assertEqual(len(new_tokenizer.trained_cache), len(tokenizer.trained_cache) + 2)

        self.assertEqual(new_tokenizer.trained_cache[" ababab"], [32, 257, 257, 257])
        self.assertEqual(new_tokenizer.trained_cache["    "], [256, 256])

        mock_print.assert_called()
