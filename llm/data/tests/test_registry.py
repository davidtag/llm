"""Unit tests for registry.py."""

import unittest


from llm.data.registry import (
    DataRegistry,
    TokenizerRegistry,
    TokenRegistry,
    ModelRegistry,
)


class TestDataRegistry(unittest.TestCase):
    """Unit tests for DataRegistry."""

    def test_all(self) -> None:
        registry = DataRegistry()

        self.assertTrue(registry.text_dir.as_posix().endswith("assets/text"))
        self.assertTrue(registry.raw_text_file.as_posix().endswith("assets/text/raw.txt"))
        self.assertTrue(registry.train_text_file.as_posix().endswith("assets/text/train.txt"))
        self.assertTrue(registry.val_text_file.as_posix().endswith("assets/text/val.txt"))
        self.assertTrue(registry.test_text_file.as_posix().endswith("assets/text/test.txt"))


class TestTokenizerRegistry(unittest.TestCase):
    """Unit tests for TokenizerRegistry."""

    def test_all(self) -> None:
        registry = TokenizerRegistry()

        self.assertTrue(registry.checkpoint_dir.as_posix().endswith("assets/bpe_checkpoints"))


class TestTokenRegistry(unittest.TestCase):
    """Unit tests for TokenRegistry."""

    def test_all(self) -> None:
        registry = TokenRegistry(tokenizer_name="tokenizer_123")

        self.assertTrue(registry.token_dir.as_posix().endswith("assets/tokens/tokenizer_123"))
        self.assertTrue(
            registry.train_token_file.as_posix().endswith("assets/tokens/tokenizer_123/train.txt")
        )
        self.assertTrue(registry.val_token_file.as_posix().endswith("assets/tokens/tokenizer_123/val.txt"))
        self.assertTrue(registry.test_token_file.as_posix().endswith("assets/tokens/tokenizer_123/test.txt"))


class TestModelRegistry(unittest.TestCase):
    """Unit tests for ModelRegistry."""

    def test_all(self) -> None:
        registry = ModelRegistry()

        self.assertTrue(registry.checkpoint_dir.as_posix().endswith("assets/model_checkpoints"))
