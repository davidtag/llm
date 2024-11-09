"""Registry of all assets."""

from pathlib import Path


class _BaseRegistry:

    @property
    def project_dir(self) -> Path:
        """Root directory for the git project."""
        return Path(__file__).parent.parent.parent

    @property
    def assets_dir(self) -> Path:
        """Root directory for all assets."""
        return Path(self.project_dir, "assets")


class DataRegistry(_BaseRegistry):
    """Registry of all data assets."""

    SOURCE_URL = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"

    _BYTE = 1
    _KB = 1024 * _BYTE
    _MB = 1024 * _KB

    EXPECTED_TOTAL_SIZE = 5_458_199

    TEST_SIZE = 1 * _MB
    VAL_SIZE = 1 * _MB
    TRAIN_SIZE = EXPECTED_TOTAL_SIZE - TEST_SIZE - VAL_SIZE

    @property
    def text_dir(self) -> Path:
        """Root directory for all text assets."""
        return Path(self.assets_dir, "text")

    @property
    def raw_text_file(self) -> Path:
        """Path containing the raw text data."""
        return Path(self.text_dir, "raw.txt")

    @property
    def train_text_file(self) -> Path:
        """Path containing the train split of the text data."""
        return Path(self.text_dir, "train.txt")

    @property
    def val_text_file(self) -> Path:
        """Path containing the val split of the text data."""
        return Path(self.text_dir, "val.txt")

    @property
    def test_text_file(self) -> Path:
        """Path containing the test split of the text data."""
        return Path(self.text_dir, "test.txt")


class TokenizerRegistry(_BaseRegistry):
    """Registry for trained tokenizer checkpoints."""

    @property
    def checkpoint_dir(self) -> Path:
        """Root directory for all trained BPE tokenizer checkpoints."""
        return Path(self.assets_dir, "bpe_checkpoints")


class TokenRegistry(_BaseRegistry):
    """Registry for all token-based assets."""

    def __init__(self, tokenizer_name: str) -> None:
        """Initialize the registry."""
        super().__init__()
        self.tokenizer_name = tokenizer_name

    @property
    def token_dir(self) -> Path:
        """Root directory for all token assets for this tokenizer."""
        return Path(self.assets_dir, "tokens", self.tokenizer_name)

    @property
    def train_token_file(self) -> Path:
        """Path containing the train split of the text data."""
        return Path(self.token_dir, "train.txt")

    @property
    def val_token_file(self) -> Path:
        """Path containing the val split of the text data."""
        return Path(self.token_dir, "val.txt")

    @property
    def test_token_file(self) -> Path:
        """Path containing the test split of the text data."""
        return Path(self.token_dir, "test.txt")


class ModelRegistry(_BaseRegistry):
    """Registry for trained model checkpoints."""

    DEFAULT_1M_NAME = "default_1m"
    DEFAULT_1M_URL = "https://dtag.ai/wp-content/uploads/2024/11/default_1m.pkl_.pdf"
    DEFAULT_1M_SIZE = 13_041_634
    DEFAULT_1M_DIGEST = "241bcfee6623871b8fb390fde870f1e4f90581131489487e288f360e9a0ac1fe"

    @property
    def checkpoint_dir(self) -> Path:
        """Root directory for all trained model checkpoints."""
        return Path(self.assets_dir, "model_checkpoints")
