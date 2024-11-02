"""Registry of all data assets."""

from pathlib import Path


class DataRegistry:
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
    def project_dir(self) -> Path:
        """Root directory for the git project."""
        return Path(__file__).parent.parent.parent

    @property
    def assets_dir(self) -> Path:
        """Root directory for all assets."""
        return Path(self.project_dir, "assets")

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
