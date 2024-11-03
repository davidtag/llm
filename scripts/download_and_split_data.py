"""Download the raw text data and generate train/text/val splits."""

import requests

from llm.data.loaders import write_text_file, load_text_file
from llm.data.registry import DataRegistry


_registry = DataRegistry()


def _download_and_save_raw_data() -> None:
    response = requests.get(url=_registry.SOURCE_URL)
    raw_text = response.text
    assert len(raw_text) == _registry.EXPECTED_TOTAL_SIZE

    _registry.text_dir.mkdir(parents=True, exist_ok=True)
    write_text_file(_registry.raw_text_file, raw_text)


def _create_splits() -> None:
    raw_text = load_text_file(_registry.raw_text_file)
    assert len(raw_text) == _registry.EXPECTED_TOTAL_SIZE
    start = 0

    train_split = raw_text[start : start + _registry.TRAIN_SIZE]
    start += _registry.TRAIN_SIZE

    val_split = raw_text[start : start + _registry.VAL_SIZE]
    start += _registry.VAL_SIZE

    test_split = raw_text[start : start + _registry.TEST_SIZE]
    start += _registry.TEST_SIZE

    assert start == len(raw_text)
    write_text_file(_registry.train_text_file, train_split)
    write_text_file(_registry.val_text_file, val_split)
    write_text_file(_registry.test_text_file, test_split)


def main() -> None:
    """Entrypoint."""
    _download_and_save_raw_data()
    _create_splits()


if __name__ == "__main__":
    main()
