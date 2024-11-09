"""Download the pre-trained model weights."""

import hashlib
from pathlib import Path

import requests

from llm.data.registry import ModelRegistry


_registry = ModelRegistry()


def _download_and_save_checkpoint() -> None:
    response = requests.get(url=_registry.DEFAULT_1M_URL, allow_redirects=False, verify=True, stream=True)
    response.raise_for_status()

    data = bytearray()
    for chunk in response.iter_content(chunk_size=4096):
        data.extend(chunk)
        if len(data) > _registry.DEFAULT_1M_SIZE:
            raise ValueError("Download data size exceeded expected size.")
    if len(data) != _registry.DEFAULT_1M_SIZE:
        raise RuntimeError("Download data size is not as expected. File host might be compromised.")

    sha256 = hashlib.sha256()
    sha256.update(data)
    digest = sha256.hexdigest()
    if digest != _registry.DEFAULT_1M_DIGEST:
        raise RuntimeError("Download data digest is not as expected. File host might be compromised.")

    checkpoint_dir = Path(_registry.checkpoint_dir, _registry.DEFAULT_1M_NAME)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir, "checkpoint.0.pkl")
    with open(path, "wb") as f:
        f.write(data)


def main() -> None:
    """Entrypoint."""
    _download_and_save_checkpoint()


if __name__ == "__main__":
    main()
