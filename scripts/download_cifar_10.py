"""Download the CIFAR-10 dataset."""

import hashlib
import tarfile

import requests

from llm.data.registry import ImageDataRegistry


_registry = ImageDataRegistry()


def _download_tar_file() -> None:
    response = requests.get(url=_registry.SOURCE_URL, allow_redirects=False, verify=True, stream=True)
    response.raise_for_status()

    data = bytearray()
    for chunk in response.iter_content(chunk_size=4096):
        data.extend(chunk)
        if len(data) > _registry.EXPECTED_TOTAL_SIZE:
            raise ValueError("Download data size exceeded expected size.")
    if len(data) != _registry.EXPECTED_TOTAL_SIZE:
        raise RuntimeError("Download data size is not as expected. File host might be compromised.")

    md5 = hashlib.md5()
    md5.update(data)
    digest = md5.hexdigest()
    if digest != _registry.MD5_SUM:
        raise RuntimeError("Download data digest is not as expected. File host might be compromised.")

    _registry.image_dir.mkdir(parents=True, exist_ok=True)
    with open(_registry.tar_file, "wb") as f:
        f.write(data)


def _unpack_tar_file() -> None:
    with tarfile.open(_registry.tar_file, "r:gz") as tar:
        tar.extractall(path=_registry.image_dir, filter="data")


def main() -> None:
    """Entrypoint."""
    _download_tar_file()
    _unpack_tar_file()


if __name__ == "__main__":
    main()
