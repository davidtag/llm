"""Example client for tokenization and text completion APIs."""

import argparse

import requests


SERVER_PROTOCOL = "http"
SERVER_HOST = "localhost"
TOKENIZE_ENDPOINT = "/api/tokenize/"
COMPLETION_ENDPOINT = "/api/complete/"


def _tokenize(host: str) -> None:
    url = host + TOKENIZE_ENDPOINT
    while True:
        user_text = input("Enter text to tokenize:")
        user_bytes = user_text.encode("utf-8")
        response = requests.post(url, data=user_bytes, stream=True)
        for chunk in response.iter_content(chunk_size=1):
            if chunk:
                print(chunk.decode("utf-8"), end="")


def _complete(host: str) -> None:
    url = host + COMPLETION_ENDPOINT
    while True:
        user_text = input("Enter text to complete:")
        user_bytes = user_text.encode("utf-8")
        response = requests.post(url, data=user_bytes, stream=True)
        for chunk in response.iter_content(chunk_size=1):
            if chunk:
                print(chunk.decode("utf-8"), end="")


def main(args: argparse.Namespace) -> None:
    """Entrypoint."""
    host = f"{SERVER_PROTOCOL}://{SERVER_HOST}:{args.port}"
    if args.endpoint == "tokenize":
        _tokenize(host)
    elif args.endpoint == "complete":
        _complete(host)
    else:
        raise ValueError("Unrecognized endpoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example client for tokenization and text completion APIs.")
    parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        required=True,
        choices=["tokenize", "complete"],
        help="The server API to call",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        required=False,
        default=8000,
        help="Port number for the HTTP server",
    )
    args = parser.parse_args()

    main(args)
