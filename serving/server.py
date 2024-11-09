"""Example server for serving tokenization and text completions.

NOTE: this is just for illustrative purposes. The current RegexTokenizer and Transformer classes are not
designed to be thread-safe.
"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from llm.data.registry import ModelRegistry, TokenizerRegistry
from llm.models.transformer import Transformer
from llm.tokenizers import RegexTokenizer
from llm.tokenizers import bpe


def _load_tokenizer(name: str) -> RegexTokenizer:
    tokenizer_registry = TokenizerRegistry()
    checkpoint_dir = Path(tokenizer_registry.checkpoint_dir, name)
    if not checkpoint_dir.exists():
        raise RuntimeError(f"Checkpoint dir {checkpoint_dir} doesn't exist. Did you train the tokenizer?")
    tokenizer = RegexTokenizer.load(checkpoint_dir)
    print(
        f"Loaded tokenizer '{name}' with "
        f"vocab_size={tokenizer.vocab_size:,} and cache_size={len(tokenizer.trained_cache):,}"
    )
    return tokenizer


def _load_model_for_eval(
    checkpoint_name: str,
    checkpoint_iter: str,
) -> Transformer:
    model_registry = ModelRegistry()
    checkpoint_path = Path(
        model_registry.checkpoint_dir,
        checkpoint_name,
        f"checkpoint.{checkpoint_iter}.pkl",
    )

    model = Transformer.load_for_eval(model_file=checkpoint_path)
    print(f"Loaded model {checkpoint_name}:{checkpoint_iter} with n_params={model.n_params:,}")
    return model


class RequestHandler(BaseHTTPRequestHandler):
    """Handler for a single client request."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize and handle a request."""
        server = args[2]
        assert isinstance(server, Server)
        self.tokenizer = server.tokenizer
        self.model = server.model
        super().__init__(*args, **kwargs)

    def do_POST(self) -> None:
        """Handle a POST request."""
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)

        if self.path == "/api/tokenize/":
            self._resolve_tokenize(body)
        elif self.path == "/api/complete/":
            self._resolve_complete(body)
        else:
            self.send_error(404, "Endpoint not found")

    def _resolve_tokenize(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()

        text = body.decode("utf-8")
        tokens = self.tokenizer.encode(text)

        for token in tokens:
            token_bytes = self.tokenizer.decode_bytes([token])
            token_text = bpe.render_bytes(token_bytes)
            self.wfile.write(f"{token:>6,}: [{token_text}]\n".encode("utf-8"))
            self.wfile.flush()  # Ensure immediate transmission

    def _resolve_complete(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()

        text = body.decode("utf-8")
        tokens = self.tokenizer.encode(text)
        start_sequence = np.array(tokens)

        for token in self.model.generate_stream(start_sequence, max_tokens=500):
            token_bytes = self.tokenizer.decode_bytes([token])
            self.wfile.write(token_bytes)
            self.wfile.flush()  # Ensure immediate transmission
        self.wfile.write(b"\n")
        self.wfile.flush()


class Server(HTTPServer):
    """Custom server implementation."""

    _LOCALHOST = "127.0.0.1"

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the server."""
        self.args = args
        self.tokenizer = _load_tokenizer(args.tokenizer_name)
        self.model = _load_model_for_eval(args.checkpoint_name, args.checkpoint_iter)
        super().__init__(
            server_address=(self._LOCALHOST, args.port),
            RequestHandlerClass=RequestHandler,
        )

    def finish_request(self, request, client_address) -> None:  # noqa: ANN001
        """Handle a single client request."""
        self.RequestHandlerClass(request, client_address, self)  # type: ignore[arg-type]


def main(args: argparse.Namespace) -> None:
    """Entrypoint."""
    print(f"Starting HTTP server on port {args.port}")
    httpd = Server(args)
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example server for serving tokenization and text completions."
    )
    parser.add_argument(
        "-t",
        "--tokenizer_name",
        type=str,
        required=True,
        default="",
        help="The name of the saved tokenizer checkpoint",
    )
    parser.add_argument(
        "-n",
        "--checkpoint_name",
        type=str,
        required=True,
        default="",
        help="The name of a previously saved model checkpoint",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_iter",
        type=int,
        required=True,
        default=0,
        help="The iteration number of the previously saved model checkpoint",
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
