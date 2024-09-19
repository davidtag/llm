"""Implementation of a Transformer model architecture."""

from typing import Optional

import numpy as np

from llm.layers.block_stack import BlockStack
from llm.layers.embedding import Embedding
from llm.layers.linear import Linear
from llm.optimizers import Optimizer
from llm.utils.math import softmax


class Transformer(object):
    """A Transformer architecture for sequence processing."""

    def __init__(
        self,
        vocab_size: int,
        n_blocks: int = 6,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        h: int = 8,
        d_ff: int = 2048,
        enable_grad: bool = True,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the model."""
        self.vocab_size = vocab_size
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff
        self.enable_grad = enable_grad
        self.optimizer = optimizer

        self.embedding_layer = Embedding(
            vocab_size=vocab_size,
            d_model=d_model,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )
        self.encoder = BlockStack(
            n_blocks=n_blocks,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            h=h,
            d_ff=d_ff,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )
        self.unembedding_layer = Linear(
            n_input=d_model,
            n_output=vocab_size,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return self.embedding_layer.n_params + self.encoder.n_params + self.unembedding_layer.n_params

    def forward(self, input_sequence: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert input_sequence.ndim == 1
        assert input_sequence.min() >= 0 and input_sequence.max() < self.vocab_size

        raw_embedding = self.embedding_layer.forward(input_sequence)  # shape = (N, d_model)
        refined_embedding = self.encoder.forward(raw_embedding)  # shape = (N, d_model)
        logits = self.unembedding_layer.forward(refined_embedding)  # shape = (N, vocab_size)

        return logits

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        assert dout.shape == (self.embedding_layer.cache["input_sequence"].shape[0], self.vocab_size)

        dlogits = dout
        self.unembedding_layer.backward(dlogits)
        drefined_embedding = self.unembedding_layer.cache["dx"]
        self.encoder.backward(drefined_embedding)
        draw_embedding = self.encoder.cache["dx"]
        self.embedding_layer.backward(draw_embedding)

    def step(self) -> None:
        """Performs a single optimization step."""
        assert self.enable_grad, "Cannot take an optimization step with enable_grad=False"
        assert self.optimizer, "Cannot take an optimization step with optimizer=None"

        self.unembedding_layer.step()
        self.encoder.step()
        self.embedding_layer.step()

    def predict(self, input_sequence: np.ndarray) -> np.ndarray:
        """Generate a probability distribution over the next token for a given input sequence."""
        logits = self.forward(input_sequence)
        next_token_logits = logits[-1]
        probabilities = softmax(next_token_logits)
        return probabilities

    def generate(self, start_sequence: np.ndarray, max_tokens: int = 5) -> np.ndarray:
        """Generate an output sequence based on predicted next token probabilities."""
        current_sequence = start_sequence.copy()
        output_sequence = []

        for _ in range(max_tokens):
            probs = self.predict(current_sequence)
            token = np.random.choice(self.vocab_size, p=probs)
            np.append(current_sequence, token)
            output_sequence.append(token)

        return np.array(output_sequence)
