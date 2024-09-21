"""Implementation of a learned embedding layer."""

from typing import Optional

import numpy as np

from llm.optimizers import Optimizer


class Embedding:
    """A learned embedding layer for a fixed vocab size.

    TODO(dtag): Support positional encodings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        enable_grad: bool = True,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the layer."""
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.enable_grad = enable_grad
        self.optimizer = optimizer
        self.cache = {}

        self.embedding_mat = np.random.normal(
            loc=0,
            scale=np.sqrt(2 / d_model),
            size=(vocab_size, d_model),
        )

        self.embedding_mat_opt = optimizer.get_parameter_optimizer(self.embedding_mat) if optimizer else None

        # Pre-allocate the gradient to save compute on each backward pass and avoid large memory allocations
        self.cache["dembedding_mat"] = np.zeros_like(self.embedding_mat)

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return self.embedding_mat.size

    def forward(self, input_sequence: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert input_sequence.ndim == 1
        assert input_sequence.min() >= 0 and input_sequence.max() < self.vocab_size

        out = self.embedding_mat[input_sequence]  # shape = (len(input_sequence), d_model)

        if self.enable_grad:
            self.cache["input_sequence"] = input_sequence

        return out

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        input_sequence = self.cache["input_sequence"]
        assert dout.shape == (input_sequence.shape[0], self.d_model)

        dembedding_mat = self.cache["dembedding_mat"]
        dembedding_mat[:] = 0  # zero-out a static gradient matrix instead of re-allocating

        for i, row in enumerate(dout):
            dembedding_mat[input_sequence[i]] += row

        self.cache["dembedding_mat"] = dembedding_mat

    def step(self) -> None:
        """Performs a single optimization step."""
        assert self.enable_grad, "Cannot take an optimization step with enable_grad=False"
        assert self.optimizer, "Cannot take an optimization step with optimizer=None"

        assert self.embedding_mat_opt is not None

        self.embedding_mat_opt.step(self.cache["dembedding_mat"])
