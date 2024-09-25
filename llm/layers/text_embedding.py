"""Implementation of a learned embedding layer."""

from typing import Optional

import numpy as np

from llm.constants import DType, DEFAULT_DTYPE
from llm.optimizers import Optimizer


class TextEmbedding:
    """A learned text embedding layer for a fixed vocab size & context window."""

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        d_model: int = 512,
        dtype: DType = DEFAULT_DTYPE,
        enable_grad: bool = True,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the layer."""
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model
        self.dtype = dtype
        self.enable_grad = enable_grad
        self.optimizer = optimizer
        self.cache = {}

        self.token_embedding_matrix = np.random.standard_normal(size=(vocab_size, d_model)).astype(dtype)
        self.position_embedding_matrix = np.random.standard_normal(size=(context_size, d_model)).astype(dtype)

        self.token_embedding_matrix_opt = (
            optimizer.get_parameter_optimizer(self.token_embedding_matrix) if optimizer else None
        )
        self.position_embedding_matrix_opt = (
            optimizer.get_parameter_optimizer(self.position_embedding_matrix) if optimizer else None
        )

        # Pre-allocate the gradient to save compute on each backward pass and avoid large memory allocations
        self.cache["dtoken_embedding_matrix"] = np.zeros_like(self.token_embedding_matrix)
        self.cache["dposition_embedding_matrix"] = np.zeros_like(self.position_embedding_matrix)

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return self.token_embedding_matrix.size + self.position_embedding_matrix.size

    def forward(self, input_sequence: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert input_sequence.ndim == 1
        assert input_sequence.min() >= 0 and input_sequence.max() < self.vocab_size
        n = len(input_sequence)
        assert n <= self.context_size

        token_embeddings = self.token_embedding_matrix[input_sequence]  # shape = (n, d_model)
        position_embeddings = self.position_embedding_matrix[:n]  # shape = (n, d_model)
        out = token_embeddings + position_embeddings

        if self.enable_grad:
            self.cache["input_sequence"] = input_sequence

        return out

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        input_sequence = self.cache["input_sequence"]
        n = len(input_sequence)
        assert dout.shape == (n, self.d_model)

        dtoken_embedding_matrix = self.cache["dtoken_embedding_matrix"]
        dposition_embedding_matrix = self.cache["dposition_embedding_matrix"]

        # Zero-out static gradient matrices instead of re-allocating on each pass
        dtoken_embedding_matrix[:] = 0
        dposition_embedding_matrix[:] = 0

        for i, row in enumerate(dout):
            dtoken_embedding_matrix[input_sequence[i]] += row

        dposition_embedding_matrix[:n] = dout

        self.cache["dtoken_embedding_matrix"] = dtoken_embedding_matrix
        self.cache["dposition_embedding_matrix"] = dposition_embedding_matrix

    def step(self) -> None:
        """Performs a single optimization step."""
        assert self.enable_grad, "Cannot take an optimization step with enable_grad=False"
        assert self.optimizer, "Cannot take an optimization step with optimizer=None"

        assert self.token_embedding_matrix_opt is not None
        assert self.position_embedding_matrix_opt is not None

        self.token_embedding_matrix_opt.step(self.cache["dtoken_embedding_matrix"])
        self.position_embedding_matrix_opt.step(self.cache["dposition_embedding_matrix"])
