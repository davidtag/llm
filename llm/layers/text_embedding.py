"""A learned text embedding layer for a fixed vocab size & context window."""

from typing import Optional

import numpy as np

from llm.constants import DType, DEFAULT_DTYPE, BaseParameter, Parameters
from llm.layers.base import Layer
from llm.optimizers import Optimizer


class TextEmbedding(Layer):
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
        super().__init__(dtype=dtype, enable_grad=enable_grad, optimizer=optimizer)
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model

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

    def get_parameters(self) -> Parameters:
        """Return the parameter map for the layer."""
        params = {
            "token_embedding_matrix": self.token_embedding_matrix,
            "position_embedding_matrix": self.position_embedding_matrix,
        }
        return params

    def load_parameters(self, params: Parameters) -> None:
        """Set the parameters."""
        if "token_embedding_matrix" not in params or "position_embedding_matrix" not in params:
            raise ValueError("Missing parameters")
        if not isinstance(params["token_embedding_matrix"], BaseParameter) or not isinstance(
            params["position_embedding_matrix"], BaseParameter
        ):
            raise ValueError("Invalid shape for parameters map")
        self.token_embedding_matrix[:] = params["token_embedding_matrix"]
        self.position_embedding_matrix[:] = params["position_embedding_matrix"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.ndim == 2
        assert x.min() >= 0 and x.max() < self.vocab_size
        _, T = x.shape  # (B, T)
        assert T <= self.context_size

        token_embeddings = self.token_embedding_matrix[x]  # shape = (B, T, d_model)
        position_embeddings = self.position_embedding_matrix[:T]  # shape = (T, d_model)
        out = token_embeddings + position_embeddings  # shape = (B, T, d_model)

        if self.enable_grad:
            self.cache["x"] = x

        return out

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        x = self.cache["x"]
        B, T = x.shape
        assert dout.shape == (B, T, self.d_model)

        dtoken_embedding_matrix = self.cache["dtoken_embedding_matrix"]
        dposition_embedding_matrix = self.cache["dposition_embedding_matrix"]

        # Zero-out static gradient matrices instead of re-allocating on each pass
        dtoken_embedding_matrix[:] = 0
        dposition_embedding_matrix[:] = 0

        for b in range(B):
            for t in range(T):
                dtoken_embedding_matrix[x[b, t]] += dout[b, t]

        dposition_embedding_matrix[:T] = dout.sum(axis=0)

        self.cache["dtoken_embedding_matrix"] = dtoken_embedding_matrix
        self.cache["dposition_embedding_matrix"] = dposition_embedding_matrix

    def step(self) -> None:
        """Perform a single optimization step."""
        assert self.enable_grad, "Cannot take an optimization step with enable_grad=False"
        assert self.optimizer, "Cannot take an optimization step with optimizer=None"

        assert self.token_embedding_matrix_opt is not None
        assert self.position_embedding_matrix_opt is not None

        self.token_embedding_matrix_opt.step(self.cache["dtoken_embedding_matrix"])
        self.position_embedding_matrix_opt.step(self.cache["dposition_embedding_matrix"])
