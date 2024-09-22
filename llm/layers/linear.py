"""Implementation of a linear layer."""

from typing import Optional

import numpy as np

from llm.constants import DType, DEFAULT_DTYPE
from llm.optimizers import Optimizer


class Linear:
    """Implements a single linear layer with a weight matrix and bias."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        dtype: DType = DEFAULT_DTYPE,
        enable_grad: bool = True,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the layer."""
        self.n_input = n_input
        self.n_output = n_output
        self.dtype = dtype
        self.enable_grad = enable_grad
        self.optimizer = optimizer
        self.cache = {}

        spread = np.sqrt(1 / n_input)
        self.w = np.random.uniform(low=-spread, high=spread, size=(n_input, n_output)).astype(dtype)
        self.b = np.random.uniform(low=-spread, high=spread, size=(1, n_output)).astype(dtype)

        self.w_opt = optimizer.get_parameter_optimizer(self.w) if optimizer else None
        self.b_opt = optimizer.get_parameter_optimizer(self.b) if optimizer else None

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return self.w.size + self.b.size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.ndim >= 2 and x.shape[-1] == self.n_input  # shape = (D_0, ..., D_k, n_input)

        out = np.matmul(x, self.w) + self.b  # shape = (D_0, ..., D_k, n_output)

        if self.enable_grad:
            self.cache["x"] = x

        return out

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        x = self.cache["x"]
        assert dout.shape == (*x.shape[:-1], self.n_output)  # shape = (D_0, ..., D_k, n_output)

        batch_axes = tuple(np.arange(x.ndim - 1))  # (0, 1, .., k)
        non_matmul_axes = batch_axes[:-1]  # (0, 1, .., k-1). Empty for x.ndim == 2.

        dx = np.matmul(dout, np.transpose(self.w))  # shape = (D_0, ..., D_k, n_input)

        x_t = np.transpose(x, axes=(*non_matmul_axes, -1, -2))  # shape = (D_0, ..., D_k-1, n_input, D_k)
        dw = np.matmul(x_t, dout)  # shape = (D_0, ..., D_k-1, n_input, n_output)
        dw = np.sum(dw, axis=non_matmul_axes)  # sum along other dims since output is parallel in them

        db = np.sum(dout, axis=batch_axes, keepdims=True).reshape(1, -1)

        self.cache["dx"] = dx
        self.cache["dw"] = dw
        self.cache["db"] = db

    def step(self) -> None:
        """Performs a single optimization step."""
        assert self.enable_grad, "Cannot take an optimization step with enable_grad=False"
        assert self.optimizer, "Cannot take an optimization step with optimizer=None"

        assert self.w_opt is not None
        assert self.b_opt is not None

        self.w_opt.step(self.cache["dw"])
        self.b_opt.step(self.cache["db"])
