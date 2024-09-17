"""Implementation for a linear layer."""

from typing import Optional

import numpy as np

from llm.optimizers import Optimizer


class Linear(object):
    """Implements a single linear layer with a weight matrix and bias."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        enable_grad: bool = True,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the layer."""
        self.n_input = n_input
        self.n_output = n_output
        self.enable_grad = enable_grad
        self.optimizer = optimizer
        self.cache = {}

        self.w = np.random.normal(loc=0, scale=np.sqrt(2 / n_input), size=(n_input, n_output))
        self.b = np.random.normal(loc=0, scale=np.sqrt(2 / n_input), size=(1, n_output))

        self.w_opt = optimizer.get_parameter_optimizer(self.w) if optimizer else None
        self.b_opt = optimizer.get_parameter_optimizer(self.b) if optimizer else None

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return self.w.size + self.b.size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.ndim == 2 and x.shape[-1] == self.n_input

        out = np.matmul(x, self.w) + self.b

        if self.enable_grad:
            self.cache["x"] = x

        return out

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        x = self.cache["x"]
        assert dout.shape == (x.shape[0], self.n_output)

        dx = np.matmul(dout, np.transpose(self.w))
        dw = np.matmul(np.transpose(x), dout)
        db = np.sum(dout, axis=0, keepdims=True)

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
