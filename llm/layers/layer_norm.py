"""Implementation of Layer Normalization."""

from typing import Optional

import numpy as np

from llm.optimizers import Optimizer


class LayerNorm:
    """Implements a single layer normalization."""

    def __init__(
        self,
        n_input: int,
        enable_grad: bool = True,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the layer."""
        self.n_input = n_input
        self.enable_grad = enable_grad
        self.optimizer = optimizer
        self.cache = {}

        self.gamma = np.random.normal(loc=0, scale=np.sqrt(2 / n_input), size=(1, n_input))
        self.beta = np.random.normal(loc=0, scale=np.sqrt(2 / n_input), size=(1, n_input))

        self.gamma_opt = optimizer.get_parameter_optimizer(self.gamma) if optimizer else None
        self.beta_opt = optimizer.get_parameter_optimizer(self.beta) if optimizer else None

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return self.gamma.size + self.beta.size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.shape[-1] == self.n_input

        ndim = x.ndim

        x_mean = np.mean(x, axis=ndim - 1, keepdims=True)
        x_var = np.mean(np.square(x - x_mean), axis=ndim - 1, keepdims=True)
        x_std = np.sqrt(x_var)
        if self.n_input == 2:
            # Need random noise to avoid z values all being exactly 1 or -1
            # The standard deviation of a set of 2 values is always 1 or -1
            x_std += 5 * np.random.random(size=(*x.shape[:-1], 1))
        z = (x - x_mean) / x_std

        out = z * self.gamma + self.beta

        if self.enable_grad:
            self.cache["x_std"] = x_std
            self.cache["z"] = z

        return out

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        x_std = self.cache["x_std"]
        z = self.cache["z"]
        assert dout.shape == z.shape  # z is the same shape as x

        ndim = z.ndim
        sum_axes = tuple(np.arange(ndim - 1))

        dgamma = np.sum(z * dout, axis=sum_axes, keepdims=True).reshape(1, self.n_input)
        dbeta = np.sum(dout, axis=sum_axes, keepdims=True).reshape(1, self.n_input)
        dx = (
            (
                self.n_input * dout * self.gamma
                - np.matmul(dout, np.transpose(self.gamma))
                - z * np.matmul(dout * z, np.transpose(self.gamma))
            )
            / x_std
            / self.n_input
        )

        self.cache["dx"] = dx
        self.cache["dgamma"] = dgamma
        self.cache["dbeta"] = dbeta

    def step(self) -> None:
        """Performs a single optimization step."""
        assert self.enable_grad, "Cannot take an optimization step with enable_grad=False"
        assert self.optimizer, "Cannot take an optimization step with optimizer=None"

        assert self.gamma_opt is not None
        assert self.beta_opt is not None

        self.gamma_opt.step(self.cache["dgamma"])
        self.beta_opt.step(self.cache["dbeta"])
