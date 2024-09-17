"""Implementation of Layer Normalization."""

from typing import Optional

import numpy as np

from llm.constants import DType, DEFAULT_DTYPE, BaseParameter, Parameters
from llm.layers.base import Layer
from llm.optimizers import Optimizer


class LayerNorm(Layer):
    """Implements a single layer normalization."""

    def __init__(
        self,
        n_input: int,
        eps: float = 1e-5,
        dtype: DType = DEFAULT_DTYPE,
        enable_grad: bool = True,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the layer."""
        super().__init__(dtype=dtype, enable_grad=enable_grad, optimizer=optimizer)
        self.n_input = n_input
        self.eps = eps

        self.gamma = np.ones(shape=(1, n_input), dtype=dtype)
        self.beta = np.zeros(shape=(1, n_input), dtype=dtype)

        self.gamma_opt = optimizer.get_parameter_optimizer(self.gamma) if optimizer else None
        self.beta_opt = optimizer.get_parameter_optimizer(self.beta) if optimizer else None

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return self.gamma.size + self.beta.size

    def get_parameters(self) -> Parameters:
        """Return the parameter map for the layer."""
        params = {
            "gamma": self.gamma,
            "beta": self.beta,
        }
        return params

    def load_parameters(self, params: Parameters) -> None:
        """Set the parameters."""
        if "gamma" not in params or "beta" not in params:
            raise ValueError("Missing parameters")
        if not isinstance(params["gamma"], BaseParameter) or not isinstance(params["beta"], BaseParameter):
            raise ValueError("Invalid shape for parameters map")
        self.gamma[:] = params["gamma"]
        self.beta[:] = params["beta"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.shape[-1] == self.n_input

        x_mean = np.mean(x, axis=-1, keepdims=True)
        x_var = np.mean(np.square(x - x_mean), axis=-1, keepdims=True)
        x_std = np.sqrt(x_var + self.eps)
        if self.n_input == 2:
            # HACK: Need random noise to avoid z values all being exactly 1 or -1
            # The z-scores of a set of 2 values is always 1 or -1.
            # In practice, embedding dimensions are >> 2, so this problem doesn't arise.
            x_std += 5 * np.random.random(size=x_std.shape)
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
        batch_axes = tuple(np.arange(ndim - 1))

        dbeta = np.sum(dout, axis=batch_axes, keepdims=True).reshape(1, self.n_input)
        dgamma = np.sum(dout * z, axis=batch_axes, keepdims=True).reshape(1, self.n_input)
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
        """Perform a single optimization step."""
        assert self.enable_grad, "Cannot take an optimization step with enable_grad=False"
        assert self.optimizer, "Cannot take an optimization step with optimizer=None"

        assert self.gamma_opt is not None
        assert self.beta_opt is not None

        self.gamma_opt.step(self.cache["dgamma"])
        self.beta_opt.step(self.cache["dbeta"])
