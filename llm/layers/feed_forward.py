"""Implementation of a 2-layer feed-forward (multi-layer perceptron, MLP) layer."""

from typing import Optional

import numpy as np

from llm.constants import DType, DEFAULT_DTYPE, BaseParameter, Parameters
from llm.layers.base import Layer
from llm.layers.linear import Linear
from llm.optimizers import Optimizer
from llm.utils.math import relu


class FeedForward(Layer):
    """A 2-layer feed-forward network with ReLU activation."""

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        dtype: DType = DEFAULT_DTYPE,
        enable_grad: bool = True,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the layer."""
        super().__init__(dtype=dtype, enable_grad=enable_grad, optimizer=optimizer)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.layer_1 = Linear(
            n_input=n_input,
            n_output=n_hidden,
            dtype=dtype,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )
        self.layer_2 = Linear(
            n_input=n_hidden,
            n_output=n_output,
            dtype=dtype,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return self.layer_1.n_params + self.layer_2.n_params

    def get_parameters(self) -> Parameters:
        """Return the parameter map for the layer."""
        params = {
            "layer_1": self.layer_1.get_parameters(),
            "layer_2": self.layer_2.get_parameters(),
        }
        return params

    def load_parameters(self, params: Parameters) -> None:
        """Set the parameters."""
        if "layer_1" not in params or "layer_2" not in params:
            raise ValueError("Missing parameters")
        if isinstance(params["layer_1"], BaseParameter) or isinstance(params["layer_2"], BaseParameter):
            raise ValueError("Invalid shape for parameters map")
        self.layer_1.load_parameters(params["layer_1"])
        self.layer_2.load_parameters(params["layer_2"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.ndim >= 2 and x.shape[-1] == self.n_input

        h = self.layer_1.forward(x)
        a = relu(h)
        out = self.layer_2.forward(a)

        return out

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        assert dout.shape == (*self.layer_1.cache["x"].shape[:-1], self.n_output)

        self.layer_2.backward(dout)
        da = self.layer_2.cache["dx"]
        a = self.layer_2.cache["x"]
        dh = da * (a > 0)
        self.layer_1.backward(dh)
        dx = self.layer_1.cache["dx"]

        self.cache["dx"] = dx

    def step(self) -> None:
        """Perform a single optimization step."""
        assert self.enable_grad, "Cannot take an optimization step with enable_grad=False"
        assert self.optimizer, "Cannot take an optimization step with optimizer=None"

        self.layer_2.step()
        self.layer_1.step()
