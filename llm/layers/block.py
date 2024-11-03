"""Implementation of a Transformer block layer."""

from typing import Optional

import numpy as np

from llm.constants import DType, DEFAULT_DTYPE, BaseParameter, Parameters
from llm.layers.base import Layer
from llm.layers.feed_forward import FeedForward
from llm.layers.layer_norm import LayerNorm
from llm.layers.multi_head_attention import MultiHeadAttention
from llm.optimizers import Optimizer


class Block(Layer):
    """A single block in a Transformer architecture."""

    def __init__(
        self,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        h: int = 8,
        d_ff: int = 2048,
        masked_attention: bool = False,
        dtype: DType = DEFAULT_DTYPE,
        enable_grad: bool = True,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the block."""
        super().__init__(dtype=dtype, enable_grad=enable_grad, optimizer=optimizer)
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff
        self.masked_attention = masked_attention

        self.norm_1 = LayerNorm(n_input=d_model, dtype=dtype, enable_grad=enable_grad, optimizer=optimizer)
        self.attention = MultiHeadAttention(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            h=h,
            masked=masked_attention,
            dtype=dtype,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )
        self.norm_2 = LayerNorm(n_input=d_model, dtype=dtype, enable_grad=enable_grad, optimizer=optimizer)
        self.ffn = FeedForward(
            n_input=d_model,
            n_hidden=d_ff,
            n_output=d_model,
            dtype=dtype,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return self.norm_1.n_params + self.attention.n_params + self.norm_2.n_params + self.ffn.n_params

    def get_parameters(self) -> Parameters:
        """Return the parameter map for the layer."""
        params = {
            "norm_1": self.norm_1.get_parameters(),
            "attention": self.attention.get_parameters(),
            "norm_2": self.norm_2.get_parameters(),
            "ffn": self.ffn.get_parameters(),
        }
        return params

    def load_parameters(self, params: Parameters) -> None:
        """Set the parameters."""
        if (
            "norm_1" not in params
            or "attention" not in params
            or "norm_2" not in params
            or "ffn" not in params
        ):
            raise ValueError("Missing parameters")
        if (
            isinstance(params["norm_1"], BaseParameter)
            or isinstance(params["attention"], BaseParameter)
            or isinstance(params["norm_2"], BaseParameter)
            or isinstance(params["ffn"], BaseParameter)
        ):
            raise ValueError("Invalid shape for parameters map")
        self.norm_1.load_parameters(params["norm_1"])
        self.attention.load_parameters(params["attention"])
        self.norm_2.load_parameters(params["norm_2"])
        self.ffn.load_parameters(params["ffn"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.ndim >= 2 and x.shape[-1] == self.d_model

        n1 = self.norm_1.forward(x)
        a1 = self.attention.forward(n1)
        o1 = x + a1

        n2 = self.norm_2.forward(o1)
        a2 = self.ffn.forward(n2)
        o2 = o1 + a2

        return o2

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"

        do2 = dout

        da2 = do2
        self.ffn.backward(da2)
        dn2 = self.ffn.cache["dx"]
        self.norm_2.backward(dn2)
        do1_residual = self.norm_2.cache["dx"]

        do1 = do2 + do1_residual

        da1 = do1
        self.attention.backward(da1)
        dn1 = self.attention.cache["dx"]
        self.norm_1.backward(dn1)
        dx_residual = self.norm_1.cache["dx"]

        dx = do1 + dx_residual

        self.cache["dx"] = dx

    def step(self) -> None:
        """Perform a single optimization step."""
        assert self.enable_grad, "Cannot take an optimization step with enable_grad=False"
        assert self.optimizer, "Cannot take an optimization step with optimizer=None"

        self.norm_2.step()
        self.ffn.step()
        self.norm_1.step()
        self.attention.step()
