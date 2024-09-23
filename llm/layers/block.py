"""Implementation of a Transformer block layer."""

from typing import Optional

import numpy as np

from llm.constants import DType, DEFAULT_DTYPE
from llm.layers.feed_forward import FeedForward
from llm.layers.layer_norm import LayerNorm
from llm.layers.multi_head_attention import MultiHeadAttention
from llm.optimizers import Optimizer


class Block:
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
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff
        self.masked_attention = masked_attention
        self.dtype = dtype
        self.enable_grad = enable_grad
        self.optimizer = optimizer
        self.cache = {}

        self.sublayer_1 = MultiHeadAttention(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            h=h,
            masked=masked_attention,
            dtype=dtype,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )
        self.norm_1 = LayerNorm(n_input=d_model, dtype=dtype, enable_grad=enable_grad, optimizer=optimizer)
        self.sublayer_2 = FeedForward(
            n_input=d_model,
            n_hidden=d_ff,
            n_output=d_model,
            dtype=dtype,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )
        self.norm_2 = LayerNorm(n_input=d_model, dtype=dtype, enable_grad=enable_grad, optimizer=optimizer)

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return (
            self.sublayer_1.n_params + self.norm_1.n_params + self.sublayer_2.n_params + self.norm_2.n_params
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.ndim == 2 and x.shape[-1] == self.d_model

        h1 = self.sublayer_1.forward(x)
        a1 = x + h1
        o1 = self.norm_1.forward(a1)

        h2 = self.sublayer_2.forward(o1)
        a2 = o1 + h2
        o2 = self.norm_2.forward(a2)

        return o2

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"

        do2 = dout
        self.norm_2.backward(do2)
        da2 = self.norm_2.cache["dx"]
        dh2 = da2
        self.sublayer_2.backward(dh2)

        do1 = da2 + self.sublayer_2.cache["dx"]
        self.norm_1.backward(do1)
        da1 = self.norm_1.cache["dx"]
        dh1 = da1
        self.sublayer_1.backward(dh1)

        dx = da1 + self.sublayer_1.cache["dx"]

        self.cache["dx"] = dx

    def step(self) -> None:
        """Performs a single optimization step."""
        assert self.enable_grad, "Cannot take an optimization step with enable_grad=False"
        assert self.optimizer, "Cannot take an optimization step with optimizer=None"

        self.norm_2.step()
        self.sublayer_2.step()
        self.norm_1.step()
        self.sublayer_1.step()
