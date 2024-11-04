"""Implementation of a stack of identical Transformer block layers."""

from typing import Optional

import numpy as np

from llm.constants import DType, DEFAULT_DTYPE, BaseParameter, Parameters
from llm.layers.base import Layer
from llm.layers.block import Block
from llm.optimizers import Optimizer


class BlockStack(Layer):
    """A stack of block in a Transformer architecture."""

    def __init__(
        self,
        n_blocks: int,
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
        """Initialize the block stack."""
        super().__init__(dtype=dtype, enable_grad=enable_grad, optimizer=optimizer)
        assert n_blocks >= 1
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff

        self.blocks = [
            Block(
                d_model=d_model,
                d_k=d_k,
                d_v=d_v,
                h=h,
                d_ff=d_ff,
                masked_attention=masked_attention,
                dtype=dtype,
                enable_grad=enable_grad,
                optimizer=optimizer,
            )
            for _ in range(n_blocks)
        ]

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return sum(block.n_params for block in self.blocks)

    def get_parameters(self) -> Parameters:
        """Return the parameter map for the layer."""
        params = {f"block_{i}": self.blocks[i].get_parameters() for i in range(self.n_blocks)}
        return params

    def load_parameters(self, params: Parameters) -> None:
        """Set the parameters."""
        for i in range(self.n_blocks):
            key = f"block_{i}"
            if key not in params:
                raise ValueError("Missing parameters")
            params_i = params[key]
            if isinstance(params_i, BaseParameter):
                raise ValueError("Invalid shape for parameters map")
            self.blocks[i].load_parameters(params_i)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.ndim >= 2 and x.shape[-1] == self.d_model

        out = x

        for block in self.blocks:
            out = block.forward(out)

        return out

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"

        cur_dout = dout

        for block in reversed(self.blocks):
            block.backward(cur_dout)
            cur_dout = block.cache["dx"]

        self.cache["dx"] = cur_dout

    def step(self) -> None:
        """Perform a single optimization step."""
        assert self.enable_grad, "Cannot take an optimization step with enable_grad=False"
        assert self.optimizer, "Cannot take an optimization step with optimizer=None"

        for block in self.blocks:
            block.step()
