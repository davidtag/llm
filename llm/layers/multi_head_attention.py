"""Implements the multi-head attention layer of a Transformer."""

from typing import Optional

import numpy as np

from llm.constants import DType, DEFAULT_DTYPE, BaseParameter, Parameters
from llm.layers.base import Layer
from llm.optimizers import Optimizer
from llm.utils.math import softmax


class MultiHeadAttention(Layer):
    """A single multi-head attention sub-layer of a Transformer block."""

    def __init__(
        self,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        h: int = 8,
        masked: bool = False,
        dtype: DType = DEFAULT_DTYPE,
        enable_grad: bool = True,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the layer."""
        super().__init__(dtype=dtype, enable_grad=enable_grad, optimizer=optimizer)
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.masked = masked

        spread_in = np.sqrt(1 / d_model)
        self.w_q = np.random.uniform(low=-spread_in, high=spread_in, size=(h, d_model, d_k)).astype(dtype)
        self.w_k = np.random.uniform(low=-spread_in, high=spread_in, size=(h, d_model, d_k)).astype(dtype)
        self.w_v = np.random.uniform(low=-spread_in, high=spread_in, size=(h, d_model, d_v)).astype(dtype)
        spread_out = np.sqrt(1 / (h * d_v))
        self.w_o = np.random.uniform(low=-spread_out, high=spread_out, size=(h * d_v, d_model)).astype(dtype)

        self.w_q_opt = optimizer.get_parameter_optimizer(self.w_q) if optimizer else None
        self.w_k_opt = optimizer.get_parameter_optimizer(self.w_k) if optimizer else None
        self.w_v_opt = optimizer.get_parameter_optimizer(self.w_v) if optimizer else None
        self.w_o_opt = optimizer.get_parameter_optimizer(self.w_o) if optimizer else None

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return self.w_q.size + self.w_k.size + self.w_v.size + self.w_o.size

    def get_parameters(self) -> Parameters:
        """Return the parameter map for the layer."""
        params = {
            "w_q": self.w_q,
            "w_k": self.w_k,
            "w_v": self.w_v,
            "w_o": self.w_o,
        }
        return params

    def load_parameters(self, params: Parameters) -> None:
        """Set the parameters."""
        if "w_q" not in params or "w_k" not in params or "w_v" not in params or "w_o" not in params:
            raise ValueError("Missing parameters")
        if (
            not isinstance(params["w_q"], BaseParameter)
            or not isinstance(params["w_k"], BaseParameter)
            or not isinstance(params["w_v"], BaseParameter)
            or not isinstance(params["w_o"], BaseParameter)
        ):
            raise ValueError("Invalid shape for parameters map")
        self.w_q[:] = params["w_q"]
        self.w_k[:] = params["w_k"]
        self.w_v[:] = params["w_v"]
        self.w_o[:] = params["w_o"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.ndim >= 2 and x.shape[-1] == self.d_model  # shape = (*B, T, d_model)

        T = x.shape[-2]
        batch_dims = x.shape[:-2]  # (*B). May be empty
        batch_axes = tuple(np.arange(len(batch_dims)))  # (0, 1, ..., num_batch_dims - 1). May be empty.
        head_axis = -3

        # Add a dimension to x for easier broadcasting across the h heads
        x_expanded = np.expand_dims(x, axis=head_axis)  # shape = (*B, 1, T, d_model)

        # Project into the query, key, and value space
        Q = np.matmul(x_expanded, self.w_q)  # shape = (*B, h, T, d_k)
        K = np.matmul(x_expanded, self.w_k)  # shape = (*B, h, T, d_k)
        V = np.matmul(x_expanded, self.w_v)  # shape = (*B, h, T, d_v)

        # Compute the attention weights
        scale = 1 / np.sqrt(self.d_k)
        logits = scale * np.matmul(  # shape = (*B, h, T, T)
            Q,
            np.transpose(K, axes=(*batch_axes, head_axis, -1, -2)),  # only transpose last 2 dims
        )
        if self.masked:
            # The output of the attention head is a weighted average of the value vectors
            # i.e., for each head h, the rows of the matrix V[h, :, :].
            # Fix a head h, position (row) i, and entry (column) j:
            #    head[h,i,j] = weighted_average(V[h, {1,2,...n}, j])
            #                = weighted_average(V[h, {1,2,...i}, j]) to preserve causality
            # This means the weights associated with entries j > i need to be 0. This correspons
            # to a mask where all upper-triangular entries are 0. Moreover, to set the weight to
            # 0, we set the logits to -inf.
            row_indices, column_indices = np.triu_indices(T, k=1)  # k=1 because diagonal isn't masked
            logits[..., row_indices, column_indices] = -np.inf
        weights = softmax(logits)  # shape = (*B, h, T, T)

        # Compute the attention heads in parallel
        head = np.matmul(weights, V)  # shape = (*B, h, T, d_v)

        # Concatenate the heads
        heads = np.split(  # List[(*B, 1, T, d_v)] of length h
            head,
            indices_or_sections=self.h,
            axis=head_axis,
        )
        heads_squeezed = [  # List[(*B, T, d_v)] length h
            head.reshape(*batch_dims, T, self.d_v) for head in heads
        ]
        concat = np.concatenate(heads_squeezed, axis=-1)  # shape = (*B, T, h * d_v). Will copy.

        # Project back into the model dimension
        out = np.matmul(concat, self.w_o)  # size = (*B, T, d_model)

        if self.enable_grad:
            self.cache["x"] = x
            self.cache["Q"] = Q
            self.cache["K"] = K
            self.cache["V"] = V
            self.cache["weights"] = weights
            self.cache["concat"] = concat

        return out

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        x = self.cache["x"]  # shape = (*B, T, d_model)
        Q = self.cache["Q"]  # shape = (*B, h, T, d_k)
        K = self.cache["K"]  # shape = (*B, h, T, d_k)
        V = self.cache["V"]  # shape = (*B, h, T, d_v)
        weights = self.cache["weights"]  # shape = (*B, h, T, T)
        concat = self.cache["concat"]  # shape = (*B, T, h * d_v)
        assert dout.shape == x.shape  # shape = (*B, T, d_model)

        batch_dims = x.shape[:-2]  # (*B). May be empty
        batch_axes = tuple(np.arange(len(batch_dims)))  # (0, 1, ..., num_batch_dims - 1). May be empty.
        head_axis = -3

        # Add a dimension to x for easier broadcasting across the h heads
        x_expanded = np.expand_dims(x, axis=head_axis)  # shape = (*B, 1, T, d_model)

        # Propagate through the projection layer
        dw_o = np.matmul(  # shape = (h * d_v, d_model)
            np.transpose(concat, axes=(*batch_axes, -1, -2)),  # only transpose last 2 dims
            dout,
        ).sum(batch_axes)
        dconcat = np.matmul(dout, np.transpose(self.w_o))  # shape = (*B, T, h * d_v)

        # De-concatenate the gradients. Will lead to a copy.
        dheads_squeezed = np.split(  # List[(*B, T, d_v)] of length h
            dconcat,
            indices_or_sections=self.h,
            axis=-1,
        )
        dhead = np.stack(dheads_squeezed, axis=head_axis)  # shape = (*B, h, T, d_v)

        # Propagate through the attention head
        dV = np.matmul(  # shape = (*B, h, T, d_v)
            np.transpose(weights, axes=(*batch_axes, head_axis, -1, -2)),  # only transpose last 2 dims
            dhead,
        )
        dweights = np.matmul(  # shape = (*B, h, T, T)
            dhead,
            np.transpose(V, axes=(*batch_axes, head_axis, -1, -2)),  # only transpose last 2 dims
        )

        # Propagate through the attention weights
        # NOTE: because dlogits is calculated by multiplying by weights and the effect of the mask is to zero
        # out the weights, the effect of the mask is implicitly captured when computing dlogits; i.e., the
        # mask zero's out the effect of the upper-triangular logits, so the derivative of the loss with
        # respect to these entries should be 0, and the formula below does just that. This is why there's
        # no special-case handling for self.masked here.
        dlogits = weights * (  # shape = (*B, h, T, T)
            dweights - np.sum(dweights * weights, axis=-1, keepdims=True)
        )
        scale = 1 / np.sqrt(self.d_k)
        dQ = scale * np.matmul(  # shape = (*B, h, T, d_k)
            dlogits,
            K,
        )
        dK = scale * np.matmul(  # shape = (*B, h, T, d_k)
            np.transpose(dlogits, axes=(*batch_axes, head_axis, -1, -2)),  # only transpose last 2 dims
            Q,
        )

        # Propagate through the the query, key, and value space projections
        x_expanded_transpose = np.transpose(x_expanded, axes=(*batch_axes, head_axis, -1, -2))
        dw_q = np.matmul(x_expanded_transpose, dQ).sum(axis=batch_axes)  # shape = (h, d_model, d_k)
        dw_k = np.matmul(x_expanded_transpose, dK).sum(axis=batch_axes)  # shape = (h, d_model, d_k)
        dw_v = np.matmul(x_expanded_transpose, dV).sum(axis=batch_axes)  # shape = (h, d_model, d_v)
        dx_from_Q = np.matmul(dQ, np.transpose(self.w_q, axes=(0, 2, 1)))  # shape = (*B, h, T, d_model)
        dx_from_K = np.matmul(dK, np.transpose(self.w_k, axes=(0, 2, 1)))  # shape = (*B, h, T, d_model)
        dx_from_V = np.matmul(dV, np.transpose(self.w_v, axes=(0, 2, 1)))  # shape = (*B, h, T, d_model)
        dx = (
            # shape = (*B, T, d_model)
            dx_from_Q.sum(axis=head_axis)
            + dx_from_K.sum(axis=head_axis)
            + dx_from_V.sum(axis=head_axis)
        )

        self.cache["dx"] = dx
        self.cache["dw_q"] = dw_q
        self.cache["dw_k"] = dw_k
        self.cache["dw_v"] = dw_v
        self.cache["dw_o"] = dw_o

    def step(self) -> None:
        """Perform a single optimization step."""
        assert self.enable_grad, "Cannot take an optimization step with enable_grad=False"
        assert self.optimizer, "Cannot take an optimization step with optimizer=None"

        assert self.w_q_opt is not None
        assert self.w_k_opt is not None
        assert self.w_v_opt is not None
        assert self.w_o_opt is not None

        self.w_q_opt.step(self.cache["dw_q"])
        self.w_k_opt.step(self.cache["dw_k"])
        self.w_v_opt.step(self.cache["dw_v"])
        self.w_o_opt.step(self.cache["dw_o"])
