# pylint: disable=invalid-name,too-many-locals
"""Implements the multi-head attention layer of a Transformer."""

from typing import Optional

import numpy as np

from llm.optimizers import Optimizer
from llm.utils.math import softmax


class MultiHeadAttention:
    """A single multi-head attention sub-layer of a Transformer block."""

    def __init__(
        self,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        h: int = 8,
        masked: bool = False,
        enable_grad: bool = True,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the layer."""
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.masked = masked
        self.enable_grad = enable_grad
        self.optimizer = optimizer
        self.cache = {}

        # See here. Gain of 2 is for relu activation but here we have sigmoid for some, linear for others
        # https://pytorch.org/docs/stable/nn.init.html
        self.w_q = np.random.normal(loc=0, scale=np.sqrt(2 / d_model), size=(h, d_model, d_k))
        self.w_k = np.random.normal(loc=0, scale=np.sqrt(2 / d_model), size=(h, d_model, d_k))
        self.w_v = np.random.normal(loc=0, scale=np.sqrt(2 / d_model), size=(h, d_model, d_v))
        self.w_o = np.random.normal(loc=0, scale=np.sqrt(2 / d_model), size=(h * d_v, d_model))

        self.w_q_opt = optimizer.get_parameter_optimizer(self.w_q) if optimizer else None
        self.w_k_opt = optimizer.get_parameter_optimizer(self.w_k) if optimizer else None
        self.w_v_opt = optimizer.get_parameter_optimizer(self.w_v) if optimizer else None
        self.w_o_opt = optimizer.get_parameter_optimizer(self.w_o) if optimizer else None

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return self.w_q.size + self.w_k.size + self.w_v.size + self.w_o.size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        # TODO(dtag): Support x.shape = (B, T, d_model)
        # To x_reshape = x[:, np.newaxis, :, :] or x.reshape(B, 1, -T, d_model) to allow for the matmul
        # below to work
        assert x.ndim == 2 and x.shape[-1] == self.d_model

        N = x.shape[0]

        Q = np.matmul(x, self.w_q)  # shape = (h, N, d_k)
        K = np.matmul(x, self.w_k)  # shape = (h, N, d_k)
        V = np.matmul(x, self.w_v)  # shape = (h, N, d_v)

        scale = 1 / np.sqrt(self.d_k)
        logits = scale * np.matmul(Q, np.transpose(K, axes=[0, 2, 1]))  # shape = (h, N, N)
        if self.masked:
            # The output of the attention head is a weighted average of the value vectors
            # i.e., for each head h, the rows of the matrix V[h, :, :].
            # Fix a head h, position (row) i, and entry (column) j:
            #    head[h,i,j] = weighted_average(V[h, {1,2,...n}, j])
            #                = weighted_average(V[h, {1,2,...i}, j]) to preserve causality
            # This means the weights associated with entries j > i need to be 0. This correspons
            # to a mask where all upper-triangular entries are 0. Moreover, to set the weight to
            # 0, we set the logits to -inf.
            row_indices, column_indices = np.triu_indices(N, k=1)  # k=1 because diagonal isn't masked
            logits[:, row_indices, column_indices] = -np.inf
        weights = softmax(logits)  # shape = (h, N, N)
        head = np.matmul(weights, V)  # shape = (h, N, d_v)

        heads = np.split(head, indices_or_sections=self.h, axis=0)  # List[(1, N, d_v)] of length h
        heads_squeezed = [head.reshape(N, self.d_v) for head in heads]  # List[(N, d_v)] of length h
        concat = np.hstack(heads_squeezed)  # shape = (N, h * d_v)

        out = np.matmul(concat, self.w_o)  # size = (N, d_model)

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
        x = self.cache["x"]  # shape = (N, d_model)
        Q = self.cache["Q"]  # shape = (h, N, d_k)
        K = self.cache["K"]  # shape = (h, N, d_k)
        V = self.cache["V"]  # shape = (h, N, d_v)
        weights = self.cache["weights"]  # shape = (h, N, N)
        concat = self.cache["concat"]  # shape = (N, h * d_v)
        assert dout.shape == x.shape  # shape = (N, d_model)

        dw_o = np.matmul(np.transpose(concat), dout)  # shape = (h * d_v, d_model)
        dconcat = np.matmul(dout, np.transpose(self.w_o))  # shape = (N, h * d_v)

        dheads_squeezed = np.hsplit(dconcat, indices_or_sections=self.h)  # List[(N, d_v)] of length h
        dhead = np.stack(dheads_squeezed, axis=0)  # shape = (h, N, d_v)

        dV = np.matmul(np.transpose(weights, axes=[0, 2, 1]), dhead)  # shape = (h, N, d_v)
        dweights = np.matmul(dhead, np.transpose(V, axes=[0, 2, 1]))  # shape = (h, N, N)

        # NOTE: because dlogits is calculated by multiplying by weights and the effect of the mask is to zero
        # out the weights, the effect of the mask is implicitly captured when computing dlogits; i.e., the
        # mask zero's out the effect of the upper-triangular logits, so the derivative of the loss with
        # respect to these entries should be 0, and the formula below does just that. This is why there's
        # no special-case handling for self.masked here.
        dlogits = weights * (
            dweights - np.sum(dweights * weights, axis=2, keepdims=True)
        )  # shape = (h, N, N)

        scale = 1 / np.sqrt(self.d_k)
        dQ = scale * np.matmul(dlogits, K)  # shape = (h, N, d_k)
        dK = scale * np.matmul(np.transpose(dlogits, axes=[0, 2, 1]), Q)  # shape = (h, N, d_k)

        dw_q = np.matmul(np.transpose(x), dQ)  # shape = (h, d_model, d_k)
        dw_k = np.matmul(np.transpose(x), dK)  # shape = (h, d_model, d_k)
        dw_v = np.matmul(np.transpose(x), dV)  # shape = (h, d_model, d_v)

        dx_from_Q = np.matmul(dQ, np.transpose(self.w_q, axes=[0, 2, 1]))  # shape = (h, N, d_model)
        dx_from_K = np.matmul(dK, np.transpose(self.w_k, axes=[0, 2, 1]))  # shape = (h, N, d_model)
        dx_from_V = np.matmul(dV, np.transpose(self.w_v, axes=[0, 2, 1]))  # shape = (h, N, d_model)

        dx = dx_from_Q.sum(axis=0) + dx_from_K.sum(axis=0) + dx_from_V.sum(axis=0)  # shape = (N, d_model)

        self.cache["dx"] = dx
        self.cache["dw_q"] = dw_q
        self.cache["dw_k"] = dw_k
        self.cache["dw_v"] = dw_v
        self.cache["dw_o"] = dw_o

    def step(self) -> None:
        """Performs a single optimization step."""
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
