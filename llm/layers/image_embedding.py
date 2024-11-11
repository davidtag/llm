"""A learned image embedding layer."""

from typing import Optional

import numpy as np

from llm.constants import DType, DEFAULT_DTYPE, BaseParameter, Parameters
from llm.layers.base import Layer
from llm.optimizers import Optimizer


class ImageEmbedding(Layer):
    """A learned image embedding layer."""

    def __init__(
        self,
        patch_size: int = 16,
        canonical_width: int = 224,
        canonical_height: int = 224,
        n_channel: int = 3,
        d_model: int = 512,
        dtype: DType = DEFAULT_DTYPE,
        enable_grad: bool = True,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the layer."""
        super().__init__(dtype=dtype, enable_grad=enable_grad, optimizer=optimizer)
        self.patch_size = patch_size  # P
        self.canonical_width = canonical_width  # W
        self.canonical_height = canonical_height  # H
        self.n_channel = n_channel  # C
        self.d_model = d_model

        if canonical_width % patch_size != 0:
            raise ValueError("Width must be a multiple of patch size")

        if canonical_height % patch_size != 0:
            raise ValueError("Height must be a multiple of patch size")

        self.d_patch = (patch_size**2) * n_channel  # (P^2)*C
        n_patches = (canonical_height * canonical_width) // (patch_size**2)  # N
        self.context_size = n_patches + 1  # N + 1: [class token, *patch_tokens]

        self.class_token = np.random.standard_normal(size=(1, d_model)).astype(dtype)
        spread = np.sqrt(1 / self.d_patch)
        self.patch_proj = np.random.uniform(
            low=-spread,
            high=spread,
            size=(self.d_patch, self.d_model),
        ).astype(dtype)
        self.position_embedding_matrix = np.random.standard_normal(
            size=(self.context_size, self.d_model)
        ).astype(dtype)

        self.class_token_opt = optimizer.get_parameter_optimizer(self.class_token) if optimizer else None
        self.patch_proj_opt = optimizer.get_parameter_optimizer(self.patch_proj) if optimizer else None
        self.position_embedding_matrix_opt = (
            optimizer.get_parameter_optimizer(self.position_embedding_matrix) if optimizer else None
        )

        # Pre-allocate the gradient to save compute on each backward pass and avoid large memory allocations
        self.cache["dposition_embedding_matrix"] = np.zeros_like(self.position_embedding_matrix)

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return self.class_token.size + self.patch_proj.size + self.position_embedding_matrix.size

    def get_parameters(self) -> Parameters:
        """Return the parameter map for the layer."""
        params = {
            "class_token": self.class_token,
            "patch_proj": self.patch_proj,
            "position_embedding_matrix": self.position_embedding_matrix,
        }
        return params

    def load_parameters(self, params: Parameters) -> None:
        """Set the parameters."""
        if (
            "class_token" not in params
            or "patch_proj" not in params
            or "position_embedding_matrix" not in params
        ):
            raise ValueError("Missing parameters")
        if (
            not isinstance(params["class_token"], BaseParameter)
            or not isinstance(params["patch_proj"], BaseParameter)
            or not isinstance(params["position_embedding_matrix"], BaseParameter)
        ):
            raise ValueError("Invalid shape for parameters map")
        self.class_token[:] = params["class_token"]
        self.patch_proj[:] = params["patch_proj"]
        self.position_embedding_matrix[:] = params["position_embedding_matrix"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.ndim == 4  # (B, H, W, C)
        B, H, W, C = x.shape
        assert H % self.patch_size == 0
        assert W % self.patch_size == 0
        assert C == self.n_channel

        N = H * W // (self.patch_size**2)
        assert N == self.context_size - 1  # TODO(dtag): Support variable size inputs using 2D proj

        # Patch Embeddings: Flatten + Project
        patches = x.reshape(B, N, self.d_patch)  # (B, N, d_patch)
        patch_embeddings = np.matmul(patches, self.patch_proj)  # (B, N, d_model)

        # Class Token Embedding: Broadcast
        class_token_embeddings = np.broadcast_to(self.class_token, (B, 1, self.d_model))  # (B, 1, d_model)

        # Token Embeddings: Prepend class token to patch tokens
        token_embeddings = np.concatenate(  # (B, N+1, d_model)
            (class_token_embeddings, patch_embeddings),
            axis=1,
        )

        # Position Embeddings: Lookup
        position_embeddings = self.position_embedding_matrix[: N + 1]  # (N + 1, d_model)

        # Final Embeddings: Token + Position
        out = token_embeddings + position_embeddings  # (B, N+1, d_model)

        if self.enable_grad:
            self.cache["x"] = x

        return out

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        x = self.cache["x"]
        B, H, W, C = x.shape
        N = H * W // (self.patch_size**2)
        assert dout.shape == (B, N + 1, self.d_model)

        # Zero-out static gradient matrix instead of re-allocating on each pass
        dposition_embedding_matrix = self.cache["dposition_embedding_matrix"]
        dposition_embedding_matrix[:] = 0
        dposition_embedding_matrix[: N + 1] = dout.sum(axis=0)
        self.cache["dposition_embedding_matrix"] = dposition_embedding_matrix

        dtoken_embeddings = dout
        dclass_token_embeddings = dtoken_embeddings[:, 0, :]  # (B, 1, d_model)
        dpatch_embeddings = dtoken_embeddings[:, 1:, :]  # (B, N, d_model)

        dclass_token = dclass_token_embeddings.sum(axis=0, keepdims=True)
        self.cache["dclass_token"] = dclass_token

        patches = x.reshape(B, N, self.d_patch)  # (B, N, d_patch)
        patches_t = np.transpose(patches, axes=(0, -1, -2))  # (B, d_patch, N)
        dpatch_proj = np.matmul(patches_t, dpatch_embeddings)  # (B, d_patch, d_model)
        dpatch_proj = np.sum(dpatch_proj, axis=0)
        self.cache["dpatch_proj"] = dpatch_proj

    def step(self) -> None:
        """Perform a single optimization step."""
        assert self.enable_grad, "Cannot take an optimization step with enable_grad=False"
        assert self.optimizer, "Cannot take an optimization step with optimizer=None"

        assert self.class_token_opt is not None
        assert self.patch_proj_opt is not None
        assert self.position_embedding_matrix_opt is not None

        self.class_token_opt.step(self.cache["dclass_token"])
        self.patch_proj_opt.step(self.cache["dpatch_proj"])
        self.position_embedding_matrix_opt.step(self.cache["dposition_embedding_matrix"])
