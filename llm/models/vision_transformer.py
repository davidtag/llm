"""Implementation of a VisionTransformer (ViT) model architecture."""

from __future__ import annotations

from pathlib import Path
import pickle
from os import PathLike
from typing import Optional

import numpy as np

from llm.constants import DType, DEFAULT_DTYPE, BaseParameter, Parameters
from llm.layers.block_stack import BlockStack
from llm.layers.linear import Linear
from llm.layers.layer_norm import LayerNorm
from llm.layers.image_embedding import ImageEmbedding
from llm.models.base import Model
from llm.optimizers import Optimizer


class VisionTransformer(Model):
    """A VisionTransformer (ViT) architecture for image processing."""

    def __init__(
        self,
        n_classes: int,
        patch_size: int = 16,
        canonical_width: int = 224,
        canonical_height: int = 224,
        n_channel: int = 3,
        n_blocks: int = 6,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        h: int = 8,
        d_ff: int = 2048,
        dtype: DType = DEFAULT_DTYPE,
        enable_grad: bool = True,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__(dtype=dtype, enable_grad=enable_grad, optimizer=optimizer)
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.canonical_width = canonical_width
        self.canonical_height = canonical_height
        self.n_channel = n_channel
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff

        self.embedding_layer = ImageEmbedding(
            patch_size=patch_size,
            canonical_width=canonical_width,
            canonical_height=canonical_height,
            n_channel=n_channel,
            d_model=d_model,
            dtype=dtype,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )
        self.initial_norm = LayerNorm(
            n_input=d_model,
            dtype=dtype,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )
        self.encoder = BlockStack(
            n_blocks=n_blocks,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            h=h,
            d_ff=d_ff,
            masked_attention=False,  # encoder instead of decoder
            dtype=dtype,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )
        self.final_norm = LayerNorm(
            n_input=d_model,
            dtype=dtype,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )
        self.prediction_head = Linear(
            n_input=d_model,
            n_output=n_classes,
            dtype=dtype,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )

    @property
    def n_params(self) -> int:
        """The number of parameters in the model."""
        return (
            self.embedding_layer.n_params
            + self.initial_norm.n_params
            + self.encoder.n_params
            + self.final_norm.n_params
            + self.prediction_head.n_params
        )

    def get_parameters(self) -> Parameters:
        """Return the parameter map for the layer."""
        params = {
            "embedding_layer": self.embedding_layer.get_parameters(),
            "initial_norm": self.initial_norm.get_parameters(),
            "encoder": self.encoder.get_parameters(),
            "final_norm": self.final_norm.get_parameters(),
            "prediction_head": self.prediction_head.get_parameters(),
        }
        return params

    def load_parameters(self, params: Parameters) -> None:
        """Set the parameters."""
        if (
            "embedding_layer" not in params
            or "initial_norm" not in params
            or "encoder" not in params
            or "final_norm" not in params
            or "prediction_head" not in params
        ):
            raise ValueError("Missing parameters")
        if (
            isinstance(params["embedding_layer"], BaseParameter)
            or isinstance(params["initial_norm"], BaseParameter)
            or isinstance(params["encoder"], BaseParameter)
            or isinstance(params["final_norm"], BaseParameter)
            or isinstance(params["prediction_head"], BaseParameter)
        ):
            raise ValueError("Invalid shape for parameters map")
        self.embedding_layer.load_parameters(params["embedding_layer"])
        self.initial_norm.load_parameters(params["initial_norm"])
        self.encoder.load_parameters(params["encoder"])
        self.final_norm.load_parameters(params["final_norm"])
        self.prediction_head.load_parameters(params["prediction_head"])

    def save(self, model_file: PathLike) -> None:
        """Checkpoint the current model state to disk."""
        model_path = Path(model_file)
        base_dir = model_path.parent
        base_dir.mkdir(parents=False, exist_ok=True)
        constructor_args = {
            "n_classes": self.n_classes,
            "patch_size": self.patch_size,
            "canonical_width": self.canonical_width,
            "canonical_height": self.canonical_height,
            "n_channel": self.n_channel,
            "n_blocks": self.n_blocks,
            "d_model": self.d_model,
            "d_k": self.d_k,
            "d_v": self.d_v,
            "h": self.h,
            "d_ff": self.d_ff,
            "dtype": self.dtype,
        }
        params = self.get_parameters()
        model_definition = [constructor_args, params]
        with open(model_path, "wb") as f:
            pickle.dump(model_definition, f)

    def load(self, model_file: PathLike) -> None:
        """Load model parameters from a checkpoint file on disk."""
        with open(model_file, "rb") as f:
            model_definition = pickle.load(f)
        if not isinstance(model_definition, list) and not len(model_definition) == 2:
            raise ValueError("Invalid model file")
        _, params = model_definition
        self.load_parameters(params)

    @classmethod
    def load_for_training(cls, model_file: PathLike, optimizer: Optimizer) -> VisionTransformer:
        """Initialize a model from a checkpoint file on disk and set the optimizer."""
        with open(model_file, "rb") as f:
            model_definition = pickle.load(f)
        if not isinstance(model_definition, list) and not len(model_definition) == 2:
            raise ValueError("Invalid model file")
        constructor_args, params = model_definition
        model = cls(
            **constructor_args,
            enable_grad=True,
            optimizer=optimizer,
        )
        model.load_parameters(params)
        return model

    @classmethod
    def load_for_eval(cls, model_file: PathLike) -> VisionTransformer:
        """Initialize a model from a checkpoint file on disk in evaluation mode."""
        with open(model_file, "rb") as f:
            model_definition = pickle.load(f)
        if not isinstance(model_definition, list) and not len(model_definition) == 2:
            raise ValueError("Invalid model file")
        constructor_args, params = model_definition
        model = cls(
            **constructor_args,
            enable_grad=False,
            optimizer=None,
        )
        model.load_parameters(params)
        return model

    def represent(self, x: np.ndarray) -> np.ndarray:
        """Compute the pre-final output for a given input."""
        assert x.ndim == 4  # (B, H, W, C)

        # T := N + 1; N = number of image patches = H*W/P^2
        raw_embedding = self.embedding_layer.forward(x)  # shape = (B, T, d_model)
        encoder_input = self.initial_norm.forward(raw_embedding)  # shape = (B, T, d_model)
        refined_embedding = self.encoder.forward(encoder_input)  # shape = (B, T, d_model)
        normed_embedding = self.final_norm.forward(refined_embedding)  # shape = (B, T, d_model)

        # Output is the representation of the prepended class token in the ImageEmbedding layer
        y = normed_embedding[:, 0, :]  # shape = (B, d_model)

        return y

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.ndim == 4  # (B, H, W, C)

        y = self.represent(x)  # shape = (B, d_model)
        logits = self.prediction_head.forward(y)  # shape = (B, n_classes)

        return logits

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        x = self.embedding_layer.cache["x"]
        B, H, W, C = x.shape
        N = H * W // (self.patch_size**2)
        T = N + 1
        assert dout.shape == (B, self.n_classes)

        dlogits = dout
        self.prediction_head.backward(dlogits)
        dy = self.prediction_head.cache["dx"]
        dnormed_embedding = np.zeros((B, T, self.d_model))
        dnormed_embedding[:, 0, :] = dy
        self.final_norm.backward(dnormed_embedding)
        drefined_embedding = self.final_norm.cache["dx"]
        self.encoder.backward(drefined_embedding)
        dencoder_input = self.encoder.cache["dx"]
        self.initial_norm.backward(dencoder_input)
        draw_embedding = self.initial_norm.cache["dx"]
        self.embedding_layer.backward(draw_embedding)

    def step(self) -> None:
        """Perform a single optimization step."""
        assert self.enable_grad, "Cannot take an optimization step with enable_grad=False"
        assert self.optimizer, "Cannot take an optimization step with optimizer=None"

        self.prediction_head.step()
        self.final_norm.step()
        self.encoder.step()
        self.initial_norm.step()
        self.embedding_layer.step()
