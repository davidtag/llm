"""Implementation of a Transformer model architecture."""

from __future__ import annotations

from pathlib import Path
import pickle
from os import PathLike
from typing import Generator, Optional

import numpy as np

from llm.constants import DType, DEFAULT_DTYPE, BaseParameter, Parameters
from llm.layers.block_stack import BlockStack
from llm.layers.linear import Linear
from llm.layers.layer_norm import LayerNorm
from llm.layers.text_embedding import TextEmbedding
from llm.models.base import Model
from llm.optimizers import Optimizer
from llm.utils.math import softmax


class Transformer(Model):
    """A Transformer architecture for sequence processing."""

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
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
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff

        self.embedding_layer = TextEmbedding(
            vocab_size=vocab_size,
            context_size=context_size,
            d_model=d_model,
            dtype=dtype,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )
        self.decoder = BlockStack(
            n_blocks=n_blocks,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            h=h,
            d_ff=d_ff,
            masked_attention=True,
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
        self.unembedding_layer = Linear(
            n_input=d_model,
            n_output=vocab_size,
            dtype=dtype,
            enable_grad=enable_grad,
            optimizer=optimizer,
        )

    @property
    def n_params(self) -> int:
        """The number of parameters in the model."""
        return (
            self.embedding_layer.n_params
            + self.decoder.n_params
            + self.final_norm.n_params
            + self.unembedding_layer.n_params
        )

    def get_parameters(self) -> Parameters:
        """Return the parameter map for the layer."""
        params = {
            "embedding_layer": self.embedding_layer.get_parameters(),
            "decoder": self.decoder.get_parameters(),
            "final_norm": self.final_norm.get_parameters(),
            "unembedding_layer": self.unembedding_layer.get_parameters(),
        }
        return params

    def load_parameters(self, params: Parameters) -> None:
        """Set the parameters."""
        if (
            "embedding_layer" not in params
            or "decoder" not in params
            or "final_norm" not in params
            or "unembedding_layer" not in params
        ):
            raise ValueError("Missing parameters")
        if (
            isinstance(params["embedding_layer"], BaseParameter)
            or isinstance(params["decoder"], BaseParameter)
            or isinstance(params["final_norm"], BaseParameter)
            or isinstance(params["unembedding_layer"], BaseParameter)
        ):
            raise ValueError("Invalid shape for parameters map")
        self.embedding_layer.load_parameters(params["embedding_layer"])
        self.decoder.load_parameters(params["decoder"])
        self.final_norm.load_parameters(params["final_norm"])
        self.unembedding_layer.load_parameters(params["unembedding_layer"])

    def save(self, model_file: PathLike) -> None:
        """Checkpoint the current model state to disk."""
        model_path = Path(model_file)
        base_dir = model_path.parent
        base_dir.mkdir(parents=False, exist_ok=True)
        constructor_args = {
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
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
    def load_for_training(cls, model_file: PathLike, optimizer: Optimizer) -> Transformer:
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
    def load_for_eval(cls, model_file: PathLike) -> Transformer:
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.ndim == 2  # shape = (B, T)

        raw_embedding = self.embedding_layer.forward(x)  # shape = (B, T, d_model)
        refined_embedding = self.decoder.forward(raw_embedding)  # shape = (B, T, d_model)
        normed_embedding = self.final_norm.forward(refined_embedding)  # shape = (B, T, d_model)
        logits = self.unembedding_layer.forward(normed_embedding)  # shape = (B, T, vocab_size)

        return logits

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        x = self.embedding_layer.cache["x"]
        B, T = x.shape
        assert dout.shape == (B, T, self.vocab_size)

        dlogits = dout
        self.unembedding_layer.backward(dlogits)
        dnormed_embedding = self.unembedding_layer.cache["dx"]
        self.final_norm.backward(dnormed_embedding)
        drefined_embedding = self.final_norm.cache["dx"]
        self.decoder.backward(drefined_embedding)
        draw_embedding = self.decoder.cache["dx"]
        self.embedding_layer.backward(draw_embedding)

    def step(self) -> None:
        """Perform a single optimization step."""
        assert self.enable_grad, "Cannot take an optimization step with enable_grad=False"
        assert self.optimizer, "Cannot take an optimization step with optimizer=None"

        self.unembedding_layer.step()
        # TODO(dtag): final_norm.step() is missing
        self.decoder.step()
        self.embedding_layer.step()

    def predict(self, input_sequence: np.ndarray) -> np.ndarray:
        """Generate a probability distribution over the next token for a given input sequence."""
        assert input_sequence.ndim == 1
        x = np.expand_dims(input_sequence, axis=0)  # add batch dimension
        logits = self.forward(x)
        next_token_logits = logits[0, -1]
        probabilities = softmax(next_token_logits)
        return probabilities

    def generate(self, start_sequence: np.ndarray, max_tokens: int = 5, is_random: bool = True) -> np.ndarray:
        """Generate an output sequence based on predicted next token probabilities."""
        output_sequence = []

        for token in self.generate_stream(
            start_sequence=start_sequence, max_tokens=max_tokens, is_random=is_random
        ):
            output_sequence.append(token)

        return np.array(output_sequence)

    def generate_stream(
        self, start_sequence: np.ndarray, max_tokens: int = 5, is_random: bool = True
    ) -> Generator[int]:
        """Generate an output stream based on predicted next token probabilities."""
        assert start_sequence.ndim == 1
        current_sequence = start_sequence.copy()

        for _ in range(max_tokens):
            probs = self.predict(current_sequence)
            if is_random:
                token = np.random.choice(self.vocab_size, p=probs)
            else:
                token = int(np.argmax(probs))
            current_sequence = np.append(current_sequence, token)[-self.context_size :]
            yield token
