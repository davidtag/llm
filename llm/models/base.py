"""Abstract base class for all neural network model implementations."""

from __future__ import annotations

import abc
from os import PathLike
from typing import Optional

from llm.constants import DType, DEFAULT_DTYPE
from llm.layers import Layer
from llm.optimizers import Optimizer


class Model(Layer):
    """Abstract base class for all neural network model implementations."""

    def __init__(
        self,
        *,
        dtype: DType = DEFAULT_DTYPE,
        enable_grad: bool = True,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the layer."""
        super().__init__(dtype=dtype, enable_grad=enable_grad, optimizer=optimizer)

    @abc.abstractmethod
    def save(self, model_file: PathLike) -> None:
        """Checkpoint the current model state to disk."""
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, model_file: PathLike) -> None:
        """Load model parameters from a checkpoint file on disk."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load_for_training(cls, model_file: PathLike, optimizer: Optimizer) -> Model:
        """Initialize a model from a checkpoint file on disk and set the optimizer."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load_for_eval(cls, model_file: PathLike) -> Model:
        """Initialize a model from a checkpoint file on disk in evaluation mode."""
        raise NotImplementedError
