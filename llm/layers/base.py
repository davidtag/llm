"""Abstract base class for all neural network layer implementations."""

import abc
from typing import Optional

import numpy as np

from llm.constants import DType, DEFAULT_DTYPE, Parameters
from llm.optimizers import Optimizer


class Layer(abc.ABC):
    """Abstract base class for all neural network layer implementations."""

    def __init__(
        self,
        *,
        dtype: DType = DEFAULT_DTYPE,
        enable_grad: bool = True,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the layer."""
        self.dtype = dtype
        self.enable_grad = enable_grad
        self.optimizer = optimizer
        self.cache: dict[str, np.ndarray] = {}

    @property
    @abc.abstractmethod
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_parameters(self) -> Parameters:
        """Return the parameter map for the layer."""
        raise NotImplementedError

    @abc.abstractmethod
    def load_parameters(self, params: Parameters) -> None:
        """Set the parameters."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        raise NotImplementedError

    @abc.abstractmethod
    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self) -> None:
        """Perform a single optimization step."""
        raise NotImplementedError
