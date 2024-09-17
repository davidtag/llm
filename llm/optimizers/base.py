"""Interface for implementing optimizers."""

import abc

import numpy as np


class ParameterOptimizer(abc.ABC):
    """Abstract base class for a function optimizer specialized to a single parameter.

    It stores all relevant state to perform update steps.
    """

    def __init__(self, w: np.ndarray) -> None:
        """Initialize the parameter optimizer."""
        self.w = w

    @abc.abstractmethod
    def step(self, dw: np.ndarray) -> None:
        """Update the parameter based on the current value of the gradient."""
        raise NotImplementedError


class Optimizer(abc.ABC):
    """Abstract base class for a function optimizer.

    An Optimizer serves as a 'template'. It doesn't itself optimize a parameter,
    but has a factory method for creating ParameterOptimizer(s) that do.
    """

    @abc.abstractmethod
    def get_parameter_optimizer(self, w: np.ndarray) -> ParameterOptimizer:
        """Create a parameter optimizer for `w`."""
        raise NotImplementedError
