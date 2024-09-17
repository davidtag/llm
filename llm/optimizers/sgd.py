"""Implements the stochastic gradient descent optimizer."""

from typing import List

import numpy as np

from llm.optimizers.base import Optimizer, ParameterOptimizer


class StochasticGradientDescentParameterOptimizer(ParameterOptimizer):
    """Implements the stochastic gradient descent parameter optimizer."""

    def __init__(
        self,
        w: np.ndarray,
        lr: float,
    ) -> None:
        """Initialize the parameter optimizer."""
        super().__init__(w=w)
        self.lr = lr

    def step(self, dw: np.ndarray) -> None:
        """Update the parameter based on the current value of the gradient."""
        assert dw.shape == self.w.shape

        self.w -= self.lr * dw


class StochasticGradientDescent(Optimizer):
    """Implements first-order stochastic gradient descent."""

    def __init__(
        self,
        lr: float = 0.001,
    ) -> None:
        """Initialize the optimizer."""
        self.lr = lr
        self.parameter_optimizers: List[StochasticGradientDescentParameterOptimizer] = []

    def set_learning_rate(self, lr: float) -> None:
        """Update the learning rate."""
        self.lr = lr
        for p_optimizer in self.parameter_optimizers:
            p_optimizer.lr = lr

    def get_parameter_optimizer(self, w: np.ndarray) -> StochasticGradientDescentParameterOptimizer:
        """Create a parameter optimizer for `w`."""
        p_optimizer = StochasticGradientDescentParameterOptimizer(
            w=w,
            lr=self.lr,
        )
        self.parameter_optimizers.append(p_optimizer)
        return p_optimizer
