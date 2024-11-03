"""Implements the Adam optimizer."""

from typing import List

import numpy as np

from llm.optimizers.base import Optimizer, ParameterOptimizer


class AdamParameterOptimizer(ParameterOptimizer):
    """Implements the Adam parameter optimizer."""

    def __init__(
        self,
        w: np.ndarray,
        lr: float,
        beta_1: float,
        beta_2: float,
        epsilon: float,
    ) -> None:
        """Initialize the parameter optimizer."""
        super().__init__(w=w)
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.t = 0
        self.m = np.zeros_like(w)
        self.v = np.zeros_like(w)

    def step(self, dw: np.ndarray) -> None:
        """Update the parameter based on the current value of the gradient."""
        assert dw.shape == self.w.shape

        self.t = self.t + 1
        self.m = (self.beta_1) * self.m + (1 - self.beta_1) * dw
        self.v = (self.beta_2) * self.v + (1 - self.beta_2) * np.square(dw)
        m_hat = self.m / (1 - np.power(self.beta_1, self.t))
        v_hat = self.v / (1 - np.power(self.beta_2, self.t))
        dw_hat = (m_hat) / (np.sqrt(v_hat) + self.epsilon)

        self.w -= self.lr * dw_hat


class Adam(Optimizer):
    """Implements the Adam optimizer."""

    def __init__(
        self,
        lr: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize the optimizer."""
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.parameter_optimizers: List[AdamParameterOptimizer] = []

    def set_learning_rate(self, lr: float) -> None:
        """Update the learning rate."""
        self.lr = lr
        for p_optimizer in self.parameter_optimizers:
            p_optimizer.lr = lr

    def get_parameter_optimizer(self, w: np.ndarray) -> AdamParameterOptimizer:
        """Create a parameter optimizer for `w`."""
        p_optimizer = AdamParameterOptimizer(
            w=w,
            lr=self.lr,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
        )
        self.parameter_optimizers.append(p_optimizer)
        return p_optimizer
