"""Mathematical utilities."""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Compute the element-wise recitified linear unit."""
    return np.maximum(0, x)


def log_sum_exp(x: np.ndarray) -> np.ndarray:
    """Compute the log of the sum of exponentials of `x` along the last dimension.

    Args:
        x: Real-valued array of dimension (*, N)

    Returns:
        y: Real-valued array of dimension (*, 1)
    """
    dims = x.ndim
    # NOTE: for numerical stability, we shift the input by its maximum. This makes all exponents <=0.
    # For a vector x: LSE(x) = LSE(x - m) + m
    x_max = np.max(x, axis=dims - 1, keepdims=True)
    x_shifted = x - x_max
    exp_x_shifted = np.exp(x_shifted)
    sum_exp_x_shifted = np.sum(exp_x_shifted, axis=dims - 1, keepdims=True)
    lse_shifted = np.log(sum_exp_x_shifted)
    out = lse_shifted + x_max
    return out


def log_softmax(x: np.ndarray) -> np.ndarray:
    """Compute the log of the softmax of `x` along the last dimension.

    Args:
        x: Real-valued array of dimension (*, N)

    Returns:
        y: Real-valued array of dimension (*, N)
    """
    return x - log_sum_exp(x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute the softmax of `x` along the last dimension.

    Args:
        x: Real-valued array of dimension (*, N)

    Returns:
        y: Collection of probability distributions of dimension (*, N)
    """
    dims = x.ndim
    # NOTE: for numerical stability, we shift the input by its maximum. This makes all exponents <=0.
    # For a vector x: Softmax(x) = Softmax(x - m); i.e., it is translation-invariant
    x_max = np.max(x, axis=dims - 1, keepdims=True)
    x_shifted = x - x_max
    exp_x_shifted = np.exp(x_shifted)
    sum_exp_x_shifted = np.sum(exp_x_shifted, axis=dims - 1, keepdims=True)
    out = exp_x_shifted / sum_exp_x_shifted
    return out
