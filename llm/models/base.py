"""Abstract base class for all neural network model implementations."""

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

    # TODO(dtag): Add method to dump/load from a file
