"""Configuration constants."""

from typing import Mapping, Union

import numpy as np


DType = Union[str, np.dtype, type]
DEFAULT_DTYPE: DType = np.float64


BaseParameter = np.ndarray
Parameters = Mapping[str, Union[BaseParameter, "Parameters"]]
