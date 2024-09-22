"""Configuration constants."""

from typing import Union

import numpy as np

DType = Union[str, np.dtype, type]
DEFAULT_DTYPE: np.dtype = np.dtype("float64")
