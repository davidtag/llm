"""Python data types for handling tokenization."""

import numpy as np
from numpy.typing import NDArray


TokenDtype = np.uint32
NumpyTokenSequence = NDArray[TokenDtype]
