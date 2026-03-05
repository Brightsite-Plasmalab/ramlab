from typing import TypeVar

import numpy as np

T = TypeVar('T', covariant=True)

# Numpy
NDArray1D = np.ndarray[tuple[int], T]
floatNDArray1D = NDArray1D[np.floating]
intNDArray1D = NDArray1D[np.integer]