import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def givens_2d(x: Arr) -> Arr:
    """
    Returns the Givens rotation matrix R that zeroes the second element of a 2D vector.
    """
    if x.shape != (2,):
        raise ValueError("Invalid shape")
    x1, x2 = x
    if np.isclose(x2, 0):
        return np.array([[1, 0], [0, 1]])

    if np.abs(x2) > np.abs(x1):
        t = -x1 / x2
        s = -np.sign(x2) / np.sqrt(1 + t**2)
        c = s * t
    else:
        t = -x2 / x1
        c = np.sign(x1) / np.sqrt(1 + t**2)
        s = c * t

    return np.array([[c, -s], [s, c]])
