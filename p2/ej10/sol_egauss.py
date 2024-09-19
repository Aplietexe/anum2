import numpy as np
from numpy.typing import NDArray

from p1.ej3.sup_fil import sol_trsupfil

from ..ej9.gauss_p import egaussp
from .tests import test_square

Arr = NDArray[np.float64]


def sol_egauss(A: Arr, b: Arr) -> Arr:
    """
    Solves the system Ax = b using Gaussian elimination.
    Overwrites A with U upper triangular and b with the solution.
    Returns the solution.
    """
    A, b = egaussp(A, b)
    return sol_trsupfil(np.triu(A), b)


if __name__ == "__main__":
    t = test_square(sol_egauss)
    print(t)
