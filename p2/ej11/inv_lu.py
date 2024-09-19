import numpy as np
from numpy.typing import NDArray

from p1.ej3.inf_fil import sol_trinffil
from p1.ej3.sup_fil import sol_trsupfil

from ..ej9.lu_p import dlup
from .tests import test_inv

Arr = NDArray[np.float64]


def inv_lu(A: Arr) -> Arr | None:
    """
    Given a square matrix A, computes its inverse if it exists.
    Overwrites A with the inverse and returns it.
    Returns None if A is not invertible.
    """
    n = A.shape[0]
    L, U, P = dlup(A)
    if np.isclose(np.prod(np.diag(U)), 0):
        return None

    for i in range(n):
        y = sol_trinffil(L, P[:, i])
        A[:, i] = sol_trsupfil(U, y)

    return A


if __name__ == "__main__":
    t = test_inv(inv_lu)
    print(t)
