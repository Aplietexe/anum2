import numpy as np
from numpy.typing import NDArray

from ..ej3.inf_fil import sol_trinffil
from ..ej3.sup_fil import sol_trsupfil
from .cholesky import cholesky
from .tests import test_posdef_solve

Arr = NDArray[np.float64]


def sol_defpos(A: Arr, b: Arr) -> Arr:
    """
    Solves Ax = b, where A is a positive definite matrix.
    Overwrites b with x, does not overwrite A.
    """
    n = A.shape[0]

    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("Invalid shapes")
    if not np.allclose(A, A.T):
        raise ValueError("Matrix is not symmetric")
    U = cholesky(A)
    if U is None:
        raise ValueError("Matrix is not positive definite")

    U = np.triu(U)
    y = sol_trinffil(U.T, b)
    x = sol_trsupfil(U, y)
    return x


if __name__ == "__main__":
    t = test_posdef_solve(sol_defpos)
    print(t)
