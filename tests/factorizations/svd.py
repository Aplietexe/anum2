import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def test_svd(
    svd: Callable[[Arr], tuple[Arr, Arr, Arr]],
    maxn: int = 20,
    maxa: float = 1e2,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    its: int = 1000,
) -> float:
    """
    Tests diagonalization. Returns the average time to diagonalize a matrix.
    """
    t = 0.0
    c = 0
    while c < its:
        n = np.random.randint(2, maxn)
        m = np.random.randint(2, maxn)
        A = np.random.uniform(-maxa, maxa, (m, n))
        _A = A.copy()
        t -= time.perf_counter()
        U, S, Vt = svd(_A)
        t += time.perf_counter()

        # Correct shapes
        assert U.shape == (m, m)
        assert S.shape == (m, n)
        assert Vt.shape == (n, n)

        # Factorizes A
        tol = max(float(rtol * np.linalg.norm(A)), atol)
        assert np.linalg.norm(U @ S @ Vt - A) < tol

        # U is orthogonal
        tol = max(float(rtol * np.linalg.norm(np.eye(m))), atol)
        assert np.linalg.norm(U.T @ U - np.eye(m)) < tol

        # Vt is orthogonal
        tol = max(float(rtol * np.linalg.norm(np.eye(n))), atol)
        assert np.linalg.norm(Vt @ Vt.T - np.eye(n)) < tol

        # S is diagonal
        p = min(m, n)
        sq = S[:p, :p]
        assert np.linalg.norm(sq - np.diag(np.diag(sq))) <= atol

        # S is sorted
        assert np.all(np.diff(np.diag(sq)) <= 0)

        c += 1
        print(c)

    return t / c
