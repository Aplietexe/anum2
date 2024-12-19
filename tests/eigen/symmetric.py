import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def test_symmetric_diagonalization(
    diagonalize: Callable[[Arr, float, int], tuple[Arr, Arr]],
    maxn: int = 30,
    maxa: float = 1e2,
    rtol: float = 1e-13,
    atol: float = 1e-15,
    eps: float = 1e-7,
    its: int = 1000,
) -> float:
    """
    Tests diagonalization. Returns the average time to diagonalize a matrix.
    """
    t = 0.0
    c = 0
    while c < its:
        n = np.random.randint(2, maxn)
        A = np.random.uniform(-maxa, maxa, (n, n))
        A[np.triu_indices(n, 1)] = A.T[np.triu_indices(n, 1)]
        off = np.linalg.norm(A - np.diag(np.diag(A)))
        if n == 2:
            iters = 1
        else:
            iters = 2 * np.log(eps / off) / np.log(1 - 2 / (n * (n - 1)))
            iters = int(np.ceil(iters))
        if iters >= 2000:
            continue
        _A = A.copy()
        t -= time.perf_counter()
        D, Q = diagonalize(_A, eps, iters)
        t += time.perf_counter()

        # Correct shapes
        assert D.shape == (n, n)
        assert Q.shape == (n, n)

        # Factorizes A
        tol = max(float(rtol * np.linalg.norm(D)), atol)
        assert np.linalg.norm(Q.T @ A @ Q - D) < tol

        # Q is orthogonal
        tol = max(float(rtol * np.linalg.norm(np.eye(n))), atol)
        assert np.linalg.norm(Q.T @ Q - np.eye(n)) < tol

        # D is diagonal
        assert np.linalg.norm(D - np.diag(np.diag(D))) <= eps

        c += 1

    return t / c
