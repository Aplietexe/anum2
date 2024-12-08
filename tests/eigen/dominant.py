import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def test_dominant_eigen(
    dominant_eigen: Callable[[Arr, Arr, np.float64, int], tuple[Arr, float]],
    norm: float,
    maxn: int = 30,
    maxa: float = 1e2,
    eps: float = 1e-7,
    rtol: float = 5e-4,
    atol: float = 1e-7,
    its: int = 7000,
) -> float:
    """
    Tests dominant eigenvalue computation. Returns the average time to compute the dominant eigenvalue.
    """
    t = 0.0
    c = 0
    while c < its:
        n = np.random.randint(2, maxn)
        eigvals = np.random.uniform(-maxa, maxa, n)
        eigvecs = np.random.uniform(-maxa, maxa, (n, n))
        A = eigvecs @ np.diag(eigvals) @ np.linalg.inv(eigvecs)
        q0 = np.random.uniform(-maxa, maxa, n)
        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvecs = eigvecs[:, idx]
        eigvals = eigvals[idx]

        gap = (np.abs(eigvals[0]) - np.abs(eigvals[1])) / np.abs(eigvals[0])
        coords = np.linalg.inv(eigvecs) @ q0
        if gap < 0.03 or coords[0] / np.linalg.norm(coords) < 1 / n:
            continue

        _A = A.copy()
        _q0 = q0.copy()
        t -= time.perf_counter()
        q, rho = dominant_eigen(_A, _q0, np.float64(eps), 2000)
        t += time.perf_counter()

        # Correct shapes
        assert q.shape == (n,)

        # rho is dominant eigenvalue
        tol = max(float(rtol * np.abs(eigvals[0])), atol)
        assert np.abs(rho - eigvals[0]) < tol

        # q is corresponding eigenvector
        tol = max(float(rtol * np.linalg.norm(rho * q)), atol)
        assert np.linalg.norm(A @ q - rho * q) < tol

        # q is normalized
        assert np.abs(np.linalg.norm(q, norm) - 1) < rtol

        c += 1

    return t / c
