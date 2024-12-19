import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def gastinel(A: Arr, b: Arr, x: Arr, eps: float, maxits: int) -> Arr:
    n = A.shape[0]
    if A.shape != (n, n) or b.shape != (n,) or x.shape != (n,):
        raise ValueError("Invalid shapes")

    r = b - A @ x
    sigma = np.linalg.norm(r, np.inf)
    d = np.sign(r)

    for k in range(maxits):
        if sigma <= eps:
            break
        else:
            alpha = np.dot(r, d) / np.dot(d, A @ d)
            x = x + alpha * d
            r = r - alpha * (A @ d)
            d = np.sign(r)
            sigma = np.linalg.norm(r, np.inf)

    return x


if __name__ == "__main__":
    from tests.systems.positive_definite import test_posdef_solve

    def solve(A: Arr, b: Arr) -> Arr:
        return gastinel(A, b, np.random.uniform(-100, 100, b.shape[0]), 1e-8, 1000000)

    t = test_posdef_solve(solve, max_cond=1e3, rtol=1e-8)
    print(t)
