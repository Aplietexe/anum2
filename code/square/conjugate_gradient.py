import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def cg(A: Arr, b: Arr, x: Arr, eps: float, maxits: int) -> Arr:
    n = A.shape[0]
    if A.shape != (n, n) or b.shape != (n,) or x.shape != (n,):
        raise ValueError("Invalid shapes")

    r = b - A @ x
    sigma = np.linalg.norm(r)
    d = r

    for _ in range(maxits):
        if sigma < eps:
            return x
        v = A @ d
        t = sigma**2 / (d @ v)
        x += t * d
        r -= t * v
        sigma_new = np.linalg.norm(r)
        s = sigma_new**2 / sigma**2
        d = r + s * d
        sigma = sigma_new

    print("Warning: Maximum number of iterations reached")

    return x


if __name__ == "__main__":
    from tests.systems.positive_definite import test_posdef_solve

    def solve(A: Arr, b: Arr) -> Arr:
        return cg(A, b, np.random.uniform(-100, 100, b.shape[0]), 1e-6, 1000000)

    t = test_posdef_solve(solve, maxn=1000, its=5)
    print(t)
