import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def cauchy(A: Arr, b: Arr, x: Arr, eps: float, maxits: int) -> Arr:
    n = A.shape[0]
    if A.shape != (n, n) or b.shape != (n,) or x.shape != (n,):
        raise ValueError("Invalid shapes")

    r = b - A @ x
    sigma = np.linalg.norm(r)

    for _ in range(maxits):
        if sigma < eps:
            return x
        v = A @ r
        t = sigma**2 / (r @ v)
        x += t * r
        r -= t * v
        sigma = np.linalg.norm(r)

    print("Warning: Maximum number of iterations reached")

    return x


if __name__ == "__main__":
    from tests.systems.positive_definite import test_posdef_solve

    def solve(A: Arr, b: Arr) -> Arr:
        return cauchy(A, b, np.random.uniform(-100, 100, b.shape[0]), 1e-8, 1000000)

    t = test_posdef_solve(solve, max_cond=1e3, rtol=1e-8)
    print(t)
