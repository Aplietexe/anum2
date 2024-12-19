import numpy as np
from numpy.typing import NDArray

from ..factorizations.arnoldi import arnoldi
from ..least_squares.householder import least_squares

Arr = NDArray[np.float64]


def gmres(A: Arr, b: Arr, x: Arr, eps: float, m: int, m_a: int) -> Arr:
    n = A.shape[0]
    if A.shape != (n, n) or b.shape != (n,) or x.shape != (n,):
        raise ValueError("Invalid shapes")

    r = b - A @ x
    sigma = np.linalg.norm(r)
    for _ in range(m):
        print(sigma)
        if sigma < eps:
            return x
        H, V = arnoldi(A.copy(), r.copy(), m_a)
        v = np.zeros(H.shape[0])
        v[0] = sigma
        alpha, sigma = least_squares(H.copy(), v.copy())
        # alpha, sigma = np.linalg.lstsq(H, v, rcond=None)[0:2]
        print(sigma)
        x += V @ alpha
        r = b - A @ x

    print("Warning: Maximum number of iterations reached")

    return x


if __name__ == "__main__":
    # TEST con calor
    import matplotlib.pyplot as plt

    from practicos.p6.calor import calor

    N = 100
    A, b = calor(N)
    x_0 = np.random.random((N - 2) ** 2)

    x_mas = gmres(A, b, x_0, 1e-5, 100, 5)
    plt.imshow(x_mas.reshape(N - 2, N - 2))
    plt.show()
