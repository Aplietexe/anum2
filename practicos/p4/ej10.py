from code.least_squares.householder import least_squares

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def min_norm_1(A: Arr, b: Arr) -> tuple[Arr, np.float64]:
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError("A and b must have compatible shapes.")

    A_ub = np.block([[A, -np.eye(m)], [-A, -np.eye(m)]])
    b_ub = np.block([b, -b])
    c = np.block([np.zeros(n), np.ones(m)])

    res = sp.optimize.linprog(c, A_ub, b_ub)
    assert res.success

    return res.x[:n], res.fun


def min_norm_inf(A: Arr, b: Arr) -> tuple[Arr, np.float64]:
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError("A and b must have compatible shapes.")

    A_ub = np.block([[A, -np.ones((m, 1))], [-A, -np.ones((m, 1))]])
    b_ub = np.block([b, -b])
    c = np.block([np.zeros(n), np.ones(1)])

    res = sp.optimize.linprog(c, A_ub, b_ub)
    assert res.success

    return res.x[:n], res.fun


def line_fit(x: Arr, y: Arr, minimizer) -> tuple[Arr, np.float64]:
    A = np.column_stack([np.ones_like(x), x])
    return minimizer(A, y)


if __name__ == "__main__":
    x = np.arange(1.0, 11)
    print(x)
    y = np.arange(1.0, 11)
    y[-1] = 0

    plt.plot(x, y, "o")
    plt.grid()

    # 2-norm
    x_hat, r = line_fit(x, y, least_squares)
    print(x_hat, r)
    plt.plot(x, np.polyval(x_hat[::-1], x), label="2-norm")

    # 1-norm
    x_hat, r = line_fit(x, y, min_norm_1)
    print(x_hat, r)
    plt.plot(x, np.polyval(x_hat[::-1], x), label="1-norm")

    # inf-norm
    x_hat, r = line_fit(x, y, min_norm_inf)
    print(x_hat, r)
    plt.plot(x, np.polyval(x_hat[::-1], x), label="inf-norm")

    plt.legend()
    plt.show()
