from code.systems.posdef.cholesky import solve_positive_definite

import matplotlib.pyplot as plt
import numpy as np


def main():
    n = 500
    A = -np.diag(np.ones(n - 1), -1) - np.diag(np.ones(n - 1), 1) + 2 * np.eye(n)

    c = 9 * n / 10
    b = np.exp(-((np.arange(n) - c) ** 2) / 100)
    x = solve_positive_definite(A, b.copy())

    plt.plot(x)
    plt.plot(b)
    plt.show()


if __name__ == "__main__":
    main()
