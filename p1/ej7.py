import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def nivel(levels: Arr):
    G = np.random.uniform(0, 10, (2, 2))
    G = np.triu(G)
    A = G.T @ G

    n = 100
    nj = 1j * n

    x = np.mgrid[-10:10:nj, -10:10:nj].reshape(2, -1)
    y = np.sum(x * (A @ x), axis=0).reshape(n, n)

    grid = x.reshape(2, n, n)
    plt.contour(grid[0], grid[1], y, levels=levels)
    plt.show()


if __name__ == "__main__":
    nivel(np.linspace(0, 200, 10))
