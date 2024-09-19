import matplotlib.pyplot as plt
import numpy as np

from .ej14.defpos import sol_defpos


def main():
    n = 500
    A = -np.diag(np.ones(n - 1), -1) - np.diag(np.ones(n - 1), 1) + 2 * np.eye(n)

    c = n / 10
    b = np.exp(-((np.arange(n) - c) ** 2) / 100)
    x = sol_defpos(A, b.copy())

    plt.plot(x)
    plt.plot(b)
    plt.show()


if __name__ == "__main__":
    main()
