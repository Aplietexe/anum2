from code.systems.gmres import gmres

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def sol_newton(func, x_0: Arr, eps: float, m: int):
    # Paso 0
    x = x_0.copy()
    delta = 0.5
    gamma = 0.5
    r, M = func(x, valfun=True, derfun=True)
    sigma = np.linalg.norm(r)

    # Paso 1
    for _ in range(m):
        if sigma <= eps:
            break
        eta = min(gamma, float(sigma) ** delta)
        zero = np.zeros_like(r)
        d = gmres(M, -r, zero, eta * float(sigma), 1000, 3)
        x = x + d
        r, M = func(x, valfun=True, derfun=True)
        sigma = np.linalg.norm(r)

    # Paso 2
    return x


if __name__ == "__main__":
    from practicos.p7.ej1 import fun_tres

    x_0 = np.random.random(3)
    x_sol = sol_newton(fun_tres, x_0, 1e-9, 1000)
    print(x_sol, fun_tres(x_sol)[0])
