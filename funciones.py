import numpy as np

# ============================================================
# FUNCIÓN DE ROSENBROCK
# ============================================================


def rosenbrock(x):
    """

    Mínimo global:
        x = (1, 1, ..., 1)

    Valor mínimo:
        f(x) = 0
    """

    x = np.asarray(x, dtype=float)

    return np.sum(
        100 * (x[1:] - x[:-1] ** 2) ** 2
        + (1 - x[:-1]) ** 2
    )

def grad_rosenbrock(x):
    """
    Gradiente de la función de Rosenbrock.
    """

    x = np.asarray(x, dtype=float)

    grad = np.zeros_like(x)

    grad[:-1] += (
        -400 * x[:-1] * (x[1:] - x[:-1] ** 2)
        - 2 * (1 - x[:-1])
    )

    grad[1:] += (
        200 * (x[1:] - x[:-1] ** 2)
    )

    return grad


# ============================================================
# FUNCIÓN DE RASTRIGIN
# ============================================================

def rastrigin(x):
    """

    Mínimo global:
        x = (0, 0, ..., 0)

    Valor mínimo:
        f(x) = 0
    """

    x = np.asarray(x, dtype=float)

    n = len(x)

    return 10 * n + np.sum(
        x ** 2 - 10 * np.cos(2 * np.pi * x)
    )


def grad_rastrigin(x):
    """
    Gradiente de la función de Rastrigin.
    """

    x = np.asarray(x, dtype=float)

    return (
        2 * x
        + 20 * np.pi * np.sin(2 * np.pi * x)
    )