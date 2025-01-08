from typing import Callable, Tuple
from nptyping import NDArray
import numpy as np


def central_gradient(
    f: Callable[[Tuple[float, ...]], float], arr: NDArray, h: float = 1e-8
):
    """
    Вычисляет численную производную функции f в точке x.

    Args:
        f: Функция, производную которой нужно вычислить.
        x: Точка, в которой нужно вычислить производную.
        h: Небольшое значение для разности (шаг).

    Returns:
        Численное приближение производной.
    """
    n = arr.size
    grad = np.zeros(n, dtype=float)

    for i in range(n):
        arr_plus_h = arr.copy()
        arr_minus_h = arr.copy()
        arr_plus_h[i] = arr[i] + h
        arr_minus_h[i] = arr[i] - h

        grad[i] = (f(*arr_plus_h) - f(*arr_minus_h)) / (2 * h)

    return grad
