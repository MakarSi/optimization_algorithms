from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from nptyping import NDArray
import numpy as np


@dataclass
class TestFunction:
    function: Callable
    bounds: List[Tuple[float, float]]
    start_point: Optional[NDArray] = None
    alpha: float = 0.01
    name: str = ""


func1 = lambda x, y: x**2.0 + y**2.0
func2 = lambda x, y: 0.26 * (x**2 + y**2) - 0.48 * x * y
func3 = (
    lambda x, y: -np.cos(x)
    * np.cos(y)
    * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))
)
func4 = (
    lambda x, y: -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    + np.e
    + 20
)
func5 = lambda x, y: (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

test_functions = [
    TestFunction(
        function=func1,
        name="func_1",
        bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
    ),
    TestFunction(
        function=func2,
        name="func_2",
        bounds=np.array([[-10.0, 10.0], [-10.0, 10.0]]),
    ),
    TestFunction(
        function=func3,
        name="func_3",
        bounds=np.array([[-10.0, 10.0], [-10.0, 10.0]]),
        start_point=[2.1, 4.1],
    ),
    TestFunction(
        function=func4,
        name="func_4",
        bounds=np.array([[-5.0, 5.0], [-5.0, 5.0]]),
        alpha=0.5,
    ),
    TestFunction(
        function=func5,
        name="func_5",
        bounds=np.array([[-5.0, 5.0], [-5.0, 5.0]]),
    ),
]
