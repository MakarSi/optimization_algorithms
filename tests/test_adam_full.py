import numpy as np
import pytest

from src.visual.visual import plot_3dpath
from src.algorithms.adam import Adam, AdamConfig
from function_data import TestFunction, test_functions


@pytest.mark.parametrize("func", test_functions)
def test_adam_full(func: TestFunction):
    adam_optimizer = Adam(
        objective=func.function,
        bounds=func.bounds,
        config=AdamConfig(alpha=func.alpha),
        start_point=func.start_point,
    )
    for _ in range(1000):
        adam_optimizer.step()
    plot_3dpath(
        func.function,
        adam_optimizer.path,
        func.bounds[0],
        func.bounds[1],
        f"Adam_{func.name}",
    )
