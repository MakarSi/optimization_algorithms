import numpy as np
import pytest

from src.algorithms.adam import Adam, AdamConfig


def test_adam():
    objective = lambda x, y: x**2.0 + y**2.0
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])

    adam_optimizer = Adam(
        objective=objective,
        bounds=bounds,
        config=AdamConfig(alpha=0.01),
    )
    for _ in range(1000):
        adam_optimizer.step()
    assert adam_optimizer.score == pytest.approx(0.0, abs=1e-5, rel=1e-5)
