from abc import abstractmethod
from typing import Callable, List, Optional, Tuple
import numpy as np

from .helpers import central_gradient


# TODO: assertions for dimenson
class Optimizer:
    def __init__(
        self,
        objective: Callable[[Tuple[float, ...]], float],
        bounds: List[Tuple[float, float]],
        derivative: Optional[
            Callable[[Tuple[float, ...]], Tuple[float, ...]]
        ] = None,
        start_point: np.NDArray = None,
    ):
        self._objective = objective
        self._derivative = (
            derivative
            if derivative is not None
            else lambda *args: central_gradient(
                f=self._objective, arr=np.array(list(args))
            )
        )
        self._bounds = bounds
        self._dimension: int = len(bounds)
        self._iteration_num: int = 0
        self._current_point = (
            bounds[:, 0]
            + np.random.rand(self._dimension) * (bounds[:, 1] - bounds[:, 0])
            if start_point is None
            else start_point
        )
        self._path = [self._current_point]

    @property
    def path(self) -> List[Tuple[float, ...]]:
        return self._path

    @property
    def score(self) -> float:
        return self._objective(*self._current_point)

    @property
    def current_point(self) -> np.NDArray:
        return self._current_point

    @abstractmethod
    def step():
        pass
