from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .optimizer import Optimizer


@dataclass
class AdamConfig:
    alpha: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8


class Adam(Optimizer):
    def __init__(self, config: AdamConfig = AdamConfig(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config
        self._moments = [np.zeros(self._dimension), np.zeros(self._dimension)]

    def step(self):
        gradient = self._derivative(*self._current_point)
        self._iteration_num += 1
        self.__update_parameters_with_adam(gradient)
        self._path.append(self._current_point)

        # Report progress
        # TODO: logging
        print(
            ">%d f(%s) = %.5f"
            % (self._iteration_num, self.current_point, self.score)
        )

    def __update_parameters_with_adam(self, gradient):
        self._moments[0] = (
            self._config.betas[0] * self._moments[0]
            + (1.0 - self._config.betas[0]) * gradient
        )
        self._moments[1] = (
            self._config.betas[1] * self._moments[1]
            + (1.0 - self._config.betas[1]) * gradient**2
        )
        bias_corrected_first_moment = self._moments[0] / (
            1.0 - self._config.betas[0] ** (self._iteration_num + 1)
        )
        bias_corrected_second_moment = self._moments[1] / (
            1.0 - self._config.betas[1] ** (self._iteration_num + 1)
        )
        self._current_point = (
            self._current_point
            - self._config.alpha
            * bias_corrected_first_moment
            / (np.sqrt(bias_corrected_second_moment) + self._config.epsilon)
        )
