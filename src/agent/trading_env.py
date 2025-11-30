"""Trading enviroment."""

import gymnasium
import numpy as np
from numpy import typing as npt


class TradingEnv(gymnasium.Env[npt.NDArray, int]):
    """Trading enviroment."""

    def __init__(
        self,
        prices: np.ndarray,
        calc_window: int = 20,
        initial_cash: float = 10_000.0,
    ):
        super().__init__()

        self._prices = prices
        self._calc_window = calc_window
        self._initial_cash = initial_cash
