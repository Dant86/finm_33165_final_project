import gym
import numpy as np
import pandas as pd
from gym import spaces


class PortfolioRebalanceEnv(gym.Env):
    """
    Portfolio Rebalancing Environment with:
    - log-return reward
    - transaction cost penalty
    - volatility penalty
    - concentration penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_df: pd.DataFrame,
        action_weights: np.ndarray,
        window: int = 20,
        initial_cash: float = 1_000_000.0,
        cost_rate: float = 0.00001,      # turnover penalty strength
        lambda_vol: float = 0.05,      # risk penalty
        lambda_conc: float = 0.05,     # concentration penalty
    ):
        super().__init__()

        # --------------------------
        # Basic settings
        # --------------------------
        self.price_df = price_df.dropna().astype(float)
        self.assets = list(self.price_df.columns)
        self.n_assets = len(self.assets)
        self.action_weights = action_weights
        self.K = action_weights.shape[0]

        self.window = window
        self.initial_cash = float(initial_cash)

        self.cost_rate = cost_rate
        self.lambda_vol = lambda_vol
        self.lambda_conc = lambda_conc

        # --------------------------
        # Build returns matrix
        # --------------------------
        self.prices = self.price_df.values
        self.returns = self.prices[1:] / self.prices[:-1] - 1.0
        self.T = len(self.returns)

        # State dimension
        self.state_dim = (self.n_assets * 4) + 1
        self.action_space = spaces.Discrete(self.K)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self.reset()

    # -------------------------------------------------------
    # Build state vector
    # -------------------------------------------------------
    def _compute_state(self, t: int):
        past_returns = self.returns[t - self.window : t]
        mean_ret = past_returns.mean(axis=0)
        vol_ret = past_returns.std(axis=0) + 1e-8
        last_ret = self.returns[t - 1]

        state = np.concatenate(
            [
                last_ret,
                mean_ret,
                vol_ret,
                self.weights,
                np.array([self.portfolio_value / self.initial_cash]),
            ]
        )
        return state.astype(np.float32)

    # -------------------------------------------------------
    # Reset environment
    # -------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.t = self.window
        self.portfolio_value = self.initial_cash
        self.weights = np.zeros(self.n_assets, dtype=np.float32)

        obs = self._compute_state(self.t)
        self._last_obs = obs

        return obs, {}

    # -------------------------------------------------------
    # Step function
    # -------------------------------------------------------
    def step(self, action: int):
        assert self.action_space.contains(action)

        old_weights = self.weights.copy()
        new_weights = self.action_weights[action]
        self.weights = new_weights

        # -------------------------------------------------
        # 1) Portfolio return & log-return reward
        # -------------------------------------------------
        asset_rets = self.returns[self.t]
        port_ret = np.dot(new_weights, asset_rets)
        log_ret = np.log1p(port_ret)

        # update portfolio value
        self.portfolio_value *= (1.0 + port_ret)

        # -------------------------------------------------
        # 2) Transaction cost (turnover-based)
        # -------------------------------------------------
        turnover = np.sum(np.abs(new_weights - old_weights))
        cost_penalty = self.cost_rate * turnover

        # -------------------------------------------------
        # 3) Portfolio volatility penalty  **FIXED**
        # -------------------------------------------------
        if self.t >= self.window + 1:
            past_returns = self.returns[self.t - self.window : self.t]   # shape (window, N)
            # compute realized portfolio returns over window
            port_hist = np.dot(past_returns, new_weights)
            vol_window = np.std(port_hist)
        else:
            vol_window = 0.0

        vol_penalty = self.lambda_vol * vol_window

        # -------------------------------------------------
        # 4) Concentration penalty (HHI)  **FIXED**
        # -------------------------------------------------
        hhi = np.sum(new_weights ** 2)
        baseline = 1.0 / self.n_assets
        conc_penalty = self.lambda_conc * (hhi - baseline)

        # -------------------------------------------------
        # 5) Final reward
        # -------------------------------------------------
        reward = log_ret - cost_penalty - vol_penalty - conc_penalty

        # move time
        self.t += 1
        terminated = self.t >= self.T
        truncated = False

        if not terminated:
            obs = self._compute_state(self.t)
        else:
            obs = self._last_obs

        self._last_obs = obs

        return obs, reward, terminated, truncated, {
            "portfolio_value": float(self.portfolio_value),
            "turnover": float(turnover),
            "log_ret": float(log_ret),
            "vol_penalty": float(vol_penalty),
            "conc_penalty": float(conc_penalty),
            "cost_penalty": float(cost_penalty),
            "hhi": float(hhi),
        }
