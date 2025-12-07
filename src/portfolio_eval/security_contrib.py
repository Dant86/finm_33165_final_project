"""Metrics to answer 'how much did each security contribute to the portfolio'?"""

from __future__ import annotations

import pandas as pd


def calculate_market_value(
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate gross market value per security.

    Args:
        positions: A ``DataFrame`` of portfolio positions with index ["date"] and columns
            ["portfolio_value", "weight_ticker1", ..., "weight_tickerN"].
    """
    weights = positions.filter(regex="weight_")
    weights.columns = weights.columns.str.replace("weight_", "")
    market_value = weights.multiply(positions["portfolio_value"], axis="index")
    market_value.index = positions["date"]

    return market_value


def calculate_cum_pnl_contrib(
    positions: pd.DataFrame,
    stock_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate cumulative PnL by security.

    Args:
        positions: A ``DataFrame`` of portfolio positions with index ["date"] and columns
            ["ticker", "weight_ticker1", ..., "weight_tickerN"].
        stock_returns: A ``DataFrame`` of stock returns with index ["date"] and columns
            [tickers].

    Returns:
        A ``DataFrame`` of cumulative PnL by security with index ["date"] and columns
        [tickers].
    """
    gmv = calculate_market_value(positions)

    pnl = (gmv * stock_returns).cumsum().dropna()

    return pnl


def calculate_rolling_volatility(
    positions: pd.DataFrame,
    stock_returns: pd.DataFrame,
    window_size: int = 126,
) -> pd.DataFrame:
    """Calculate rolling dollar volatility.

    Args:
        positions: A ``DataFrame`` of portfolio positions with index ["date"] and columns
            ["ticker", "weight_ticker1", ..., "weight_tickerN"].
        stock_returns: A ``DataFrame`` of stock returns with index ["date"] and columns
            [tickers].
        window_size: Rolling window size in days. Defaults to 126.

    Returns:
        A ``DataFrame`` of rolling dollar volatility with index ["date"] and columns
        ["volatility"].
    """
    gmv = calculate_market_value(positions)

    cov = stock_returns.rolling(window_size).cov().dropna()

    vol = {}

    for date in gmv.index.unique():
        if date not in cov.index.get_level_values("date"):
            continue

        cov_on_date = cov.loc[cov.index.get_level_values("date") == date].values
        gmv_on_date = gmv.loc[date].values[:,None]

        vol[date] = (gmv_on_date.T @ cov_on_date @ gmv_on_date)[0, 0] ** 0.5

    srs = pd.Series(vol).to_frame(name="volatility")
    srs.index = srs.index.rename("date")

    return pd.DataFrame(srs)


def calculate_rolling_volatility_ratio(
    positions: pd.DataFrame,
    stock_returns: pd.DataFrame,
    window_size: int = 126,
) -> pd.DataFrame:
    """Calculate rolling volatility ratio."""
    gmv = calculate_market_value(positions)

    total_vol = calculate_rolling_volatility(positions, stock_returns, window_size)


    sec_vol = stock_returns.rolling(window_size).std().dropna()
    weighted_sec_vol = gmv * sec_vol

    return weighted_sec_vol.dropna().divide(total_vol["volatility"], axis="index")
