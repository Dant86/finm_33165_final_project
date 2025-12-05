"""Metrics to answer 'is this portfolio outperforming the market'?"""

from __future__ import annotations

import pandas as pd


def calculate_returns(
    open_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate daily returns.

    Args:
        open_prices: A ``DataFrame`` of open prices with index ["date"] and
            columns [tickers].

    Returns:
        A ``DataFrame`` of daily returns with index ["date"] and columns
        [tickers].
    """
    return open_prices.pct_change().dropna()


def calculate_rolling_beta(
    stock_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
    window_size: int = 126,
) -> pd.DataFrame:
    """Calculate rolling beta (with intercept).

    Args:
        stock_returns: A ``DataFrame`` of stock returns with index ["date"]
            and columns [tickers].
        market_returns: A ``DataFrame`` of market returns with index ["date"]
            and columns ["return"].
        window_size: Rolling window size in days. Defaults to 126.

    Returns:
        A ``DataFrame`` of rolling betas with index ["date"] and
        columns [tickers].
    """
    cov = stock_returns.rolling(window_size).cov(market_returns["return"])
    var = stock_returns.rolling(window_size).var()

    return cov / var


def calculate_market_adjusted_returns(
    stock_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate market-adjusted returns.

    Args:
        stock_returns: A ``DataFrame`` of stock returns with index ["date"]
            and columns [tickers].
        market_returns: A ``DataFrame`` of market returns with index ["date"]
            and columns ["return"].

    Returns:
        A ``DataFrame`` of market-adjusted returns with index ["date"] and
        columns [tickers].
    """
    beta = calculate_rolling_beta(stock_returns, market_returns)

    return stock_returns - beta.multiply(market_returns["return"], axis="index")
