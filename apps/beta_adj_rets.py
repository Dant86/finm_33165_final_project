"""Calculate beta-adjusted returns."""

from __future__ import annotations

import argparse

import pandas as pd
import yfinance as yf

from src import constants
from src.portfolio_eval import market_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for historical data in the format YYYY-MM-DD.",
        default="2005-01-01",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for historical data in the format YYYY-MM-DD.",
        default=pd.Timestamp.today(tz="UTC").strftime("%Y-%m-%d"),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to save the output CSV file.",
        default="../data",
    )

    return parser.parse_args()


def get_open_prices(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = yf.Tickers(constants.TICKERS + ["SPY"])
    yf_hist = tickers.history(start=start_date, end=end_date, period=None)

    open_prices = yf_hist["Open"]

    open_prices.index = open_prices.index.rename("date")
    open_prices.columns = open_prices.columns.rename(None)

    market_price = open_prices[["SPY"]].rename(columns={"SPY": "return"})

    return open_prices.loc[:, constants.TICKERS], market_price


def main() -> None:
    args = parse_args()
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)

    open_prices, market_price = get_open_prices(start_date, end_date)

    stock_returns = market_analysis.calculate_returns(open_prices)
    market_returns = market_analysis.calculate_returns(market_price)

    market_adjusted_returns = market_analysis.calculate_market_adjusted_returns(
        stock_returns,
        market_returns,
    ).dropna()

    market_adjusted_returns.to_parquet(args.output_dir + "/beta_adj_rets.parquet")
    stock_returns.to_parquet(args.output_dir + "/stock_rets.parquet")
    open_prices.to_parquet(args.output_dir + "/open_prices.parquet")


if __name__ == "__main__":
    main()
