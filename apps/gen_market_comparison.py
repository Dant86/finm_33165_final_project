"""Generate a series of visualizations comparing portfolio performance to that of
the broader market.
"""

from __future__ import annotations

import argparse

import pandas as pd
from matplotlib import pyplot as plt

from src import constants
from src.portfolio_eval import security_contrib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=str,
        help="Path to load the input CSV file.",
        default="../data",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to save the output CSV file.",
        default="../presentation/figures",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    positions = pd.read_csv(f"{args.input_dir}/test_portfolio_weights.csv")
    positions["date"] = pd.to_datetime(positions["date"])
    equity_curve = pd.read_csv(f"{args.input_dir}/test_equity_curve.csv")
    equity_curve["date"] = pd.to_datetime(equity_curve["date"])
    positions = positions.merge(equity_curve, on="date")
    for ticker in constants.TICKERS:
        positions = positions.rename(columns={ticker: f"weight_{ticker}"})

    stock_rets = pd.read_parquet(f"{args.input_dir}/stock_rets.parquet")
    beta_adj_rets = pd.read_parquet(f"{args.input_dir}/beta_adj_rets.parquet")

    gmv = security_contrib.calculate_market_value(positions)

    beta_adj_pnl = (gmv * beta_adj_rets).sum(axis="columns", min_count=1).dropna()
    total_pnl = (gmv * stock_rets).sum(axis="columns", min_count=1).dropna()

    total_port_rets = total_pnl.cumsum().pct_change()
    beta_adj_port_rets = beta_adj_pnl.cumsum().pct_change()

    beta_adj_port_sharpe = beta_adj_port_rets.rolling(126).mean() / beta_adj_port_rets.rolling(126).std()
    total_port_sharpe = total_port_rets.rolling(126).mean() / total_port_rets.rolling(126).std()

    fig, ax = plt.subplots(1, 2, figsize=(17, 11))

    ax[0].plot(total_pnl.cumsum(), label="Total", c="black")
    ax[1].plot(total_port_sharpe, label="Total", c="black")
    ax[0].plot(beta_adj_pnl.cumsum(), label="Beta-adjusted", c="blue")
    ax[1].plot(beta_adj_port_sharpe, label="Beta-adjusted", c="blue")

    ax[0].set_xlabel("date")
    ax[1].set_xlabel("date")
    ax[0].set_ylabel("PnL (\\$)")
    ax[1].set_ylabel("Sharpe Ratio")

    ax[0].set_title("Total v/s Beta-adjusted PnL")
    ax[1].set_title("Total v/s Beta-adjusted Sharpe Ratio (126 days)")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1), ncol=1)

    fig.tight_layout()
    fig.savefig(f"{args.output_dir}/market_comparison.png", dpi=300)


if __name__ == "__main__":
    main()
