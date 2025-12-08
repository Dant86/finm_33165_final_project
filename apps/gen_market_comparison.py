"""Generate a series of visualizations comparing portfolio performance to that of
the broader market.
"""

from __future__ import annotations

import argparse

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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
        "--portfolio-fname",
        type=str,
        help="Path to load the input CSV file.",
        default="sac_weights_history.parquet",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to save the output PNG file.",
        default="../presentation/figures",
    )

    parser.add_argument(
        "--output-fname",
        type=str,
        help="Path to save the output PNG file.",
        default="sac_market_comparison.png",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    positions = pd.read_parquet(f"{args.input_dir}/{args.portfolio_fname}")

    stock_rets = pd.read_parquet(f"{args.input_dir}/stock_rets.parquet")
    beta_adj_rets = pd.read_parquet(f"{args.input_dir}/beta_adj_rets.parquet")

    gmv = security_contrib.calculate_market_value(positions)

    beta_adj_pnl = (gmv * beta_adj_rets).sum(axis="columns", min_count=1).dropna()
    total_pnl = (gmv * stock_rets).sum(axis="columns", min_count=1).dropna()

    total_port_rets = total_pnl.cumsum().pct_change()
    beta_adj_port_rets = beta_adj_pnl.cumsum().pct_change()

    beta_adj_port_sharpe = (
        beta_adj_port_rets.rolling(126).mean()
        / beta_adj_port_rets.rolling(126).std()
        / np.sqrt(252)
    )
    total_port_sharpe = (
        total_port_rets.rolling(126).mean()
        / total_port_rets.rolling(126).std()
        / np.sqrt(252)
    )

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
    fig.legend(handles, labels, loc="lower center", ncol=2)

    fig.tight_layout()
    fig.savefig(f"{args.output_dir}/{args.output_fname}", dpi=300)

    max_dd_total = (
        total_port_rets.cumsum() - total_port_rets.cumsum().cummax()
    ).min() * -1
    calmar_total = total_port_rets.cumsum().iloc[-1] / np.sqrt(252) / max_dd_total

    max_dd_ba = (
        beta_adj_port_rets.cumsum() - beta_adj_port_rets.cumsum().cummax()
    ).min() * -1
    calmar_ba = beta_adj_port_rets.cumsum().iloc[-1] / np.sqrt(252) / max_dd_ba

    df = pd.DataFrame(
        [
            {
                "Space": "Total",
                "PnL (\\$)": total_pnl.sum(),
                "Max Drawdown (\\%)": max_dd_total,
                "Calmar Ratio": calmar_total,
                "Sharpe Ratio": total_port_rets.mean()
                / total_port_rets.std()
                / np.sqrt(252),
            },
            {
                "Space": "Beta-adjusted",
                "PnL (\\$)": beta_adj_pnl.sum(),
                "Max Drawdown (\\%)": max_dd_ba,
                "Calmar Ratio": calmar_ba,
                "Sharpe Ratio": beta_adj_port_rets.mean()
                / beta_adj_port_rets.std()
                / np.sqrt(252),
            },
        ]
    )

    df = df.set_index("Space").T
    print(df.to_latex(float_format="%.3f"))


if __name__ == "__main__":
    main()
