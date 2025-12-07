"""Visualize security contribution."""

from __future__ import annotations

import argparse

from matplotlib import pyplot as plt
import pandas as pd

from src.portfolio_eval import security_contrib as sc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to save the output CSV file.",
        default="../presentation/figures",
    )

    return parser.parse_args()


def visualize_vol_contrib(
    positions: pd.DataFrame,
    stock_rets: pd.DataFrame,
    ax: plt.Axes,
) -> None:
    port_vol = sc.calculate_rolling_volatility(positions, stock_rets)
    vol_ratio = sc.calculate_rolling_volatility_ratio(positions, stock_rets)

    ax2 = ax.twinx()

    vol_ratio.plot(ax=ax, ylabel="Volatility Ratio")
    port_vol.plot(
        ax=ax2,
        color="black",
        ylabel="Portfolio Volatility (\\$)",
        legend=False,
        linewidth=3,
    )

    ax.set_title("Volatility Contribution by Security")


def visualize_pnl_contrib(
    positions: pd.DataFrame,
    stock_rets: pd.DataFrame,
    ax: plt.Axes,
) -> None:
    del positions, stock_rets, ax  # TODO: Make this chart.


def main() -> None:
    positions = pd.read_parquet("../data/positions.parquet")
    stock_rets = pd.read_parquet("../data/stock_rets.parquet")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(17, 11),
    )

    visualize_vol_contrib(
        positions, stock_rets, axes[0, 0]
    )

    visualize_pnl_contrib(
        positions,
        stock_rets,
        axes[0, 1],
    )

    fig.tight_layout()
    fig.savefig("../presentation/figures/vol_contrib.png", dpi=300)


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"

    main()
