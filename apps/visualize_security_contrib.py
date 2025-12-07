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


def visualize_gmv_contrib(
    positions: pd.DataFrame,
    ax: plt.Axes,
) -> None:
    gmv = sc.calculate_market_value(positions)
    total_gmv = gmv.sum(axis="columns")

    ax2 = ax.twinx()

    gmv.plot(ax=ax, ylabel="GMV(\\$)")
    total_gmv.plot(
        ax=ax2,
        color="black",
        ylabel="Total GMV(\\$)",
        legend=False,
        linewidth=3,
    )

    ax.set_title("GMV Contribution by Security")


def visualize_pnl_contrib(
    positions: pd.DataFrame,
    stock_rets: pd.DataFrame,
    ax: plt.Axes,
) -> None:
    pnl = sc.calculate_cum_pnl_contrib(positions, stock_rets)
    total_pnl = pnl.sum(axis="columns")

    ax2 = ax.twinx()

    pnl.plot(ax=ax, ylabel="PnL (\\$)")
    total_pnl.plot(
        ax=ax2,
        color="black",
        ylabel="Portfolio PnL (\\$)",
        legend=False,
        linewidth=3,
    )



def main() -> None:
    positions = pd.read_parquet("../data/positions.parquet")
    stock_rets = pd.read_parquet("../data/stock_rets.parquet")

    fig = plt.Figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    visualize_vol_contrib(positions, stock_rets, ax0)
    visualize_gmv_contrib(positions, ax1)
    visualize_pnl_contrib(positions, stock_rets, ax2)

    fig.tight_layout()
    fig.savefig("../presentation/figures/vol_contrib.png", dpi=300)


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"

    main()
