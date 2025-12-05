"""Visualize security contribution."""

from __future__ import annotations

from matplotlib import pyplot as plt
import pandas as pd

from src.portfolio_eval import security_contrib as sc


def visualize_vol_contrib(
    positions: pd.DataFrame,
    stock_rets: pd.DataFrame,
    ax: plt.Axes,
) -> None:
    port_vol = sc.calculate_rolling_volatility(positions, stock_rets)
    vol_ratio = sc.calculate_rolling_volatility_ratio(positions, stock_rets)

    ax2 = ax.twinx()

    vol_ratio.plot(ax=ax)
    port_vol.plot(ax=ax2)


def main() -> None:
    positions = pd.read_parquet("../data/positions.parquet")
    stock_rets = pd.read_parquet("../data/stock_rets.parquet")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
    )

    visualize_vol_contrib(
        positions, stock_rets, axes[0, 0]
    )

    plt.show()


if __name__ == "__main__":
    main()
