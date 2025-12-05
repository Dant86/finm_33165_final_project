"""Generate a fake portfolio."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import constants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to save the output CSV file.",
        default="../data",
    )

    return parser.parse_args()


def main() -> None:
    np.random.seed(0)

    args = parse_args()

    dates = pd.date_range(
        start="2023-01-01",
        end="2025-12-04",
    )

    df = pd.DataFrame({"date": dates})
    df["portfolio_value"] = 1_000 * np.cumprod(
        1 + 0.001 * np.random.randn(len(dates))
    )
    w = np.random.rand(len(dates), 4)
    w /= w.sum(axis=1, keepdims=True)

    for i, a in enumerate(constants.TICKERS):
        df[f"weight_{a}"] = w[:, i]

    df.to_parquet(args.output_dir + "/positions.parquet")


if __name__ == "__main__":
    main()
