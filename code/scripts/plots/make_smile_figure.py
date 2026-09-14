"""
Figure: implied volatility smiles on three valuation dates.

One panel per valuation date. Each panel shows the implied volatility
published by the exchange and the implied volatility of the simulated
prices of GBM, the two bootstrap modes and G, against moneyness.

G was priced with five seeds. The line is the mean across the seeds at each
strike, and the band spans the lowest and highest seed. Strikes with no
implied volatility (Sec. 4.6) are missing from the curves, so the curves do
not all cover the full moneyness band.

The sigma in each panel title is the implied volatility GBM was priced
with, read from the GBM output.

Usage:
    python make_smile_figure.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import OUT_DIR, FIG_DIR

DATES = ["2025-10-28", "2025-01-24", "2026-03-25"]

# Label, file, colour, line style.
BENCHMARKS = [
    ("GBM",              "prices_gbm_atm.csv",          "#888888", "--"),
    ("Bootstrap, iid",   "prices_bootstrap_iid.csv",    "#4c72b0", ":"),
    ("Bootstrap, block", "prices_bootstrap_block.csv",  "#55a868", "-."),
]
GENERATOR = ("$G$", "prices_ForGAN_c9.csv", "#c44e52")
MARKET_COLOR = "#000000"


def read(filename):
    return pd.read_csv(OUT_DIR / filename, parse_dates=["valuation"])


def main():
    benchmarks = [(lab, read(f), c, ls) for lab, f, c, ls in BENCHMARKS]
    gen = read(GENERATOR[1])
    gbm = benchmarks[0][1]

    fig, axes = plt.subplots(1, len(DATES), figsize=(11, 3.6))

    for ax, date in zip(axes, DATES):
        # Market, from any file: the columns are identical across models.
        mkt = gbm[gbm["valuation"] == date].dropna(subset=["market_iv"])
        ax.plot(mkt["moneyness"], mkt["market_iv"], color=MARKET_COLOR,
                lw=1.6, label="Market")

        for label, df, colour, ls in benchmarks:
            d = df[df["valuation"] == date].dropna(subset=["implied_vol"])
            ax.plot(d["moneyness"], d["implied_vol"], color=colour,
                    ls=ls, lw=1.3, label=label)

        # G: mean across seeds, band between the lowest and highest seed.
        g = gen[gen["valuation"] == date].dropna(subset=["implied_vol"])
        if not g.empty:
            by_strike = g.groupby("moneyness")["implied_vol"]
            mean, lo, hi = by_strike.mean(), by_strike.min(), by_strike.max()
            ax.fill_between(mean.index, lo, hi, color=GENERATOR[2],
                            alpha=0.18, lw=0)
            ax.plot(mean.index, mean.values, color=GENERATOR[2],
                    lw=1.5, label=GENERATOR[0])

        # The sigma GBM was priced with on this date.
        sigma = gbm.loc[gbm["valuation"] == date, "sigma"].iloc[0]
        ax.set_title(f"{date}\n$\\sigma = {sigma:.3f}$", fontsize=9.5,
                     color="#333333", linespacing=1.4)
        ax.set_xlabel("Moneyness")
        ax.set_xlim(0.5, 2.0)
        ax.grid(alpha=0.25, lw=0.5)

    axes[0].set_ylabel("Implied volatility")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout()
    out_path = FIG_DIR / "smiles.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print("panel sigmas:")
    for date in DATES:
        s = gbm.loc[gbm["valuation"] == date, "sigma"].iloc[0]
        print(f"  {date}  {s:.3f}")
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()