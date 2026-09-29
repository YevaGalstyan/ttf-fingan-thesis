"""
Slide 15: implied volatility smiles on three
valuation dates, in the colours of the slides.

Saves one file per date, all of the same size and
margins, plus one legend file to place underneath:
  slide15_smile_<date>.png
  slide15_smile_legend.png

No panel titles: the date and sigma go on the
slide. The sigmas are printed at the end.

G was priced with five seeds. The line is the mean
across the seeds at each strike, and the band spans
the lowest and highest seed.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent))
from paths import OUT_DIR, SLIDE_DIR

DATES = ["2025-10-28", "2025-01-24", "2026-03-25"]

FONT = "Arial"
SIZE = (5.6, 4.6)  # same for all three files
MARGINS = dict(left=0.17, right=0.95,
               bottom=0.16, top=0.97)

DARK = "#1F2A24"
GREEN = "#4E8A2A"
SHADE = "#70B840"
GREY = "#6B7280"

# label, file, colour, line style
BENCHMARKS = [
    ("GBM", "prices_gbm_atm.csv",
     "#9CA3AF", "--"),
    ("Bootstrap, iid", "prices_bootstrap_iid.csv",
     "#6B7280", ":"),
    ("Bootstrap, block",
     "prices_bootstrap_block.csv",
     "#4A5568", "-."),
]
GENERATOR = ("G", "prices_ForGAN_c9.csv")

plt.rcParams.update({
    "font.family": [FONT, "Arial", "sans-serif"],
    "font.size": 15,
    "axes.edgecolor": GREY,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
})


def read(filename):
    return pd.read_csv(OUT_DIR / filename,
                       parse_dates=["valuation"])


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E5E7EB", lw=0.8)
    ax.set_axisbelow(True)


def main():
    benchmarks = [(lab, read(f), c, ls)
                  for lab, f, c, ls in BENCHMARKS]
    gen = read(GENERATOR[1])
    gbm = benchmarks[0][1]

    for date in DATES:
        fig, ax = plt.subplots(figsize=SIZE)
        fig.subplots_adjust(**MARGINS)

        # market: identical columns in every file
        mkt = gbm[gbm["valuation"] == date].dropna(
            subset=["market_iv"])
        ax.plot(mkt["moneyness"], mkt["market_iv"],
                color=DARK, lw=2.4, zorder=5,
                label="Market")

        for label, df, colour, ls in benchmarks:
            d = df[df["valuation"] == date].dropna(
                subset=["implied_vol"])
            ax.plot(d["moneyness"], d["implied_vol"],
                    color=colour, ls=ls, lw=1.6,
                    zorder=3, label=label)

        # G: mean across seeds, band between the
        # lowest and highest seed
        g = gen[gen["valuation"] == date].dropna(
            subset=["implied_vol"])
        if not g.empty:
            by_k = g.groupby("moneyness")["implied_vol"]
            mean = by_k.mean()
            ax.fill_between(mean.index, by_k.min(),
                            by_k.max(), color=SHADE,
                            alpha=0.25, lw=0, zorder=2)
            ax.plot(mean.index, mean.values,
                    color=GREEN, lw=2.2, zorder=4,
                    label=GENERATOR[0])

        ax.set_xlim(0.5, 2.0)
        ax.set_xticks([0.5, 1.0, 1.5, 2.0])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Moneyness")
        ax.set_ylabel("Implied volatility")
        style(ax)

        out = SLIDE_DIR / f"slide15_smile_{date}.png"
        fig.savefig(out, dpi=300, facecolor="white")
        print(f"Saved {out}")

        handles, labels = ax.get_legend_handles_labels()
        plt.close(fig)

    # legend as its own file, one row
    handles.append(plt.Rectangle(
        (0, 0), 1, 1, facecolor=SHADE,
        alpha=0.25, lw=0))
    labels.append("G, seed range")
    fig = plt.figure(figsize=(14, 0.6))
    fig.legend(handles, labels, loc="center",
               ncol=len(labels), frameon=False,
               fontsize=14)
    out = SLIDE_DIR / "slide15_smile_legend.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")

    print("\nsigma per date (for the slide):")
    for date in DATES:
        s = gbm.loc[gbm["valuation"] == date,
                    "sigma"].iloc[0]
        print(f"  {date}  {s:.3f}")


if __name__ == "__main__":
    main()