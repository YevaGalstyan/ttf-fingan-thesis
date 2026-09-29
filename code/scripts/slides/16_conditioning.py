"""
Slide 16: response of G to the market state, in
the colours of the slides.

Plot only: reads conditioning_check.csv, written
by check_conditioning.py, so the rollouts are not
run again. Run check_conditioning.py first if the
CSV does not exist yet.

Prints the two numbers for the slide: the
correlation with the condition-window spread and
the number of dates on which G lies below the
market.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent))
from paths import OUT_DIR, SLIDE_DIR

OUT = SLIDE_DIR / "slide16_conditioning.png"
FONT = "Arial"

DARK = "#1F2A24"
GREEN = "#4E8A2A"
SHADE = "#70B840"
GREY = "#6B7280"
BOOT = "#4A5568"

plt.rcParams.update({
    "font.family": [FONT, "Arial", "sans-serif"],
    "font.size": 15,
    "axes.edgecolor": GREY,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
})


def main():
    g = pd.read_csv(OUT_DIR / "conditioning_check.csv",
                    parse_dates=["valuation"],
                    index_col="valuation")
    boot = pd.read_csv(
        OUT_DIR / "prices_bootstrap_block.csv",
        parse_dates=["valuation"])
    boot = (boot.groupby("valuation")["sd_ann"]
            .first().reindex(g.index))

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.subplots_adjust(left=0.07, right=0.99,
                        bottom=0.16, top=0.97)
    x = g.index

    # G: mean over seeds, band = mean +/- seed spread
    ax.fill_between(x, g["sd_ann"] - g["sd_ann_sd"],
                    g["sd_ann"] + g["sd_ann_sd"],
                    color=SHADE, alpha=0.25, lw=0,
                    label="G, seed range", zorder=1)
    ax.plot(x, g["sd_ann"], color=GREEN, lw=2.4,
            marker="o", ms=6, label="G", zorder=4)
    ax.plot(x, boot.values, color=BOOT, lw=1.8,
            ls="-.", marker="D", ms=5,
            label="Bootstrap, block", zorder=3)
    ax.plot(x, g["sigma_mkt"], color=DARK, lw=2.4,
            marker="o", ms=6, label="Market",
            zorder=5)

    ax.set_ylabel("Annualized volatility")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(
        mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b %Y"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    ax.set_axisbelow(True)

    # legend in the order of importance
    h, lab = ax.get_legend_handles_labels()
    order = ["Market", "G", "G, seed range",
             "Bootstrap, block"]
    idx = [lab.index(o) for o in order]
    ax.legend([h[i] for i in idx], order,
              frameon=False, fontsize=14,
              loc="upper left", ncol=2)

    fig.savefig(OUT, dpi=300, facecolor="white")
    print(f"Saved {OUT}")

    corr = g["sd_ann"].corr(g["cond_sd"])
    below = int((g["sd_ann"] < g["sigma_mkt"]).sum())
    print(f"correlation with window spread: {corr:.3f}")
    print(f"G below market on {below} of {len(g)} dates")
    print(f"G range {g.sd_ann.min():.3f} to "
          f"{g.sd_ann.max():.3f}, bootstrap range "
          f"{boot.min():.3f} to {boot.max():.3f}")


if __name__ == "__main__":
    main()