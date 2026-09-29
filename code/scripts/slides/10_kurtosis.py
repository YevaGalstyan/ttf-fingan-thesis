"""Slide 10: the three stylized facts for the
ForGAN baseline, saved as three separate files
of the same size and margins:
  slide10_kurtosis.png  excess kurtosis
  slide10_skewness.png  skewness
  slide10_c2.png        C2(tau), config 8 vs test
Mean +/- seed spread over five seeds.

Values from Tables 7-13 of the thesis.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent))
from paths import SLIDE_DIR

FONT = "Arial"

SIZE = (5.6, 5)  # same for all three files
MARGINS = dict(left=0.18, right=0.97,
               bottom=0.16, top=0.97)

DARK = "#1F2A24"
GREEN = "#4E8A2A"
SHADE = "#70B840"
GREY = "#6B7280"
CONF = "#6B7280"  # other configurations

SELECTED = "8"
CONFIGS = ["1", "2", "3", "4", "5", "6",
           "7", "8", "9", "10", "11"]

# (mean, seed spread) per configuration
KURT = [(0.32, 0.42), (4.51, 1.52), (4.70, 1.53),
        (7.95, 1.68), (7.04, 1.83), (3.74, 1.00),
        (7.97, 1.22), (10.12, 2.59), (13.84, 8.77),
        (7.85, 3.39), (5.55, 2.52)]
SKEW = [(-0.18, 0.39), (0.19, 0.58), (0.02, 0.68),
        (-0.09, 0.38), (0.41, 0.68), (0.15, 0.45),
        (0.04, 0.30), (0.54, 0.88), (1.78, 0.99),
        (-0.14, 0.61), (0.30, 0.36)]
TEST_KURT, TEST_SKEW = 13.27, 0.92

# C2(tau) at tau = 1, 2, 5, 10
LAGS = [1, 2, 5, 10]
C2_TEST = [0.289, 0.061, 0.094, 0.012]
C2_SEL = [(0.263, 0.201), (0.170, 0.113),
          (0.167, 0.112), (0.144, 0.103)]

plt.rcParams.update({
    "font.family": [FONT, "Arial", "sans-serif"],
    "font.size": 15,
    "axes.edgecolor": GREY,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
})


def new_fig():
    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(**MARGINS)
    return fig, ax


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    out = SLIDE_DIR / name
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def per_config(ax, values, test, ylabel):
    for i, (cfg, (m, s)) in enumerate(
            zip(CONFIGS, values)):
        col = GREEN if cfg == SELECTED else CONF
        ax.errorbar(i, m, yerr=s, fmt="o",
                    color=col, ecolor=col,
                    elinewidth=2, capsize=4,
                    markersize=8 if cfg == SELECTED
                    else 6, zorder=3)
    ax.axhline(test, color=DARK, lw=1.6, ls="--")
    ax.text(-0.5, test, "test",
            ha="left", va="bottom",
            color=DARK, fontsize=13)
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels(CONFIGS, fontsize=13)
    ax.set_xlim(-0.6, len(CONFIGS) - 0.4)
    ax.set_xlabel("Configuration")
    ax.set_ylabel(ylabel)
    style(ax)


# file 1: heavy tails
fig, ax = new_fig()
per_config(ax, KURT, TEST_KURT, "Excess kurtosis")
ax.set_ylim(0, 25)
save(fig, "slide10_kurtosis.png")

# file 2: asymmetry
fig, ax = new_fig()
per_config(ax, SKEW, TEST_SKEW, "Skewness")
ax.axhline(0, color=GREY, lw=0.8)
ax.set_ylim(-1.2, 3.0)
save(fig, "slide10_skewness.png")

# file 3: volatility clustering
fig, ax = new_fig()
m = np.array([v[0] for v in C2_SEL])
s = np.array([v[1] for v in C2_SEL])
ax.fill_between(LAGS, m - s, m + s,
                color=SHADE, alpha=0.25, lw=0)
ax.plot(LAGS, m, "o-", color=GREEN, lw=2,
        markersize=7, label="Configuration 8")
ax.plot(LAGS, C2_TEST, "o--", color=DARK, lw=1.6,
        markersize=6, label="Test split")
ax.set_xticks(LAGS)
ax.set_xlabel("Lag τ")
ax.set_ylabel("Autocorrelation")
ax.set_ylim(-0.05, 0.5)
ax.legend(frameon=False, fontsize=13,
          loc="upper right")
style(ax)
save(fig, "slide10_c2.png")