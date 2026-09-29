"""Slide 11: simplified target plot of the cost
function branches under configuration 8.
Skewness against excess kurtosis.

  - ForGAN, PnL*, PnL* & MSE: seed range as a box
    (mean +/- seed spread in both directions)
  - branches with SR*: one grey area, no points
  - other branches: small grey dots
  - test split: star

Values from Tables 14-16 of the thesis.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent))
from paths import SLIDE_DIR

OUT = SLIDE_DIR / "slide11_target.png"
FONT = "Arial"

DARK = "#1F2A24"
GREY = "#6B7280"
LIGHT = "#9CA3AF"
SR_BG = "#DDE0E5"

TEST = (0.92, 13.27)  # (skewness, kurtosis)

# boxes: name, (skew mean, spread),
# (kurtosis mean, spread), colour, line style
BOXES = [
    ("ForGAN", (0.54, 0.88), (10.12, 2.59),
     "#2F5E1A", "-"),
    ("PnL*", (0.93, 0.30), (12.03, 5.45),
     "#6FA83A", "--"),
    ("PnL* & MSE", (0.81, 0.46), (10.69, 2.97),
     "#A8CF7E", ":"),
]

# other branches without SR*: (skew, kurtosis),
# label offset in points
OTHERS = {
    "MSE": ((0.23, 10.13), (-8, 8)),
    "PnL* & STD": ((-0.09, 8.21), (8, 6)),
    "PnL* & MSE & STD": ((1.11, 6.78), (8, -4)),
}

# area covering the four SR* branch means
SR_CENTER = (-0.60, 8.0)

plt.rcParams.update({
    "font.family": [FONT, "Arial", "sans-serif"],
    "font.size": 15,
    "axes.edgecolor": GREY,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
})

fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(left=0.10, right=0.98,
                    bottom=0.13, top=0.97)

ax.axvline(0, color=GREY, lw=0.8, ls=":", zorder=1)

# SR* branches as one round area (a marker, so it
# stays round whatever the axis scales)
ax.plot(*SR_CENTER, "o", markersize=70,
        color=SR_BG, zorder=1)

# seed-range boxes
for name, (sk, sks), (ku, kus), col, ls in BOXES:
    ax.add_patch(Rectangle(
        (sk - sks, ku - kus), 2 * sks, 2 * kus,
        facecolor=col, alpha=0.12, lw=0, zorder=2))
    ax.add_patch(Rectangle(
        (sk - sks, ku - kus), 2 * sks, 2 * kus,
        fill=False, edgecolor=col, lw=2,
        linestyle=ls, zorder=3, label=name))

# other branches: small grey dots
for name, ((sk, ku), (dx, dy)) in OTHERS.items():
    ax.plot(sk, ku, "o", color=LIGHT,
            markersize=6, zorder=3)
    ax.annotate(name, (sk, ku), xytext=(dx, dy),
                textcoords="offset points",
                ha="right" if dx < 0 else "left",
                fontsize=12, color=GREY)

# the target
ax.plot(*TEST, marker="*", markersize=22,
        color=DARK, zorder=6)

ax.set_xlim(-1.2, 2.0)
ax.set_ylim(4, 19)
ax.set_xlabel("Skewness")
ax.set_ylabel("Excess kurtosis")
# legend: the three boxes, the SR* area, the star
handles, labels = ax.get_legend_handles_labels()
handles += [
    Line2D([], [], marker="o", ls="", markersize=14,
           color=SR_BG),
    Line2D([], [], marker="*", ls="", markersize=16,
           color=DARK),
]
labels += ["With SR*", "Test split"]
ax.legend(handles, labels, frameon=False,
          fontsize=13, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#E5E7EB", lw=0.8)
ax.set_axisbelow(True)

fig.savefig(OUT, dpi=300, facecolor="white")
print(f"Saved {OUT}")