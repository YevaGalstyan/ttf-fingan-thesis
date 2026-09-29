"""Slide 12: construction of the options panel as
horizontal bars, from the joined panel to the
filtered panel used for pricing.

Values from Sec. 4.1.6 of the thesis.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent))
from paths import SLIDE_DIR

OUT = SLIDE_DIR / "slide12_options_funnel.png"
FONT = "Arial"

DARK = "#1F2A24"
GREY = "#6B7280"

# step, records, contracts, bar colour
STEPS = [
    ("Joined options panel",
     5_615_794, 20_252, "#DCEBCD"),
    ("Restricted to the test split",
     1_763_210, 9_144, "#8DB36B"),
    ("Filtered\n(expiry day, missing front\n"
     "month, moneyness 0.5–2.0)",
     1_250_980, 7_208, "#4E8A2A"),
]

plt.rcParams.update({
    "font.family": [FONT, "Arial", "sans-serif"],
    "font.size": 15,
    "axes.edgecolor": GREY,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
})

fig, ax = plt.subplots(figsize=(10, 4.2))
fig.subplots_adjust(left=0.30, right=0.97,
                    bottom=0.06, top=0.96)

y = list(range(len(STEPS)))[::-1]  # top to bottom
xmax = STEPS[0][1]

for yi, (name, rec, con, col) in zip(y, STEPS):
    ax.barh(yi, rec, height=0.6, color=col, lw=0)
    ax.text(rec + xmax * 0.015, yi,
            f"{rec:,} records\n{con:,} contracts",
            va="center", ha="left",
            fontsize=13, color=DARK)

ax.set_yticks(y)
ax.set_yticklabels([s[0] for s in STEPS],
                   fontsize=14)
ax.set_xlim(0, xmax * 1.45)
ax.set_xticks([])
ax.spines[["top", "right", "bottom"]].set_visible(
    False)
ax.tick_params(axis="y", length=0)

fig.savefig(OUT, dpi=300, facecolor="white")
print(f"Saved {OUT}")