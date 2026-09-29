"""Slide 8: daily TTF log returns with the
training, validation, and test splits.

Same size and margins as the slide 3 plots.
"""
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent))
from paths import TFM_CSV, SLIDE_DIR

OUT = SLIDE_DIR / "slide08_returns_splits.png"
FONT = "Arial"

SIZE = (14, 4.2)
MARGINS = dict(left=0.10, right=0.98,
               bottom=0.14, top=0.80)

DARK = "#1F2A24"
GREY = "#6B7280"

# split boundaries from Sec. 4.1.4
# name, start, end, shade, label alignment
SPLITS = [
    ("Training", "2010-01-05", "2023-01-18",
     None, "center"),
    ("Validation", "2023-01-19", "2024-09-04",
     "#F1F7EC", "right"),
    ("Test", "2024-09-05", "2026-04-24",
     "#DCEBCD", "left"),
]

plt.rcParams.update({
    "font.family": [FONT, "Arial", "sans-serif"],
    "font.size": 16,
    "axes.edgecolor": GREY,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
})

QUERY = f"""
    SELECT date, AdjClose AS price
    FROM read_csv_auto('{TFM_CSV}')
    ORDER BY date
"""
con = duckdb.connect()
df = con.execute(QUERY).df()
con.close()
df["date"] = pd.to_datetime(df["date"])
df["ret"] = np.log(df["price"]).diff()
df = df.dropna(subset=["ret"])

fig, ax = plt.subplots(figsize=SIZE)
fig.subplots_adjust(**MARGINS)

gap = pd.Timedelta(days=30)
for name, start, end, color, ha in SPLITS:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if color:
        ax.axvspan(s, e, color=color, lw=0)
    n = int(((df["date"] >= s)
             & (df["date"] <= e)).sum())
    x = {"center": s + (e - s) / 2,
         "right": e - gap,
         "left": s + gap}[ha]
    ax.text(x, 1.02, f"{name}\n{n:,} returns",
            transform=ax.get_xaxis_transform(),
            ha=ha, va="bottom",
            color=DARK, fontsize=13)

ax.plot(df["date"], df["ret"],
        color=DARK, lw=0.7)

ax.set_ylim(-0.45, 0.45)
ax.yaxis.set_major_formatter(
    mticker.PercentFormatter(1.0, decimals=0))
ax.set_ylabel("Daily log return")
ax.xaxis.set_major_locator(
    mdates.YearLocator(2))
ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y"))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E5E7EB", lw=0.8)
ax.set_axisbelow(True)

fig.savefig(OUT, dpi=300, facecolor="white")
print(f"Saved {OUT}; returns: {len(df)}")