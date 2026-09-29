"""Two plots of the same size, made to be stacked
on one slide:
  1. density of daily TTF log returns (filled
     curve) against the normal bell curve
  2. 21-day rolling annualized volatility with
     the constant volatility of Black-76

Both use the same margins, so the plot areas
line up exactly when placed under each other.
"""
import math
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

OUT_DIST = SLIDE_DIR / "slide_return_dist.png"
OUT_VOL = SLIDE_DIR / "slide_annual_vol.png"
FONT = "Arial"
WINDOW = 21  # trading days, about one month

SIZE = (14, 4.2)  # same for both plots
MARGINS = dict(left=0.10, right=0.98,
               bottom=0.20, top=0.95)

DARK = "#1F2A24"
GREEN = "#4E8A2A"
SHADE = "#70B840"
GREY = "#6B7280"

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

ret = df["ret"].dropna().to_numpy()
mu, sd, n = ret.mean(), ret.std(ddof=1), len(ret)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    ax.set_axisbelow(True)


# ---------- plot 1: return density ----------
x = np.linspace(-0.4, 0.4, 801)

# smooth TTF density (Gaussian kernel,
# Silverman bandwidth x SMOOTH)
SMOOTH = 1.5  # higher = smoother tails
h = SMOOTH * 1.06 * sd * n ** (-1 / 5)
z = (x[:, None] - ret[None, :]) / h
kde = (np.exp(-0.5 * z ** 2).sum(axis=1)
       / (n * h * math.sqrt(2 * math.pi)))

normal = (np.exp(-0.5 * ((x - mu) / sd) ** 2)
          / (sd * math.sqrt(2 * math.pi)))

FLOOR = 0.01
fig, ax = plt.subplots(figsize=SIZE)
fig.subplots_adjust(**MARGINS)

ax.fill_between(x, FLOOR, kde,
                where=kde > FLOOR,
                color=SHADE, alpha=0.35, lw=0)
ax.plot(x, np.where(kde > FLOOR, kde, np.nan),
        color=GREEN, lw=1.6,
        label="TTF daily returns")
ax.plot(x, np.where(normal > FLOOR,
                    normal, np.nan),
        color=DARK, lw=2.2,
        label="Normal distribution (Black-76)")

ax.set_yscale("log")
ax.set_ylim(FLOOR, 30)
ax.yaxis.set_major_locator(
    mticker.FixedLocator([0.01, 0.1, 1, 10]))
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlim(-0.4, 0.4)
ax.xaxis.set_major_formatter(
    mticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Daily log return")
ax.set_ylabel("Density\n(log scale)")
ax.legend(frameon=False, fontsize=14,
          loc="upper left")
style(ax)

fig.savefig(OUT_DIST, dpi=300, facecolor="white")
plt.close(fig)

# ---------- plot 2: annual volatility ----------
df["vol"] = (df["ret"].rolling(WINDOW).std()
             * np.sqrt(252) * 100)
const_vol = sd * math.sqrt(252) * 100

fig, ax = plt.subplots(figsize=SIZE)
fig.subplots_adjust(**MARGINS)

for start, end in [("2021-01-01", "2022-12-31"),
                   ("2026-02-01", "2026-04-30")]:
    ax.axvspan(pd.Timestamp(start),
               pd.Timestamp(end),
               color=SHADE, alpha=0.15, lw=0)
ax.plot(df["date"], df["vol"],
        color=GREEN, lw=1.4)
ax.axhline(const_vol, color=DARK,
           lw=1.4, ls="--")
ax.annotate("constant volatility (Black-76)",
            xy=(df["date"].iloc[0], const_vol),
            xytext=(4, 6),
            textcoords="offset points",
            va="bottom", color=DARK, fontsize=14,
            bbox=dict(facecolor="white",
                      edgecolor="none", pad=1))

vmax = df["vol"].max()
ax.set_ylim(0, (vmax // 50 + 1) * 50)
ax.set_ylabel("Annualized\nvolatility (%)")
ax.xaxis.set_major_locator(
    mdates.YearLocator(2))
ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y"))
style(ax)

fig.savefig(OUT_VOL, dpi=300, facecolor="white")
plt.close(fig)

print(f"Saved {OUT_DIST}")
print(f"Saved {OUT_VOL}")
print(f"n={n}, std={sd:.4f}, "
      f"constant vol {const_vol:.1f} %, "
      f"max vol {vmax:.0f} %")