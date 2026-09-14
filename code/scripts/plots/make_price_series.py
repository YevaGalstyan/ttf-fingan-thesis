"""
Front-month price series plot.

Reads the front-month settlement price series and plots it with the
train, validation and test splits shaded.

Writes the figure used in Data Split.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import OUT_DIR, FIG_DIR

# ---------------------------------------------------------------- load

front = pd.read_parquet(OUT_DIR / "front_month_full.parquet")
front = front.dropna(subset=["front_price"]).reset_index(drop=True)

dates = front["date"].to_numpy()
prices = front["front_price"].to_numpy()

# ---------------------------------------------------------------- splits

n = len(prices)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

train_end = dates[n_train - 1]
val_end = dates[n_train + n_val - 1]

# ------------------------------------------------------------------ plot

fig, ax = plt.subplots(figsize=(10, 3.2))

ax.axvspan(dates[0], train_end, color="tab:blue", alpha=0.08, label="train")
ax.axvspan(train_end, val_end, color="tab:orange", alpha=0.08, label="validation")
ax.axvspan(val_end, dates[-1], color="tab:green", alpha=0.08, label="test")

ax.plot(dates, prices, color="black", linewidth=0.4)

pad = (dates[-1] - dates[0]) * 0.05
ax.set_xlim(dates[0] - pad, dates[-1] + pad)
ax.set_xlabel("Date")
ax.set_ylabel("EUR/MWh")
ax.legend(loc="upper left", frameon=False)

fig.tight_layout()
fig.savefig(FIG_DIR / "ttf_price_split.pdf")