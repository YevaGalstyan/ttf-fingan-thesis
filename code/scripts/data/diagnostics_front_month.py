"""
Diagnostics for the front-month TTF series.

Reports the two figures quoted in the thesis:
  - how often the settlement price falls outside the intraday high-low range
  - how contract transition days affect the skewness and kurtosis of returns

Run build_front_month.py first.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from paths import FUTURES_DIR, OUT_DIR

ohlc = pd.read_parquet(FUTURES_DIR / "TFM_ohlc.parquet")
front = pd.read_parquet(OUT_DIR / "front_month_full.parquet")

# The settlement price is not a traded price: it lies outside the day's own
# high-low range on a substantial fraction of records.
rng = ohlc.dropna(subset=["px_last", "px_high", "px_low"])
outside = rng[(rng.px_last > rng.px_high) | (rng.px_last < rng.px_low)]
dev = np.maximum(outside.px_last - outside.px_high, outside.px_low - outside.px_last)

print(
    f"\nsettlement outside high-low range: {len(outside)} of {len(rng)} "
    f"({len(outside) / len(rng):.1%}), "
    f"median deviation {dev.median():.2f}, max {dev.max():.2f} EUR/MWh"
)

# Contract transitions inflate the skewness of the return series but add
# little to its tail weight.
print()
print(front.groupby("switch")["ret"].std().round(3).to_string())
print()
for name, arr in [
    ("all", front["ret"].dropna()),
    ("roll", front.loc[front.switch, "ret"].dropna()),
    ("non-roll", front.loc[~front.switch, "ret"].dropna()),
]:
    print(
        f"{name:10s} n={len(arr):5d} "
        f"skew={skew(arr):.3f} kurt={kurtosis(arr):.3f}"
    )