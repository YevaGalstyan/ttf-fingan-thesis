"""
Check that condition_window picks the l returns preceding the valuation date.

Off-by-one here would silently feed the generator the wrong history.
"""

import numpy as np
import pandas as pd

from rollout import load_series, condition_window
from paths import PRICING_DATES

dates, prices, returns = load_series()
val = pd.read_csv(PRICING_DATES, parse_dates=["expiry", "valuation"])

l = 10
row = val.iloc[0]
vd = row["valuation"]

idx = int(np.searchsorted(dates, np.datetime64(vd)))

print(f"valuation date   {pd.Timestamp(vd).date()}")
print(f"index in series  {idx}")
print(f"dates[idx]       {pd.Timestamp(dates[idx]).date()}")
print(f"prices[idx]      {prices[idx]:.3f}")
print(f"F_t from csv     {row['F_t']:.3f}")
print()

cond = condition_window(returns, dates, vd, l)

print("condition window, with the dates the returns were observed on:")
for k in range(l):
    j = idx - l + k          # index into returns
    d_from = pd.Timestamp(dates[j]).date()
    d_to   = pd.Timestamp(dates[j + 1]).date()
    manual = np.log(prices[j + 1] / prices[j])
    print(f"  {d_from} -> {d_to}   {cond[k]:+.5f}   check {manual:+.5f}")

print()
print(f"last return in window ends on "
      f"{pd.Timestamp(dates[idx]).date()}  (should be the valuation date)")