"""
Build the pricing date table for the option pricing benchmark.

One row per option expiry in the test split, carrying everything the pricer
needs: the valuation date, the settlement price on that date, the rollout
horizon and the discount rate.

Run build_front_month.py and build_options_panel.py first.
"""

import numpy as np
import pandas as pd
import duckdb
from paths import MASTER_PATH, ESTR_CSV, PRICING_DATES, TFM_CSV

# Rollout horizon in trading days. A month is 21 to 22 trading days, so 20
# keeps the whole path inside the life of a single front-month contract.
H = 20

TEST_START = "2024-09-05"
TEST_END   = "2026-04-24"

con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")


# --- 1. risk-free rate ------------------------------------------------------
# Euro short-term rate, published by the ECB on TARGET business days.
estr = pd.read_csv(ESTR_CSV, parse_dates=["DATE"])
estr.columns = ["date", "period", "estr"]
estr = estr[["date", "estr"]].dropna()
estr_series = estr.set_index("date")["estr"].sort_index()


# --- 2. option expiries in the test split -----------------------------------
expiries = con.sql(f"""
    SELECT DISTINCT expiration_date
    FROM read_parquet('{MASTER_PATH}')
    WHERE expiration_date BETWEEN DATE '{TEST_START}' AND DATE '{TEST_END}'
    ORDER BY 1
""").df()["expiration_date"].values


# --- 3. valuation date H trading days before each expiry --------------------
fut = pd.read_csv(TFM_CSV, parse_dates=["date"])
dates = fut["date"].values
prices = fut["AdjClose"].values

rows = []
for e in expiries:
    idx = np.searchsorted(dates, e)          # position of the expiry in the series
    if idx < H:
        rows.append({"expiry": e, "valuation": None, "F_t": None})
        continue
    rows.append({
        "expiry":    e,
        "valuation": dates[idx - H],         # step back H trading days
        "F_t":       prices[idx - H],        # settlement price on that date
    })

val = pd.DataFrame(rows)

# Sanity check: H trading days is normally 28 calendar days. Longer gaps come
# from the days missing at each contract roll (Sec. 4.1.2).
val["gap_days"] = (val.expiry - val.valuation).dt.days


# --- 4. rate and time to maturity -------------------------------------------
# asof() carries the last published rate forward over TARGET holidays.
val["estr"] = val["valuation"].map(estr_series.asof) / 100.0   # percent to decimal
val["r"] = np.log1p(val["estr"])                               # continuous compounding
val["h"] = H
val["T"] = H / 252.0                                           # trading-day year fraction

val = val[["expiry", "valuation", "F_t", "gap_days", "h", "T", "estr", "r"]]

val.to_csv(PRICING_DATES, index=False)
print(val.to_string())
print(f"\n{len(val)} valuation dates written to {PRICING_DATES}")