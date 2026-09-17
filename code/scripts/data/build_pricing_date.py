"""
Build the pricing date table for the option pricing benchmark.

One row per option expiry in the test split, carrying everything the pricer
needs: the valuation date, the futures price of the option's underlying on
that date, the rollout horizon and the discount rate.

The valuation date lies H trading days before the expiry, which falls in the
month before the expiry month. On that date the front-month series still
holds the preceding contract, so F_t is taken from the option's own
underlying contract instead of the front-month series.

Run build_front_month.py and build_options_panel.py first.
"""

import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import (MASTER_PATH, ESTR_CSV, PRICING_DATES, TFM_CSV,
                   FUTURES_DIR, OUT_DIR)

# Rollout horizon in trading days.
H = 20

TEST_START = "2024-09-05"
TEST_END   = "2026-04-24"

con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")

# Keep the previous table once, for comparison.
old = OUT_DIR / "pricing_dates_old.csv"
if PRICING_DATES.exists() and not old.exists():
    shutil.copy(PRICING_DATES, old)


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
        rows.append({"expiry": e, "valuation": None, "F_front": None})
        continue
    rows.append({
        "expiry":    e,
        "valuation": dates[idx - H],         # step back H trading days
        "F_front":   prices[idx - H],        # front-month price, for comparison
    })

val = pd.DataFrame(rows)
# One datetime unit everywhere, so joins and lookups match.
NS = "datetime64[ns]"
val["expiry"] = pd.to_datetime(val["expiry"]).astype(NS)
val["valuation"] = pd.to_datetime(val["valuation"]).astype(NS)


# --- 4. underlying contract and its price ----------------------------------
# The underlying of an option is the first futures contract still trading on
# the option expiry date.
ohlc = pd.read_parquet(FUTURES_DIR / "TFM_ohlc.parquet")
meta = pd.read_parquet(FUTURES_DIR / "TFM_meta.parquet")
ohlc["date"] = pd.to_datetime(ohlc["date"]).dt.tz_localize(None).astype(NS)
meta["last_tradeable_dt"] = (pd.to_datetime(meta["last_tradeable_dt"])
                             .dt.tz_localize(None).astype(NS))
meta = (meta[["symbol", "last_tradeable_dt"]]
        .dropna()
        .drop_duplicates("symbol")
        .sort_values("last_tradeable_dt"))
settle = ohlc.set_index(["symbol", "date"])["px_last"].sort_index()


def underlying(expiry):
    later = meta[meta["last_tradeable_dt"] >= expiry]
    return later["symbol"].iloc[0] if len(later) else None


def price_on(symbol, date):
    try:
        return float(settle.loc[(symbol, date)])
    except KeyError:
        return np.nan


def trading_days(symbol, start, end):
    d = ohlc[(ohlc["symbol"] == symbol)
             & (ohlc["date"] > start) & (ohlc["date"] <= end)]
    return len(d)


val["underlying"] = val["expiry"].map(underlying)
val["F_t"] = [price_on(s, d) for s, d in zip(val["underlying"], val["valuation"])]
val["spread"] = val["F_t"] - val["F_front"]
val["h_real"] = [trading_days(s, d, e) for s, d, e in
                 zip(val["underlying"], val["valuation"], val["expiry"])]

# Check: the underlying must be the front-month contract on the expiry date.
front = pd.read_parquet(OUT_DIR / "front_month_full.parquet")[["date", "front_symbol"]]
front["date"] = pd.to_datetime(front["date"]).dt.tz_localize(None).astype(NS)
chk = pd.merge_asof(val[["expiry", "underlying"]].sort_values("expiry"),
                    front.sort_values("date"),
                    left_on="expiry", right_on="date", direction="backward")
val["check"] = (chk.set_index("expiry")["front_symbol"]
                .reindex(val["expiry"]).values == val["underlying"].values)

# Sanity check: H trading days is normally 28 calendar days. Longer gaps come
# from the days missing at each contract roll (Sec. 4.1.2).
val["gap_days"] = (val.expiry - val.valuation).dt.days


# --- 5. rate and time to maturity -------------------------------------------
# asof() carries the last published rate forward over TARGET holidays.
val["estr"] = val["valuation"].map(estr_series.asof) / 100.0   # percent to decimal
val["r"] = np.log1p(val["estr"])                               # continuous compounding
val["h"] = val["h_real"]
val["T"] = val["h"] / 252.0                                        # trading-day year fraction

val = val[["expiry", "valuation", "underlying", "F_t", "F_front", "spread",
           "gap_days", "h", "h_real", "T", "estr", "r", "check"]]

val.to_csv(PRICING_DATES, index=False)

pd.set_option("display.width", 200)
print(val[["expiry", "valuation", "underlying", "F_front", "F_t",
           "spread", "gap_days", "h_real", "check"]].to_string())
print(f"\nmean |spread|: {val.spread.abs().mean():.3f} EUR/MWh, "
      f"max |spread|: {val.spread.abs().max():.3f}")
print(f"mean |spread| / F_t: {(val.spread.abs() / val.F_t).mean():.2%}")
print(f"missing F_t: {val.F_t.isna().sum()}, failed checks: {(~val.check).sum()}")
print(f"\n{len(val)} valuation dates written to {PRICING_DATES}")