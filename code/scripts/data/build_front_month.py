"""
Front-month TTF series construction.

Reads the three Bloomberg parquet files, builds the daily front-month
settlement price series, and writes it to CSV as the model input.

Also reports the diagnostics quoted in the thesis:
  - settlement prices outside the intraday high-low range (Source Data)
  - contract transition day statistics (Front-Month Series Construction)
"""

import numpy as np
import pandas as pd
from paths import FUTURES_DIR, OUT_DIR, TFM_CSV

# ---------------------------------------------------------------- load

ohlc = pd.read_parquet(FUTURES_DIR / "TFM_ohlc.parquet")
meta = pd.read_parquet(FUTURES_DIR / "TFM_meta.parquet")
chains = pd.read_parquet(FUTURES_DIR / "TFM_chains.parquet")

ohlc["date"] = pd.to_datetime(ohlc["date"])
chains["as_of"] = pd.to_datetime(chains["as_of"])
meta["last_tradeable_dt"] = pd.to_datetime(meta["last_tradeable_dt"])


# ------------------------------------------------- front-month series

# Attach each contract's expiry to the monthly listings, then keep the
# earliest-expiring contract in each snapshot: that is the front month.
chains = chains.merge(meta[["symbol", "last_tradeable_dt"]], on="symbol", how="left")

front_snapshots = (
    chains.sort_values(["as_of", "last_tradeable_dt"])
    .drop_duplicates("as_of")[["as_of", "symbol"]]
    .rename(columns={"symbol": "front_symbol"})
)

# Snapshots are monthly, the series is daily: carry each snapshot forward
# until the next one.
all_dates = ohlc[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)

daily_chain = pd.merge_asof(
    all_dates,
    front_snapshots.sort_values("as_of"),
    left_on="date",
    right_on="as_of",
    direction="backward",
)

# Attach the settlement price of the selected contract.
front = daily_chain.merge(
    ohlc[["symbol", "date", "px_last"]].rename(columns={"px_last": "front_price"}),
    left_on=["date", "front_symbol"],
    right_on=["date", "symbol"],
    how="left",
)

front = (
    front.dropna(subset=["front_price"])
    .sort_values("date")
    .reset_index(drop=True)
)

front["switch"] = front["front_symbol"] != front["front_symbol"].shift(1)
front["ret"] = np.log(front["front_price"]).diff()


# -------------------------------------------------------------- export

pd.DataFrame(
    {
        "date": front["date"],
        "AdjClose": front["front_price"],  # column name expected by Fin-GAN
    }
).to_csv(TFM_CSV, index=False)

front.to_parquet(OUT_DIR / "front_month_full.parquet", index=False)

# -------------------------------------------------------------- diagnostics

print(f"Front-month series: {len(front):,} trading days")
print(f"  Period:  {front.date.min():%Y-%m-%d} to {front.date.max():%Y-%m-%d}")
print(f"  Price:   {front.front_price.min():.3f} to {front.front_price.max():.3f} EUR/MWh")
print(f"  Returns: {front.ret.notna().sum():,}")
front["gap_days"] = front["date"].diff().dt.days
gaps = front.loc[front["switch"], "gap_days"]
print(f"  Transition gaps: {gaps.min():.0f} to {gaps.max():.0f} days, median {gaps.median():.0f}")