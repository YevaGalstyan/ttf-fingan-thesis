"""
Build the options master table: one row per option contract per trading day.

Four sources are combined:
  definition -> contracts     (one row per contract: strike, expiry, type)
  statistics -> daily_stats   (settlement price, implied vol, delta)
  bbo        -> daily_quotes  (daily average mid and spread)
  ohlc       -> daily_trades  (daily open/high/low/close/volume)

The spine is daily_stats: a row exists only where a settlement price exists.
Quotes and trades are joined on as optional enrichment.
"""

from pathlib import Path
import duckdb

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" / "options"
FUTURES_DIR = BASE / "data" / "futures"
OUT_DIR = BASE / "output"

DEF_GLOB   = str(DATA_DIR / "definition" / "*.parquet")
STATS_GLOB = str(DATA_DIR / "statistics" / "*.parquet")
BBO_GLOB   = str(DATA_DIR / "bbo" / "*.parquet")
OHLC_GLOB  = str(DATA_DIR / "ohlc" / "*.parquet")

TFM_CSV  = OUT_DIR / "TFM.csv"
OUT_PATH = OUT_DIR / "master_options_daily.parquet"

con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")


# --- 1. contracts: one row per contract -----------------------------------
# definition is an event log, so keep only the latest record per contract.
con.execute(f"""
CREATE OR REPLACE TABLE contracts AS
SELECT
    instrument_id,
    raw_symbol,
    instrument_class,                       -- 'C' or 'P'
    strike_price,                           -- EUR/MWh
    CAST(expiration AS DATE) AS expiration_date,
    underlying_id
FROM (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY instrument_id
               ORDER BY ts_event DESC
           ) AS rn
    FROM read_parquet('{DEF_GLOB}')
    WHERE instrument_class IN ('C', 'P')    -- vanilla calls and puts only
)
WHERE rn = 1
""")


# --- 2. daily_stats: settlement price and implied volatility --------------
# ts_ref is populated only on final settlements, so it doubles as the filter
# for preliminary marks. Duplicates are resolved by the highest sequence.
con.execute(f"""
CREATE OR REPLACE TABLE daily_stats AS
WITH base AS (
    SELECT
        instrument_id,
        CAST(ts_ref AS DATE) AS trading_date,
        stat_type,
        price,
        sequence
    FROM read_parquet('{STATS_GLOB}')
    WHERE stat_type IN (3, 14)          -- settlement, implied vol
      AND ts_ref IS NOT NULL
),
dedup AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY instrument_id, trading_date, stat_type
               ORDER BY sequence DESC
           ) AS rn
    FROM base
)
SELECT
    instrument_id,
    trading_date,
    MAX(CASE WHEN stat_type = 3  THEN price END) AS settlement_price,
    MAX(CASE WHEN stat_type = 14 THEN price END) AS implied_vol,
FROM dedup
WHERE rn = 1
GROUP BY instrument_id, trading_date
""")

# --- 3. master ------------------------------------------------------------
con.execute("""
CREATE OR REPLACE TABLE master AS
SELECT
    s.trading_date,
    s.instrument_id,
    c.raw_symbol,
    c.instrument_class,
    c.strike_price,
    c.expiration_date,
    c.underlying_id,
    datediff('day', s.trading_date, c.expiration_date)         AS days_to_expiry,
    datediff('day', s.trading_date, c.expiration_date) / 365.0 AS ttm_years,
    s.settlement_price,
    s.implied_vol
FROM daily_stats s
JOIN contracts c USING (instrument_id)
WHERE s.trading_date <= c.expiration_date       -- drop post-expiry rows
  AND s.settlement_price IS NOT NULL            -- spine requires a settlement
""")


# --- 4. futures price and moneyness ---------------------------------------
# Joins the stitched front-month series on trading_date. Valid only where the
# option's underlying is the front-month contract; see Sec. 4.1.5.
con.execute(f"""
CREATE OR REPLACE TABLE master AS
SELECT m.*,
       f.AdjClose                   AS futures_price,
       m.strike_price / f.AdjClose  AS moneyness_k_over_f
FROM master m
LEFT JOIN read_csv_auto('{TFM_CSV}') f
       ON CAST(f.date AS DATE) = m.trading_date
""")

OUT_DIR.mkdir(parents=True, exist_ok=True)
con.execute(f"COPY master TO '{OUT_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD)")

n_rows = con.execute("SELECT COUNT(*) FROM master").fetchone()[0]
print(f"{n_rows:,} rows written to {OUT_PATH}")