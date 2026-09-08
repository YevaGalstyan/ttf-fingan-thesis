"""Shared paths for all scripts."""

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

DATA_DIR    = BASE / "data"
FUTURES_DIR = DATA_DIR / "futures"
OPTIONS_DIR = DATA_DIR / "options"
ESTR_CSV    = DATA_DIR / "ESTR" / "ESTR_2019_2026.csv"

OUT_DIR = BASE / "output"
RUNS    = OUT_DIR / "runs"

TFM_CSV       = OUT_DIR / "TFM.csv"
MASTER_PATH   = OUT_DIR / "master_options_daily.parquet"
PRICING_DATES = OUT_DIR / "pricing_dates.csv"

DEF_GLOB    = str(OPTIONS_DIR / "definition" / "*.parquet")
STATS_GLOB  = str(OPTIONS_DIR / "statistics" / "*.parquet")
BBO_GLOB    = str(OPTIONS_DIR / "bbo" / "*.parquet")
OHLC_GLOB   = str(OPTIONS_DIR / "ohlc" / "*.parquet")
STATUS_GLOB = str(OPTIONS_DIR / "status" / "*.parquet")
