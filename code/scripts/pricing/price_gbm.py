"""
Price European calls on TTF futures under geometric Brownian motion.

Simulates lognormal price paths from each valuation date and prices every
listed strike through the same pricing core as the generator, so the two
are directly comparable.

Two choices of sigma, selected with --sigma:
    atm         the exchange implied volatility at the strike nearest the
                money on the valuation date. GBM then matches the
                at-the-money price by construction, and the smile is flat
    historical  the realized volatility of the front-month returns over
                the --window trading days before the valuation date

Run build_pricing_dates.py and build_options_panel.py first.

Usage:
    python price_gbm.py --sigma atm
    python price_gbm.py --sigma historical --window 60
"""

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PRICING_DATES, OUT_DIR
from pricing_core import price_strikes
from rollout import load_series
from price_options import market_strikes

TRADING_DAYS = 252


def sigma_atm(mkt, F_t):
    """Exchange implied volatility at the strike nearest the money."""
    iv = mkt.dropna(subset=["market_iv"])
    if iv.empty:
        return np.nan
    j = (iv["strike"] - F_t).abs().values.argmin()
    return float(iv["market_iv"].values[j])


def sigma_historical(returns, dates, valuation_date, window):
    """Annualized realized volatility over the window before the date."""
    idx = int(np.searchsorted(dates, np.datetime64(valuation_date)))
    return float(returns[idx - window:idx].std() * np.sqrt(TRADING_DAYS))


def simulate(sigma, h, T, M, rng):
    """
    Lognormal returns with zero drift, matching the risk-neutral condition
    for a futures price. The variance per step is sigma^2 * T / h, so the
    h steps sum to sigma^2 * T.
    """
    step_sd = sigma * np.sqrt(T / h)
    return rng.normal(-0.5 * step_sd**2, step_sd, size=(M, h))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sigma", choices=["atm", "historical"], default="atm")
    p.add_argument("--window", type=int, default=60,
                   help="trading days for the historical volatility")
    p.add_argument("--M", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dates, prices, returns = load_series()
    val = pd.read_csv(PRICING_DATES, parse_dates=["expiry", "valuation"])

    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    rng = np.random.default_rng(args.seed)

    rows = []
    print(f"GBM, sigma from {args.sigma}"
          + (f", window {args.window}" if args.sigma == "historical" else ""))

    for _, v in val.iterrows():
        mkt = market_strikes(con, v["valuation"], v["expiry"])
        if mkt.empty:
            print(f"  {v['valuation']:%Y-%m-%d}  no strikes, skipped")
            continue

        if args.sigma == "atm":
            sigma = sigma_atm(mkt, v["F_t"])
        else:
            sigma = sigma_historical(returns, dates, v["valuation"], args.window)

        if not np.isfinite(sigma) or sigma <= 0:
            print(f"  {v['valuation']:%Y-%m-%d}  no sigma, skipped")
            continue

        sim = simulate(sigma, int(v["h"]), v["T"], args.M, rng)
        priced = price_strikes(sim, v["F_t"], mkt["strike"].values,
                               v["r"], v["T"])

        for row, (_, m) in zip(priced, mkt.iterrows()):
            rows.append({
                "model":        "GBM",
                "sigma_source": args.sigma,
                "sigma":        sigma,
                "valuation":    v["valuation"],
                "expiry":       v["expiry"],
                "F_t":          v["F_t"],
                "T":            v["T"],
                "r":            v["r"],
                **row,
                "market_price": m["market_price"],
                "market_iv":    m["market_iv"],
            })

        n_iv = sum(1 for row in priced if not np.isnan(row["implied_vol"]))
        print(f"  {v['valuation']:%Y-%m-%d}  sigma {sigma:.3f}  "
              f"{len(priced):3d} strikes, {n_iv:3d} inverted")

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / f"prices_gbm_{args.sigma}.csv"
    out.to_csv(out_path, index=False)
    print(f"\n{len(out):,} rows written to {out_path}")


if __name__ == "__main__":
    main()