"""
Price European calls on TTF futures by historical simulation.

Draws paths of h returns from the realized front-month series and prices
every listed strike through the same pricing core as the generator, so the
three models are directly comparable.

Two sampling modes, selected with --mode:
    iid     each of the h returns is drawn independently from the history.
            Keeps the unconditional distribution, including the skewness
            and the heavy tails, but destroys the order, so there is no
            volatility clustering
    block   each path is one contiguous block of h consecutive returns.
            Keeps the clustering as well, at the cost of fewer distinct
            paths, since the history contains only so many blocks

The history is the training split by default, so no test data enters the
benchmark. With --expanding the history grows to everything observed
before each valuation date.

Usage:
    python price_bootstrap.py --mode block
    python price_bootstrap.py --mode iid --expanding
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

TRAIN_FRAC = 0.8      # matches _scaling_params in rollout.py
TRADING_DAYS = 252


def history(returns, dates, valuation_date, expanding):
    """The returns the bootstrap draws from."""
    if expanding:
        idx = int(np.searchsorted(dates, np.datetime64(valuation_date)))
        return returns[:idx]
    return returns[:int(TRAIN_FRAC * len(returns))]


def simulate_iid(hist, h, M, rng):
    """Each return drawn independently, with replacement."""
    return rng.choice(hist, size=(M, h), replace=True)


def simulate_block(hist, h, M, rng):
    """Each path is one contiguous block of h consecutive returns."""
    starts = rng.integers(0, len(hist) - h + 1, size=M)
    return hist[starts[:, None] + np.arange(h)[None, :]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["iid", "block"], default="block")
    p.add_argument("--expanding", action="store_true",
                   help="draw from everything before the valuation date "
                        "instead of the training split only")
    p.add_argument("--M", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dates, prices, returns = load_series()
    val = pd.read_csv(PRICING_DATES, parse_dates=["expiry", "valuation"])

    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    rng = np.random.default_rng(args.seed)

    rows = []
    print(f"Historical simulation, {args.mode}, "
          f"{'expanding' if args.expanding else 'training split'} history")

    for _, v in val.iterrows():
        mkt = market_strikes(con, v["valuation"], v["expiry"])
        if mkt.empty:
            print(f"  {v['valuation']:%Y-%m-%d}  no strikes, skipped")
            continue

        h = int(v["h"])
        hist = history(returns, dates, v["valuation"], args.expanding)

        if args.mode == "iid":
            sim = simulate_iid(hist, h, args.M, rng)
        else:
            sim = simulate_block(hist, h, args.M, rng)

        priced = price_strikes(sim, v["F_t"], mkt["strike"].values,
                               v["r"], v["T"])

        # Annualized spread of the h-day return, for comparison with the
        # implied volatility the market was quoting.
        sd_ann = sim.sum(axis=1).std() * np.sqrt(TRADING_DAYS / h)

        for row, (_, m) in zip(priced, mkt.iterrows()):
            rows.append({
                "model":        "bootstrap",
                "mode":         args.mode,
                "expanding":    args.expanding,
                "n_history":    len(hist),
                "sd_ann":       sd_ann,
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
        print(f"  {v['valuation']:%Y-%m-%d}  sd {sd_ann:.3f}  "
              f"n_hist {len(hist):4d}  {len(priced):3d} strikes, "
              f"{n_iv:3d} inverted")

    out = pd.DataFrame(rows)
    suffix = "_expanding" if args.expanding else ""
    out_path = OUT_DIR / f"prices_bootstrap_{args.mode}{suffix}.csv"
    out.to_csv(out_path, index=False)
    print(f"\n{len(out):,} rows written to {out_path}")


if __name__ == "__main__":
    main()