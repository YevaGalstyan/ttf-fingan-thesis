"""
Price European calls on TTF futures from the trained generator.

For each valuation date in pricing_dates.csv and each seed, runs the
recursive rollout, applies the martingale correction, prices every listed
strike in the moneyness band, and inverts to implied volatility.

Run build_pricing_dates.py and build_options_panel.py first.

Usage:
    python price_options.py c9
    python price_options.py c9 --M 10000 --seeds 0 1 2 3 4
"""

import argparse

import duckdb
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import MASTER_PATH, PRICING_DATES, OUT_DIR
from pricing_core import price_strikes
from rollout import CONFIGS, load_series, load_generator, make_tag, condition_window, rollout

MNY_LO, MNY_HI = 0.5, 2.0


def market_strikes(con, valuation_date, expiry):
    """
    Listed call strikes for one expiry on one valuation date, restricted to
    the moneyness band of Sec. 4.1.5, with the exchange implied volatility.
    """
    return con.sql(f"""
        SELECT strike_price AS strike,
               settlement_price AS market_price,
               implied_vol AS market_iv
        FROM read_parquet('{MASTER_PATH}')
        WHERE trading_date    = DATE '{valuation_date:%Y-%m-%d}'
          AND expiration_date = DATE '{expiry:%Y-%m-%d}'
          AND instrument_class = 'C'
          AND moneyness_k_over_f BETWEEN {MNY_LO} AND {MNY_HI}
        ORDER BY strike_price
    """).df()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", choices=CONFIGS.keys())
    p.add_argument("--loss", default="ForGAN")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--M", type=int, default=10_000)
    args = p.parse_args()

    dates, prices, returns = load_series()
    val = pd.read_csv(PRICING_DATES, parse_dates=["expiry", "valuation"])

    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")

    rows = []
    for seed in args.seeds:
        tag = make_tag(args.config, args.loss, seed)
        gen, cfg = load_generator(tag, returns)
        print(f"\n{tag}")

        for _, v in val.iterrows():
            mkt = market_strikes(con, v["valuation"], v["expiry"])
            if mkt.empty:
                print(f"  {v['valuation']:%Y-%m-%d}  no strikes, skipped")
                continue

            cond0 = condition_window(returns, dates, v["valuation"], cfg["l"])
            sim = rollout(gen, cond0, args.M, int(v["h"]), cfg, seed)

            priced = price_strikes(sim, v["F_t"], mkt["strike"].values,
                                   v["r"], v["T"])

            for row, (_, m) in zip(priced, mkt.iterrows()):
                rows.append({
                    "config":      args.config,
                    "loss":        args.loss,
                    "seed":        seed,
                    "valuation":   v["valuation"],
                    "expiry":      v["expiry"],
                    "F_t":         v["F_t"],
                    "T":           v["T"],
                    "r":           v["r"],
                    **row,
                    "market_price": m["market_price"],
                    "market_iv":    m["market_iv"],
                })

            n_iv = sum(1 for row in priced if not np.isnan(row["implied_vol"]))
            print(f"  {v['valuation']:%Y-%m-%d}  {len(priced):3d} strikes, "
                  f"{n_iv:3d} inverted")

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / f"prices_{args.loss}_{args.config}.csv"
    out.to_csv(out_path, index=False)
    print(f"\n{len(out):,} rows written to {out_path}")


if __name__ == "__main__":
    main()