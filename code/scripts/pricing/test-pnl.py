"""
Print the step-by-step spread of the recursive rollout.

The generator is trained one step ahead on real conditions. In the rollout
it is fed its own output, so errors can compound over the h steps. This
prints the standard deviation of the simulated return at each step, to show
whether the spread grows, shrinks or stays flat.

Usage:
    python diagnose_steps.py c9
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PRICING_DATES
from rollout import CONFIGS, load_series, load_generator, make_tag, condition_window, rollout

# (loss, seed, valuation date, what was seen)
CASES = [
    ("PnL",    4, "2026-01-26", "exploding, 0/63 inverted"),
    ("PnL",    0, "2026-01-26", "healthy,  63/63 inverted"),
    ("PnL",    1, "2024-10-28", "collapsing, 29/70 inverted"),
    ("ForGAN", 4, "2025-08-28", "narrow,   26/48 inverted"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", choices=CONFIGS.keys())
    p.add_argument("--M", type=int, default=10_000)
    args = p.parse_args()

    dates, prices, returns = load_series()
    val = pd.read_csv(PRICING_DATES, parse_dates=["expiry", "valuation"])

    # Spread of the realized returns, for scale.
    real_sd = returns.std()
    print(f"realized daily return sd: {real_sd:.4f}\n")

    for loss, seed, date, note in CASES:
        v = val[val["valuation"] == pd.Timestamp(date)].iloc[0]

        tag = make_tag(args.config, loss, seed)
        gen, cfg = load_generator(tag, returns)

        cond0 = condition_window(returns, dates, v["valuation"], cfg["l"])
        sim = rollout(gen, cond0, args.M, int(v["h"]), cfg, seed)

        sd = sim.std(axis=0)          # (h,) spread across paths at each step
        mean = sim.mean(axis=0)

        # lag-1 autocorrelation of the steps, within each path, averaged over paths
        a = sim - sim.mean(axis=1, keepdims=True)
        num = (a[:, :-1] * a[:, 1:]).sum(axis=1)
        den = (a ** 2).sum(axis=1)
        print(f"  within-path lag-1 autocorr: {np.nanmean(num / den):+.3f}")
        print(f"  cum sd {sim.sum(axis=1).std():.4f}  vs independent "
        f"{sim.std(axis=0).mean() * np.sqrt(sim.shape[1]):.4f}")

        print(f"{loss} seed {seed}  {date}   ({note})")
        print("  step:  " + " ".join(f"{k:>8d}" for k in range(1, len(sd) + 1)))
        print("  sd:    " + " ".join(f"{s:8.4f}" for s in sd))
        print("  mean:  " + " ".join(f"{m:8.4f}" for m in mean))
        print(f"  ratio sd[last]/sd[first]: {sd[-1] / sd[0]:.2f}")
        print()


if __name__ == "__main__":
    main()