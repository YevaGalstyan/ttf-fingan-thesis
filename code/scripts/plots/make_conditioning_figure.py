"""
Response of the simulated distribution to the market state.

For each valuation date, runs the rollout of Sec. 4.6 and computes the
annualized spread of the simulated h-day return, the same quantity the
bootstrap reports. The spread is compared with the spread of the realized
returns in the condition window and with the implied volatility the market
was quoting, to check whether the condition window produces a response.

Writes the numbers to conditioning_check.csv and the figure to
conditioning.pdf.

Usage:
    python check_conditioning.py
    python check_conditioning.py --seeds 0 1 2 3 4 --M 20000
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PRICING_DATES, OUT_DIR, FIG_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pricing"))
from rollout import load_series, load_generator, make_tag, condition_window, rollout

TRADING_DAYS = 252
CONFIG = "c9"
LOSS = "ForGAN"

G_COLOR      = "#c44e52"
BOOT_COLOR   = "#55a868"
MARKET_COLOR = "#000000"


def compute(args):
    """The annualized spread of the simulated return, per date and seed."""
    dates, prices, returns = load_series()
    val = pd.read_csv(PRICING_DATES, parse_dates=["expiry", "valuation"])

    rows = []
    for seed in args.seeds:
        tag = make_tag(CONFIG, LOSS, seed)
        gen, cfg = load_generator(tag, returns)
        print(f"\n{tag}")

        for _, v in val.iterrows():
            h = int(v["h"])
            cond0 = condition_window(returns, dates, v["valuation"], cfg["l"])
            sim = rollout(gen, cond0, args.M, h, cfg, seed)

            sd_ann = sim.sum(axis=1).std() * np.sqrt(TRADING_DAYS / h)
            cond_sd = cond0.std() * np.sqrt(TRADING_DAYS)

            rows.append({"valuation": v["valuation"], "seed": seed,
                         "sd_ann": sd_ann, "cond_sd": cond_sd})
            print(f"  {v['valuation']:%Y-%m-%d}  sd {sd_ann:.3f}  "
                  f"cond {cond_sd:.3f}")

    return pd.DataFrame(rows)


def summarize(df):
    """Mean and seed spread per date, with the market implied volatility."""
    gbm = pd.read_csv(OUT_DIR / "prices_gbm_atm.csv",
                      parse_dates=["valuation"])
    out = df.groupby("valuation").agg(
        sd_ann=("sd_ann", "mean"),
        sd_ann_sd=("sd_ann", "std"),
        cond_sd=("cond_sd", "first"),
    )
    out["sigma_mkt"] = gbm.groupby("valuation")["sigma"].first()

    print("\nmean over seeds, by valuation date")
    print(out.round(3).to_string())
    print(f"\nG        spread range {out.sd_ann.min():.3f} "
          f"to {out.sd_ann.max():.3f}")
    print(f"market   sigma  range {out.sigma_mkt.min():.3f} "
          f"to {out.sigma_mkt.max():.3f}")
    print(f"window   spread range {out.cond_sd.min():.3f} "
          f"to {out.cond_sd.max():.3f}")
    print(f"\ncorrelation with the market sigma:   "
          f"{out.sd_ann.corr(out.sigma_mkt):.3f}")
    print(f"correlation with the window spread:  "
          f"{out.sd_ann.corr(out.cond_sd):.3f}")
    return out


def plot(g):
    """One panel: the spread of each model across the valuation dates."""
    boot = pd.read_csv(OUT_DIR / "prices_bootstrap_block.csv",
                       parse_dates=["valuation"])
    boot = boot.groupby("valuation")["sd_ann"].first().reindex(g.index)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = g.index

    ax.plot(x, g["sigma_mkt"], color=MARKET_COLOR, lw=1.6, marker="o", ms=3.5,
            label="Market")
    ax.plot(x, boot.values, color=BOOT_COLOR, lw=1.6, marker="o", ms=3.5,
            label="Bootstrap, block")
    ax.fill_between(x, g["sd_ann"] - g["sd_ann_sd"],
                    g["sd_ann"] + g["sd_ann_sd"],
                    color=G_COLOR, alpha=0.18, lw=0)
    ax.plot(x, g["sd_ann"], color=G_COLOR, lw=1.6, marker="o", ms=3.5,
            label="$G$")

    ax.set_xlabel("Valuation date")
    ax.set_ylabel("Annualized volatility")
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in x], rotation=45, ha="right", fontsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.autofmt_xdate(rotation=45)

    fig.tight_layout()
    out_path = FIG_DIR / "conditioning.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--M", type=int, default=20_000)
    args = p.parse_args()

    summary = summarize(compute(args))

    csv_path = OUT_DIR / "conditioning_check.csv"
    summary.to_csv(csv_path)
    fig_path = plot(summary)

    print(f"\nwritten to {csv_path}")
    print(f"written to {fig_path}")


if __name__ == "__main__":
    main()