"""
Tables: implied volatility error of the generated prices and the benchmarks,
overall, by moneyness, and by cost function branch.

Implements Eq. (strike-error) and Eq. (model-error) of Sec. 5.7:

    e_{d,k} = | sigma_hat_{d,k} - sigma_{d,k} |            one strike
    e       = 1/20 sum_d 1/n_d sum_k e_{d,k}               one model

The signed error s is the same average without the absolute value. Strikes
with no implied volatility are excluded from both, and the percentage of
strikes inverted is reported alongside.

Values are reported as mean and seed spread, following the convention of
Chapter 5.

Usage:
    python make_price_error_table.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import OUT_DIR

# Label, file. The benchmarks of Sec. 4.4 and the model selected in Sec. 5.6.
MODELS = [
    ("Geometric Brownian motion", "prices_gbm_atm.csv"),
    ("Bootstrap, iid",            "prices_bootstrap_iid.csv"),
    ("Bootstrap, block",          "prices_bootstrap_block.csv"),
    ("G",                         "prices_ForGAN_c9.csv"),
]

# Label, file. The cost function branches of Sec. 4.3, configuration 8.
BRANCHES = [
    ("ForGAN",      "prices_ForGAN_c9.csv"),
    ("PnL & MSE",   "prices_Pnl_Mse_c9.csv"),
    ("PnL",         "prices_Pnl_c9.csv"),
]

# Column label, column, decimal places.
COLS = [("e", "e", 3), ("s", "s", 3), ("inverted", "inverted", 1)]

# Moneyness ranges, following the positions of Sec. 2.4. The band of
# Sec. 4.1.5 runs from 0.5 to 2.0.
EDGES  = [0.5, 0.9, 1.1, 1.5, 2.001]
LABELS = ["0.5-0.9", "0.9-1.1", "1.1-1.5", "1.5-2.0"]


def load(filename):
    """Prices with the strike errors of Eq. (strike-error) attached."""
    df = pd.read_csv(OUT_DIR / filename, parse_dates=["valuation"])
    df["e"] = (df["implied_vol"] - df["market_iv"]).abs()
    df["s"] = df["implied_vol"] - df["market_iv"]
    df["bucket"] = pd.cut(df["moneyness"], EDGES, labels=LABELS, right=False)
    return df


def summarize(df):
    """Eq. (model-error) for both columns, and the inversion rate."""
    ok = df.dropna(subset=["e"])
    return {
        "e":        ok.groupby("valuation")["e"].mean().mean(),
        "s":        ok.groupby("valuation")["s"].mean().mean(),
        "inverted": 100 * len(ok) / len(df),
    }


def collect(df, keys):
    """Mean and seed spread of the given keys, where there are seeds."""
    if "seed" not in df.columns:
        s = summarize(df)
        return {k: (s[k], None) for k in keys}
    per_seed = pd.DataFrame([summarize(g) for _, g in df.groupby("seed")])
    return {k: (per_seed[k].mean(), per_seed[k].std()) for k in keys}


def fmt(mean, sd, dp):
    return f"{mean:.{dp}f}" + (f" \u00b1 {sd:.{dp}f}" if sd is not None else "")


def overall(frames, first="model"):
    """One row per entry: e, s and the inversion rate over all strikes."""
    rows = []
    print(f"\n{first:28s}" + "".join(f"{c[0]:>18}" for c in COLS))
    for label, df in frames:
        r = collect(df, [c[1] for c in COLS])
        row, cells = {first: label}, []
        for _, key, dp in COLS:
            mean, sd = r[key]
            row[key] = round(mean, dp)
            row[f"{key}_sd"] = None if sd is None else round(sd, dp)
            cells.append(fmt(mean, sd, dp))
        rows.append(row)
        print(f"{label:28s}" + "".join(f"{x:>18}" for x in cells))
    return pd.DataFrame(rows)


def by_moneyness(frames):
    """One row per model: e within each moneyness range."""
    rows = []
    print(f"\n{'model':28s}" + "".join(f"{b:>18}" for b in LABELS))
    for label, df in frames:
        row, cells = {"model": label}, []
        for b in LABELS:
            mean, sd = collect(df[df["bucket"] == b], ["e"])["e"]
            row[b] = round(mean, 3)
            row[f"{b}_sd"] = None if sd is None else round(sd, 3)
            cells.append(fmt(mean, sd, 3))
        rows.append(row)
        print(f"{label:28s}" + "".join(f"{x:>18}" for x in cells))
    return pd.DataFrame(rows)


def by_generator(df):
    """One row per moneyness range, for G only."""
    rows = []
    print(f"\n{'moneyness':28s}" + "".join(f"{c[0]:>18}" for c in COLS))
    for b in LABELS:
        r = collect(df[df["bucket"] == b], [c[1] for c in COLS])
        row, cells = {"moneyness": b}, []
        for _, key, dp in COLS:
            mean, sd = r[key]
            row[key] = round(mean, dp)
            row[f"{key}_sd"] = None if sd is None else round(sd, dp)
            cells.append(fmt(mean, sd, dp))
        rows.append(row)
        print(f"{b:28s}" + "".join(f"{x:>18}" for x in cells))
    return pd.DataFrame(rows)


def main():
    frames = [(label, load(f)) for label, f in MODELS]
    branches = [(label, load(f)) for label, f in BRANCHES]
    generator = dict(frames)["G"]

    tables = [
        (overall(frames),                      "table_price_error.csv"),
        (by_moneyness(frames),                 "table_price_error_moneyness.csv"),
        (by_generator(generator),              "table_price_error_generator.csv"),
        (overall(branches, first="branch"),    "table_price_error_branches.csv"),
    ]
    print()
    for table, name in tables:
        table.to_csv(OUT_DIR / name, index=False)
        print(f"written to {OUT_DIR / name}")


if __name__ == "__main__":
    main()