"""
Aggregate the ablation results into the tables reported in the thesis.

Reads the per-configuration CSVs written by run_ablation.py and reports
means and standard deviations across the five seeds.

Distributional statistics are taken from the pooled set and the autocorrelation
of squared returns from the path set, following the sampling modes of
ttf_eval.py.

Usage:
    python summarize_runs.py                  # summarize all 11 configs
    python summarize_runs.py c3               # summarize only config c3
    python summarize_runs.py --table          # table rows, ForGAN branch
    python summarize_runs.py --table --loss SR_MSE
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import RUNS

COLS = ["path_skew", "path_kurt", "pool_skew", "pool_kurt", "sr_w", "sr_w_val"]

# Table rows: label, generated column, realized column, decimal places.
ROWS = [
    ("Mean",               "pool_mean",             "real_mean",             4),
    ("Standard deviation", "pool_std",              "real_std",              4),
    ("Skewness",           "pool_skew",             "real_skew",             2),
    ("Excess kurtosis",    "pool_kurt",             "real_kurt",             2),
    ("Minimum",            "pool_min",              "real_min",              4),
    ("Maximum",            "pool_max",              "real_max",              4),
    ("1 % quantile",       "pool_q01",              "real_q01",              4),
    ("99 % quantile",      "pool_q99",              "real_q99",              4),
    ("P(|x| > 0.05)",      "pool_frac_abs_gt_0.05", "real_frac_abs_gt_0.05", 3),
    ("P(|x| > 0.10)",      "pool_frac_abs_gt_0.1",  "real_frac_abs_gt_0.1",  3),
    (None, None, None, None),   # separator
    ("C2(1)",              "path_acf2_lag1",        "real_acf2_lag1",        3),
    ("C2(2)",              "path_acf2_lag2",        "real_acf2_lag2",        3),
    ("C2(5)",              "path_acf2_lag5",        "real_acf2_lag5",        3),
    ("C2(10)",             "path_acf2_lag10",       "real_acf2_lag10",       3),
]

LABEL_W = 20
CELL_W = 20


def stats_table(df, loss):
    """Print the table rows of one loss branch, one column per configuration."""
    sub = df[df["loss"] == loss]
    if sub.empty:
        raise SystemExit(f"no rows with loss '{loss}'; "
                         f"available: {sorted(df['loss'].unique())}")

    configs = sorted(sub["config"].unique(), key=lambda c: int(c.lstrip("c")))
    grouped = sub.groupby("config")
    n_seeds = sub.groupby("config").size().to_dict()

    head = f"{loss:<{LABEL_W}} {'Real':>10}" + "".join(
        f" {c:>{CELL_W}}" for c in configs)
    print(head)
    print("-" * len(head))

    for label, gen_col, real_col, dp in ROWS:
        if label is None:
            print("-" * len(head))
            continue
        real = sub[real_col].iloc[0]
        line = f"{label:<{LABEL_W}} {real:>10.{dp}f}"
        for c in configs:
            g = grouped.get_group(c)[gen_col]
            line += f" {f'{g.mean():.{dp}f} +/- {g.std():.{dp}f}':>{CELL_W}}"
        print(line)

    print("-" * len(head))
    print(f"{'seeds':<{LABEL_W}} {'':>10}" + "".join(
        f" {n_seeds[c]:>{CELL_W}}" for c in configs))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "config",
    nargs="?",
    default=None,
    help="Config to summarize, e.g. 'c3' (default: all configs, c1-c11)",
)
parser.add_argument(
    "--table",
    action="store_true",
    help="Print the thesis table rows of one loss branch instead of the full summary",
)
parser.add_argument(
    "--loss",
    default="ForGAN",
    help="Loss branch for --table (default: ForGAN, the baseline)",
)
args = parser.parse_args()

pattern = f"results_{args.config}.csv" if args.config else "results_*.csv"
files = sorted(RUNS.glob(pattern))
if not files:
    parser.error(f"no files matching '{pattern}' found in {RUNS}")

df = pd.concat([pd.read_csv(f) for f in files])

if args.table:
    stats_table(df, args.loss)
else:
    summary = (df.groupby(["config", "loss"])[COLS]
                 .agg(["mean", "std"])
                 .round(3))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(summary.to_string())