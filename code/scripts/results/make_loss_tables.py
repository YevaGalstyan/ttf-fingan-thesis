"""
Cost function branch tables for Sec. 5.5.

Reads the ablation results for one configuration and writes three LaTeX
tables: single terms, two-term combinations, and three-term combinations.
The ForGAN baseline is repeated in each table as the reference column,
following the convention of Sec. 5.3.

Usage:
    python make_loss_tables.py c9        # to stdout
    python make_loss_tables.py c9 --out  # to file
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import RUNS, OUT_DIR

# Script config names map to the thesis configuration numbers of Table 1.
CONFIGS = {
    "c1": "1", "c2": "2", "c3": "3", "c4": "4", "c5": "5",
    "c6": "6", "c7": "7", "c8": "the input scaling run",
    "c9": "8", "c10": "9", "c11": "10",
}

# Column label as it appears in the table header.
LABELS = {
    "ForGAN":       "ForGAN",
    "MSE":          r"MSE",
    "PnL":          r"PnL$^*_a$",
    "SR":           r"SR$^*$",
    "PnL_MSE":      r"PnL$^*_a$ \& MSE",
    "PnL_SR":       r"PnL$^*_a$ \& SR$^*$",
    "PnL_STD":      r"PnL$^*_a$ \& STD",
    "SR_MSE":       r"SR$^*$ \& MSE",
    "PnL_MSE_SR":   r"PnL$^*_a$ \& MSE \& SR$^*$",
    "PnL_MSE_STD":  r"PnL$^*_a$ \& MSE \& STD",
}

# The three tables. Title fragment, label key, branches.
TABLES = [
    ("the single cost function terms", "single",
     ["ForGAN", "MSE", "PnL", "SR"]),
    ("the combinations without the SR term", "nosr",
     ["ForGAN", "PnL_MSE", "PnL_STD", "PnL_MSE_STD"]),
    ("the combinations with the SR term", "sr",
     ["ForGAN", "PnL_SR", "SR_MSE", "PnL_MSE_SR"]),
]

# Row label, generated column, decimal places. None marks a rule.
ROWS = [
    ("Mean",               "pool_mean",             4),
    ("Standard deviation", "pool_std",              4),
    ("Skewness",           "pool_skew",             2),
    ("Excess kurtosis",    "pool_kurt",             2),
    ("Minimum",            "pool_min",              4),
    ("Maximum",            "pool_max",              4),
    ("1 \\% quantile",     "pool_q01",              4),
    ("99 \\% quantile",    "pool_q99",              4),
    ("$P(|x| > 0.05)$",    "pool_frac_abs_gt_0.05", 3),
    ("$P(|x| > 0.10)$",    "pool_frac_abs_gt_0.1",  3),
    (None, None, None),
    ("$C_2(1)$",           "path_acf2_lag1",        3),
    ("$C_2(2)$",           "path_acf2_lag2",        3),
    ("$C_2(5)$",           "path_acf2_lag5",        3),
    ("$C_2(10)$",          "path_acf2_lag10",       3),
]


def make_table(sub, config, title, key, branches):
    """One LaTeX table: statistics as rows, cost function branches as columns."""
    where = CONFIGS[config]
    where = where if not where.isdigit() else f"configuration {where}"
    caption = (f"Distributional statistics and autocorrelation of generated "
               f"daily TTF log returns on the test split, {title}, {where}")

    grouped = sub.groupby("loss")

    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular*}{\textwidth}{l@{\extracolsep{\fill}}" + "r" * len(branches) + "}",
        r"    \toprule",
        "    " + " & ".join([""] + [LABELS[b] for b in branches]) + r" \\",
        r"    \midrule",
    ]

    for label, col, dp in ROWS:
        if label is None:
            lines.append(r"    \midrule")
            continue
        cells = []
        for b in branches:
            g = grouped.get_group(b)[col]
            cells.append(rf"\msa{{{g.mean():.{dp}f}}}{{{g.std():.{dp}f}}}")
        lines.append("    " + " & ".join([label] + cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular*}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:loss_{config}_{key}}}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("config", choices=CONFIGS.keys())
p.add_argument("--out", action="store_true", help="write to file instead of stdout")
args = p.parse_args()

df = pd.read_csv(RUNS / f"results_{args.config}.csv")
if "config" not in df.columns:
    df["config"] = args.config

blocks = [make_table(df, args.config, title, key, branches)
          for title, key, branches in TABLES]
text = "\n".join(blocks)

if args.out:
    path = OUT_DIR / f"loss_tables_{args.config}.tex"
    path.write_text(text)
    print(f"wrote 3 tables to {path}")
else:
    print(text)