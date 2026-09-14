"""
Appendix tables for the full experimental grid.

Reads the per-configuration ablation results and writes three LaTeX tables per
configuration, with the ten cost function branches as rows: distributional
statistics, tail statistics, and the autocorrelation of squared returns.

Usage:
    python make_appendix_tables.py c1          # one configuration, to stdout
    python make_appendix_tables.py             # all configurations, to file
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import RUNS, GRID_TABLES_TEX as OUT

# Script config names map to the thesis configuration numbers of Table 1.
CONFIGS = {
    "c1": "1", "c2": "2", "c3": "3", "c4": "4", "c5": "5",
    "c6": "6", "c7": "7", "c8": "the input scaling run",
    "c9": "8", "c10": "9", "c11": "10",
}

# Branch order and labels, following Sec. 4.3.
BRANCHES = [
    ("ForGAN",       "1"),
    ("MSE",          "2"),
    ("PnL",          "3"),
    ("SR",           "4"),
    ("PnL_MSE",      "5"),
    ("PnL_SR",       "6"),
    ("PnL_STD",      "7"),
    ("SR_MSE",       "8"),
    ("PnL_MSE_SR",   "9"),
    ("PnL_MSE_STD",  "10"),
]

# Three groups of columns, one table each.
# Column label, generated column, realized column, decimal places.
GROUPS = [
    ("distributional statistics", "dist", [
        ("Mean", "pool_mean", 4),
        ("SD",   "pool_std",  4),
        ("Skew", "pool_skew", 2),
        ("Kurt", "pool_kurt", 2),
        ("Min",  "pool_min",  4),
        ("Max",  "pool_max",  4),
    ]),
    ("tail statistics", "tail", [
        (r"$q_{1}$",    "pool_q01",              4),
        (r"$q_{99}$",   "pool_q99",              4),
        (r"$P_{0.05}$", "pool_frac_abs_gt_0.05", 3),
        (r"$P_{0.10}$", "pool_frac_abs_gt_0.1",  3),
    ]),
    ("autocorrelation of squared returns", "acf", [
        (r"$C_2(1)$",  "path_acf2_lag1",  3),
        (r"$C_2(2)$",  "path_acf2_lag2",  3),
        (r"$C_2(5)$",  "path_acf2_lag5",  3),
        (r"$C_2(10)$", "path_acf2_lag10", 3),
    ]),
]


def make_table(sub, config, group_name, group_key, cols):
    """Build one LaTeX table for one configuration and one column group."""
    label = CONFIGS[config]
    where = label if not label.isdigit() else f"configuration {label}"
    caption = f"All cost function branches, {where}: {group_name}"

    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{l" + "r" * len(cols) + "}",
        r"    \toprule",
        "    " + " & ".join([""] + [c[0] for c in cols]) + r" \\",
        r"    \midrule",
    ]

    for name, tex in BRANCHES:
        rows = sub[sub["loss"] == name]
        if rows.empty:
            continue
        cells = [rf"\msa{{{rows[gen].mean():.{dp}f}}}{{{rows[gen].std():.{dp}f}}}"
                 for _, gen, dp in cols]
        lines.append("    " + " & ".join([tex] + cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:grid_{config}_{group_key}}}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def make_config(df, config):
    """The three tables for one configuration."""
    sub = df[df["config"] == config]
    if sub.empty:
        raise SystemExit(f"no rows for config '{config}'")
    return "\n".join(make_table(sub, config, name, key, cols)
                     for name, key, cols in GROUPS)


df = pd.concat([pd.read_csv(f) for f in sorted(RUNS.glob("results_*.csv"))])

if len(sys.argv) > 1:
    print(make_config(df, sys.argv[1]))
else:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    blocks = [make_config(df, c) for c in CONFIGS if (df["config"] == c).any()]
    OUT.write_text("\n".join(blocks))
    print(f"wrote {len(blocks) * 3} tables to {OUT}")