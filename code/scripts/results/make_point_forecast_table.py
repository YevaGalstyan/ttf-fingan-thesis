"""
Appendix table for the point forecast metrics.

Reads the per-configuration ablation results and writes one LaTeX table with
the ten cost function branches as rows: the accuracy of the mean set against
the realized returns of the test split.

These metrics are not the criterion of this thesis. They are reported for
comparability with the equity results of Vuletic et al. (2024).

Usage:
    python make_point_forecast_table.py c9     # one configuration, to stdout
    python make_point_forecast_table.py        # selected configuration, to file
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import RUNS, POINT_FORECAST_TEX as OUT

# Script config names map to the thesis configuration numbers of Table 1.
CONFIGS = {
    "c1": "1", "c2": "2", "c3": "3", "c4": "4", "c5": "5",
    "c6": "6", "c7": "7", "c8": "the input scaling run",
    "c9": "8", "c10": "9", "c11": "10",
}

# The configuration selected in Sec. 5.6.
SELECTED = "c9"

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

# Column label, results column, decimal places.
COLS = [
    ("MAE",              "mae",           4),
    ("RMSE",             "rmse",          4),
    ("Corr",             "corr",          4),
    (r"$P(m_i > 0)$",    "frac_pos_mean", 4),
    (r"$\mathrm{SD}(m)$", "means_std",    4),
]


def make_table(df, config):
    """Build the point forecast table for one configuration."""
    sub = df[df["config"] == config]
    if sub.empty:
        raise SystemExit(f"no rows for config '{config}'")

    label = CONFIGS[config]
    where = label if not label.isdigit() else f"configuration {label}"
    caption = ("Point forecast metrics of the mean set against the realized "
               f"returns of the test split, all cost function branches, {where}")

    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{l" + "r" * len(COLS) + "}",
        r"    \toprule",
        "    " + " & ".join([""] + [c[0] for c in COLS]) + r" \\",
        r"    \midrule",
    ]

    for name, tex in BRANCHES:
        rows = sub[sub["loss"] == name]
        if rows.empty:
            continue
        cells = [rf"\msa{{{rows[col].mean():.{dp}f}}}{{{rows[col].std():.{dp}f}}}"
                 for _, col, dp in COLS]
        lines.append("    " + " & ".join([tex] + cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:point_forecast_{config}}}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


df = pd.concat([pd.read_csv(f) for f in sorted(RUNS.glob("results_*.csv"))])

if len(sys.argv) > 1:
    print(make_table(df, sys.argv[1]))
else:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(make_table(df, SELECTED))
    print(f"wrote the point forecast table to {OUT}")