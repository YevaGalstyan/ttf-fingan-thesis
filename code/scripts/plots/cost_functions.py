"""
Skewness and maximum of the generated returns, cost function branches.

Reads the per-branch ablation results and plots the mean and seed spread of
each of the ten cost function branches against the test split.
Configuration 8 throughout.

Distributional statistics are taken from the pooled set, following the
sampling modes of ttf_eval.py.

Writes the figure used in Effect of the Cost Function Terms.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import RUNS, FIG_DIR

CONFIG = "c9"  # thesis configuration 8

# Script loss name -> label as printed in Tables 12-14, and colour group.
# Groups follow the composition of the cost function.
LOSSES = {
    "ForGAN":      (r"ForGAN",                   "baseline"),
    "MSE":         (r"MSE",                      "single"),
    "PnL":         (r"PnL$^*_a$",                "single"),
    "SR":          (r"SR$^*$",                   "single"),
    "PnL_MSE":     (r"PnL$^*_a$ & MSE",          "combo"),
    "PnL_STD":     (r"PnL$^*_a$ & STD",          "combo"),
    "PnL_MSE_STD": (r"PnL$^*_a$ & MSE & STD",    "combo"),
    "PnL_SR":      (r"PnL$^*_a$ & SR$^*$",       "sr"),
    "SR_MSE":      (r"SR$^*$ & MSE",             "sr"),
    "PnL_MSE_SR":  (r"PnL$^*_a$ & MSE & SR$^*$", "sr"),
}

COLORS = {
    "baseline": "#4d4d4d",   # grey
    "single":   "#2a9d8f",   # teal
    "combo":    "#1f77b4",   # blue
    "sr":       "#6a4c93",   # purple
}

LEGEND = [
    ("baseline", "ForGAN baseline"),
    ("single",   "single term"),
    ("combo",    r"combination without SR$^*$"),
    ("sr",       r"combination with SR$^*$"),
]

BOX_W = 0.32
PAD = 0.02   # fraction of the data range left blank above and below

# Panel: generated column, realized column, axis label.
PANELS = [
    ("pool_skew", "real_skew", "Skewness"),
    ("pool_max",  "real_max",  "Maximum"),
]

# ---------------------------------------------------------------- load

df = pd.concat([pd.read_csv(f) for f in sorted(RUNS.glob("results_*.csv"))])
df = df[(df["config"] == CONFIG) & (df["loss"].isin(LOSSES))]

missing = set(LOSSES) - set(df["loss"].unique())
if missing:
    raise SystemExit(f"no rows for {sorted(missing)} under config {CONFIG}")

# ------------------------------------------------------------------ plot

positions = {loss: i for i, loss in enumerate(LOSSES)}

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

for ax, (gen_col, real_col, label) in zip(axes, PANELS):
    stats = {loss: (df[df["loss"] == loss][gen_col].mean(),
                    df[df["loss"] == loss][gen_col].std())
             for loss in positions}

    for loss, x in positions.items():
        mean, spread = stats[loss]
        color = COLORS[LOSSES[loss][1]]

        ax.bar(x, 2 * spread, bottom=mean - spread, width=BOX_W,
               color=color, alpha=0.35, edgecolor=color, linewidth=1.0,
               zorder=2)
        ax.hlines(mean, x - BOX_W / 2, x + BOX_W / 2,
                  color=color, linewidth=2.0, zorder=3)

    real = df[real_col].iloc[0]
    ax.axhline(real, color="black", linewidth=1.3, zorder=4)

    # Explicit limits: ax.margins has no effect once bar() has set the
    # data limits, so the boxes were being clipped.
    lo = min(m - s for m, s in stats.values())
    hi = max(m + s for m, s in stats.values())
    lo, hi = min(lo, real), max(hi, real)
    pad = PAD * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_ylabel(label)
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

axes[0].axhline(0, color="grey", linewidth=0.6, linestyle="--", zorder=1)
axes[1].set_xticks(list(positions.values()))
axes[1].set_xticklabels([lab for lab, _ in LOSSES.values()],
                        rotation=45, ha="right")

handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[g], alpha=0.35,
                         edgecolor=COLORS[g], label=text)
           for g, text in LEGEND]
handles.append(plt.Line2D([], [], color="black", linewidth=1.3,
                          label="Test split"))
fig.legend(handles=handles, frameon=False, fontsize=8,
           loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=5)

fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(FIG_DIR / "loss_skew_max.pdf", bbox_inches="tight")