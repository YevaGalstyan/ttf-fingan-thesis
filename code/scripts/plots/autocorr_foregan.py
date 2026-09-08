"""
Autocorrelation of squared returns, ForGAN baseline.

Reads the per-configuration ablation results and plots C2(tau) for the six
capacity configurations against the test split.

Writes the figure used in Network Capacity.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import RUNS, FIG_DIR

LAGS = [1, 2, 5, 10]
GEN_COLS = [f"path_acf2_lag{t}" for t in LAGS]
REAL_COLS = [f"real_acf2_lag{t}" for t in LAGS]

# Script config names map to the thesis configuration numbers of Table 1.
CONFIGS = {
    "c2": 2,
    "c3": 3,
    "c4": 4,
    "c5": 5,
    "c9": 8,
    "c11": 10,
}

# ---------------------------------------------------------------- load

df = pd.concat([pd.read_csv(f) for f in sorted(RUNS.glob("results_*.csv"))])
df = df[(df["loss"] == "ForGAN") & (df["config"].isin(CONFIGS))]

# ------------------------------------------------------------------ plot

fig, ax = plt.subplots(figsize=(6, 3.5))

for config, number in CONFIGS.items():
    means = df[df["config"] == config][GEN_COLS].mean()
    ax.plot(LAGS, means, linewidth=0.9, marker="o", markersize=3,
            label=f"Configuration {number}")

real = df[REAL_COLS].iloc[0]
ax.plot(LAGS, real, color="black", linewidth=1.8, marker="o", markersize=6,
        label="Test split", zorder=5)

ax.axhline(0, color="grey", linewidth=0.5)
ax.set_xticks(LAGS)
ax.set_xlabel(r"$\tau$")
ax.set_ylabel(r"$C_2(\tau)$")
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=1)

fig.tight_layout()
fig.savefig(FIG_DIR / "c2_capacity.pdf")