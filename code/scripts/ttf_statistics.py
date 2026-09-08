"""
Summary statistics of the TTF return series, full sample and per data split.

Uses distribution_stats from ttf_eval so that the numbers match those reported
for the generated returns.

Run build_front_month.py first.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
from ttf_eval import distribution_stats, acf

BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "output" / "TFM.csv"

df = pd.read_csv(CSV_PATH, parse_dates=["date"]).sort_values("date")
df["x"] = np.log(df["AdjClose"] / df["AdjClose"].shift(1))
df = df.dropna(subset=["x"]).reset_index(drop=True)

print(f"{len(df)} return observations, "
      f"{df.date.min().date()} to {df.date.max().date()}")

SPLITS = {
    "Full":       ("2010-01-05", "2026-04-24"),
    "Train":      ("2010-01-05", "2023-01-18"),
    "Validation": ("2023-01-19", "2024-09-04"),
    "Test":       ("2024-09-05", "2026-04-24"),

    # The 2026 supply disruption. Excluded to show that the distributional
    # properties of the test split come from this episode.
    "Spike": ("2026-02-01", "2026-04-30")
}

def subset(start, end):
    m = (df.date >= start) & (df.date <= end)
    return df.loc[m, "x"].to_numpy()

rows = {name: distribution_stats(subset(s, e)) for name, (s, e) in SPLITS.items()}

test = df[(df.date >= SPLITS["Test"][0]) & (df.date <= SPLITS["Test"][1])]
excl = test[~((test.date >= SPLITS["Spike"][0]) & (test.date <= SPLITS["Spike"][1]))]["x"].to_numpy()
rows["Test excl. spike"] = distribution_stats(excl)

table = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\n" + table.to_string(float_format=lambda v: f"{v:.4f}"))

series = {name: subset(s, e) for name, (s, e) in SPLITS.items()}
series["Test excl. spike"] = excl

print("\nAutocorrelation")
print(f"{'':16s}" + "".join(f"{'C('+str(k)+')':>9s}" for k in (1, 2, 5, 10))
                  + "".join(f"{'C2('+str(k)+')':>9s}" for k in (1, 2, 5, 10)))
for name, x in series.items():
    a1, a2 = acf(x, 10), acf(x ** 2, 10)
    print(f"{name:16s}" + "".join(f"{a1[k]:>9.3f}" for k in (1, 2, 5, 10))
                        + "".join(f"{a2[k]:>9.3f}" for k in (1, 2, 5, 10)))