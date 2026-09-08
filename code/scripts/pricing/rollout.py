"""
Recursive rollout from a trained generator.

Loads one trained generator and produces simulated return paths, in the
shape expected by pricing_core.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
import FinGAN

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import RUNS, TFM_CSV

# Configuration names, matching run_ablation.py
CONFIGS = {
    "c1":  "l10_z8_hg8_hd8_ng25_ne100_reference",
    "c2":  "l10_z8_hg8_hd8_ng100_ne500_reference",
    "c3":  "l10_z32_hg8_hd8_ng100_ne500_reference",
    "c4":  "l10_z8_hg8_hd64_ng100_ne500_reference",
    "c5":  "l10_z32_hg8_hd64_ng100_ne500_reference",
    "c6":  "l30_z32_hg8_hd64_ng100_ne500_reference",
    "c7":  "l10_z32_hg8_hd64_ng100_ne1500_reference",
    "c8":  "l10_z32_hg8_hd64_ng100_ne500_full",
    "c9":  "l10_z32_hg64_hd64_ng100_ne500_reference",
    "c10": "l10_z32_hg64_hd64_ng100_ne1500_reference",
    "c11": "l10_z8_hg64_hd64_ng100_ne500_reference",
}

TAG_RE = re.compile(
    r"(?P<loss>.+)_l(?P<l>\d+)_z(?P<z_dim>\d+)_hg(?P<hid_g>\d+)"
    r"_hd(?P<hid_d>\d+)_ng(?P<ngrad>\d+)_ne(?P<n_epochs>\d+)"
    r"_(?P<scaling>\w+)_seed(?P<seed>\d+)$"
)


def parse_tag(tag):
    """Extract the hyperparameters encoded in a run directory name."""
    m = TAG_RE.match(tag)
    if m is None:
        raise ValueError(f"cannot parse tag: {tag}")
    d = m.groupdict()
    for k in ("l", "z_dim", "hid_g", "hid_d", "ngrad", "n_epochs", "seed"):
        d[k] = int(d[k])
    return d


def make_tag(config, loss="ForGAN", seed=0):
    return f"{loss}_{CONFIGS[config]}_seed{seed}"


def load_series():
    """Dates, prices and daily log returns of the front-month series."""
    df = pd.read_csv(TFM_CSV, parse_dates=["date"])
    prices = df["AdjClose"].values
    returns = np.diff(np.log(prices))
    return df["date"].values, prices, returns


def _scaling_params(returns, l, scaling, pred=1, batch_size=100):
    """Reproduce the standardization used during training."""
    N = len(returns)
    N_tr = int(0.8 * N)
    train_sr = returns[:N_tr]

    n = len(train_sr) - l - pred + 1
    windows = np.zeros((n, l + pred))
    for i in range(n):
        windows[i, :] = train_sr[i : i + l + pred]

    ref = windows[0:batch_size, :] if scaling == "reference" else windows
    ref = torch.from_numpy(ref).to(torch.float)
    return torch.mean(ref), torch.std(ref)


def load_generator(tag, returns):
    """Rebuild the network and load the trained weights."""
    cfg = parse_tag(tag)
    ref_mean, ref_std = _scaling_params(returns, cfg["l"], cfg["scaling"])

    gen = FinGAN.Generator(cfg["z_dim"], cfg["l"], cfg["hid_g"], 1,
                           ref_mean, ref_std)
    gen.load_state_dict(torch.load(RUNS / tag / "generator.pth"))
    gen.eval()
    return gen, cfg


def rollout(gen, cond0, M, h, cfg, seed=None):
    """
    Recursive rollout, Sec. 4.6.

    cond0: array (l,) of realized returns preceding the valuation date
    returns: array (M, h) of simulated log returns
    """
    if seed is not None:
        torch.manual_seed(seed)

    l, z_dim, hid_g = cfg["l"], cfg["z_dim"], cfg["hid_g"]

    cond = torch.from_numpy(np.tile(cond0, (M, 1))).to(torch.float)
    out = np.zeros((M, h))

    with torch.no_grad():
        for step in range(h):
            h0 = torch.zeros(1, M, hid_g, dtype=torch.float)
            c0 = torch.zeros(1, M, hid_g, dtype=torch.float)
            noise = torch.randn(1, M, z_dim, dtype=torch.float)

            nxt = gen(noise, cond.unsqueeze(0), h0, c0)      # (1, M, 1)
            nxt = nxt.squeeze(0).squeeze(-1)                 # (M,)

            out[:, step] = nxt.numpy()
            cond = torch.cat([cond[:, 1:], nxt.unsqueeze(1)], dim=1)

    return out


def condition_window(returns, dates, valuation_date, l):
    """The l realized returns preceding the valuation date."""
    # returns[i] is the return from dates[i] to dates[i+1], so the return
    # observed on dates[j] is returns[j-1].
    idx = int(np.searchsorted(dates, np.datetime64(valuation_date)))
    return returns[idx - l : idx]