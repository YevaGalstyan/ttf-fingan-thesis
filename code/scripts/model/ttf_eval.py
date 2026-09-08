"""
Distributional evaluation of a trained generator on TTF returns.

Separate from FinGAN.py: nothing here is inherited from Vuletić et al., except
the weighted strategy, which is reimplemented without the pairing of two
returns per calendar day.

Two sampling modes:
  - path:   one draw per condition, preserving time order (dynamics: ACF, price path)
  - pooled: B draws per condition, flattened (shape: tails, skew, kurtosis)
"""

import numpy as np
import torch
from scipy.stats import skew, kurtosis


def acf(x, nlags=30):
    """Autocorrelation function of x up to nlags. Returns array of length nlags+1."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = np.sum(x ** 2)
    if var == 0:
        return np.zeros(nlags + 1)
    return np.array([np.sum(x[:len(x) - k] * x[k:]) / var for k in range(nlags + 1)])


def sample_generator(gen, conditions, hid_g, z_dim, B=1, device="cpu", noise_fn=None):
    """
    Draw B samples per condition.

    conditions: tensor (n_cond, l)
    returns:    array (n_cond, B)
    """
    if noise_fn is None:
        def noise_fn(shape):
            return torch.randn(*shape, dtype=torch.float, device=device)

    gen = gen.to(device).eval()
    conditions = conditions.to(device).to(torch.float)
    n_cond = conditions.shape[0]

    cond_in = conditions.unsqueeze(0)                       # (1, n_cond, l)
    out = np.empty((n_cond, B), dtype=float)

    with torch.no_grad():
        for b in range(B):
            h0 = torch.zeros(1, n_cond, hid_g, dtype=torch.float, device=device)
            c0 = torch.zeros(1, n_cond, hid_g, dtype=torch.float, device=device)
            noise = noise_fn((1, n_cond, z_dim))
            fake = gen(noise, cond_in, h0, c0)              # (1, n_cond, 1)
            out[:, b] = fake.squeeze(0).squeeze(-1).cpu().numpy()

    return out


def distribution_stats(x, prefix="", tail_thresholds=(0.05, 0.10)):
    """Shape statistics of a 1-D return array."""
    x = np.asarray(x, dtype=float).ravel()
    d = {
        f"{prefix}n":        len(x),
        f"{prefix}mean":     float(np.mean(x)),
        f"{prefix}std":      float(np.std(x)),
        f"{prefix}skew":     float(skew(x)),
        f"{prefix}kurt":     float(kurtosis(x)),      # excess kurtosis
        f"{prefix}min":      float(np.min(x)),
        f"{prefix}max":      float(np.max(x)),
        f"{prefix}q01":      float(np.quantile(x, 0.01)),
        f"{prefix}q99":      float(np.quantile(x, 0.99)),
    }
    for t in tail_thresholds:
        d[f"{prefix}frac_abs_gt_{t}"] = float(np.mean(np.abs(x) > t))
    return d


def accuracy_stats(means, real):
    """Point forecast accuracy of the generated means against the realizations."""
    means = np.asarray(means, dtype=float)
    real = np.asarray(real, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean((means - real) ** 2))),
        "mae":  float(np.mean(np.abs(means - real))),
        "corr": float(np.corrcoef(means, real)[0, 1]),
    }


def weighted_strategy(pooled, real):
    """
    Weighted strategy of Vuletic et al., one trade per trading day.

    The position is the fraction of positive draws minus the fraction of
    negative draws, so the size reflects the certainty of the sign. The pairing
    of two returns per calendar day used in the reference implementation does
    not apply here and is omitted.

    Returns mean PnL in basis points and the annualized Sharpe ratio.
    """
    pooled = np.asarray(pooled, dtype=float)
    real = np.asarray(real, dtype=float)

    p_up = (pooled >= 0).mean(axis=1)
    position = p_up - (1.0 - p_up)
    pnl = 10000.0 * position * real

    pnl_mean = float(np.mean(pnl))
    pnl_std = float(np.std(pnl))
    sr = float(np.sqrt(252) * pnl_mean / pnl_std) if pnl_std > 0 else np.nan

    return pnl_mean, sr, pnl


def evaluate_ttf(gen, data, l, pred, hid_g, z_dim,
                 B=1000, device="cpu", nlags=30, noise_fn=None, eps=0.0002):
    """
    Evaluate a trained generator on a TTF data split.

    Works on any split; pass the validation or the test window matrix.

    Returns (stats_dict, arrays_dict).

    arrays_dict holds the raw draws so new statistics can be computed later
    without retraining:
        real     (n_cond,)      realised next returns
        path     (n_cond,)      one draw per condition, in time order
        pooled   (n_cond, B)    B draws per condition
        pnl      (n_cond,)      weighted strategy PnL in basis points
    """
    data = data.to("cpu")
    conditions = data[:, 0:l]
    real = data[:, l:l + pred].numpy().ravel()

    pooled = sample_generator(gen, conditions, hid_g, z_dim,
                              B=B, device=device, noise_fn=noise_fn)
    path = pooled[:, 0]                                   # first draw = the path
    means = pooled.mean(axis=1)

    stats = {}
    stats.update(distribution_stats(real,           prefix="real_"))
    stats.update(distribution_stats(path,           prefix="path_"))
    stats.update(distribution_stats(pooled.ravel(), prefix="pool_"))
    stats.update(accuracy_stats(means, real))

    # weighted strategy
    pnl_mean, sr, pnl = weighted_strategy(pooled, real)
    stats["pnl_w"] = pnl_mean
    stats["sr_w"] = sr

    # volatility clustering: ACF of squared returns, first few lags
    acf_real = acf(real ** 2, nlags)
    acf_path = acf(path ** 2, nlags)
    for k in (1, 2, 5, 10):
        stats[f"real_acf2_lag{k}"] = float(acf_real[k])
        stats[f"path_acf2_lag{k}"] = float(acf_path[k])

    # directional bias: fraction of conditions with a positive mean forecast
    stats["frac_pos_mean"] = float(np.mean(means > 0))
    stats["means_std"] = float(np.std(means))

    # mode collapse, Vuletic et al. threshold eps = 0.0002
    # checked across all conditions, not a single arbitrary one
    cond_stds = pooled.std(axis=1)
    stats["cond_std_min"] = float(cond_stds.min())
    stats["cond_std_median"] = float(np.median(cond_stds))
    stats["narrow_dist"] = bool(cond_stds.min() < eps)
    stats["narrow_means"] = bool(np.std(means) < eps)

    arrays = {
        "real": real,
        "path": path,
        "pooled": pooled,
        "pnl": pnl,
        "acf_real": acf_real,
        "acf_path": acf_path,
    }
    return stats, arrays