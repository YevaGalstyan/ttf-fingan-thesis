"""
Ablation grid for Fin-GAN on TTF returns.

Trains all ten cost function branches under each configuration, with five seeds
per configuration. Writes one directory per run and one CSV per configuration
under output/runs.

Run build_front_month.py first.
"""

import random, json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import FinGAN
import copy
import time
import ttf_eval
import contextlib
import io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import OUT_DIR as DATA_DIR, RUNS

RUNS.mkdir(parents=True, exist_ok=True)

# --- data parameters ---
contract = "TFM"
tr   = 0.8    # 80% train
vl   = 0.1    # 10% val
pred = 1      # predict 1 return ahead
h    = 1      # sliding window step size

# --- model hyperparameters ---
hid_g    = 8      # generator hidden dimension
lrg      = 0.0001 # generator learning rate
lrd      = 0.0001 # discriminator learning rate
tanh_coeff = 100  # controls sharpness of sign approximation in PnL loss

# --- training parameters ---
diter         = 1    # discriminator updates per generator update
checkpoint_epoch = 5
batch_size    = 100


def rawreturns_closeonly_ttf(dataloc, contract):
    """
    TTF-specific version of rawreturns().
    Uses only daily settlement prices (AdjClose) — no open price available.
    Produces one log return per trading day: r_t = log(P_t / P_{t-1})
    No date cutoff — full history 2010 to 2026 is used.
    """
    s_df = pd.read_csv(dataloc / f"{contract}.csv")
    dates_dt = pd.to_datetime(s_df['date'])

    s_logclose = np.log(s_df['AdjClose'])
    s_ret = np.diff(s_logclose)     
    return s_ret, dates_dt

def split_train_val_test_ttf(dataloc, contract, tr, vl, h, l, pred):
    returns, dates_dt = rawreturns_closeonly_ttf(dataloc, contract)

    N     = len(returns)
    N_tr  = int(tr * N)
    N_vl  = int(vl * N)
    N_tst = N - N_tr - N_vl

    train_sr = returns[0        : N_tr]
    val_sr   = returns[N_tr     : N_tr + N_vl]
    test_sr  = returns[N_tr + N_vl :]

    def make_windows(series, h, l, pred):
        n = int((len(series) - l - pred) / h) + 1
        data = np.zeros(shape=(n, l + pred))
        for i in range(n):
            data[i, :] = series[i*h : i*h + l + pred]
        return data

    train_data = make_windows(train_sr, h, l, pred)
    val_data   = make_windows(val_sr,   h, l, pred)
    test_data  = make_windows(test_sr,  h, l, pred)
    return train_data, val_data, test_data, dates_dt

def load_splits(l):
    """Build the window matrices for the three splits."""
    train_data, val_data, test_data, _ = split_train_val_test_ttf(
        dataloc=DATA_DIR, contract=contract, tr=tr, vl=vl, h=h, l=l, pred=pred)
    return (torch.from_numpy(train_data).to(torch.float),
            torch.from_numpy(val_data).to(torch.float),
            torch.from_numpy(test_data).to(torch.float))


def scaling_params(data_train, batch_size, scaling):
    """Mean and standard deviation used to standardize the network inputs."""
    if scaling == "reference":
        ref = data_train[0:batch_size, :]
    elif scaling == "full":
        ref = data_train
    else:
        raise ValueError(f"unknown scaling: {scaling}")
    return torch.mean(ref), torch.std(ref)

def build_networks(l, z_dim, hid_g, hid_d, ref_mean, ref_std):
    """Generator, discriminator and their optimizers."""
    gen  = FinGAN.Generator(z_dim, l, hid_g, pred, ref_mean, ref_std)
    disc = FinGAN.Discriminator(l + pred, hid_d, ref_mean, ref_std)
    gen_opt  = torch.optim.RMSprop(gen.parameters(),  lr=lrg)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=lrd)
    return gen, disc, gen_opt, disc_opt


def gradient_matching(gen, disc, gen_opt, disc_opt, criterion,
                      data_train, l, z_dim, hid_g, hid_d, ngrad, batch_size):
    """
    Phase one. Train on BCE alone for ngrad epochs and return the
    coefficients alpha, beta, gamma, delta. The weights are updated here,
    so the effective training length is ngrad + n_epochs.
    """
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        return FinGAN.GradientCheck(
            contract, gen, disc, gen_opt, disc_opt, criterion, ngrad,
            data_train, batch_size, hid_d, hid_g, z_dim, lrd, lrg,
            h, l, pred, diter, tanh_coeff, "cpu", False)

def train_branch(train_fn, base_gen, base_disc, criterion, coeffs,
                 data_train, data_val, l, z_dim, hid_g, hid_d, n_epochs, batch_size):
    """
    Phase two. Train one loss branch from a copy of the phase-one networks,
    so that the comparison isolates the loss.
    """
    a, b, g, d = coeffs
    g_i, d_i = copy.deepcopy(base_gen), copy.deepcopy(base_disc)
    opt_g = torch.optim.RMSprop(g_i.parameters(), lr=lrg)
    opt_d = torch.optim.RMSprop(d_i.parameters(), lr=lrd)

    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        g_i, d_i, _, _ = train_fn(
            g_i, d_i, opt_g, opt_d, criterion, a, b, g, d,
            n_epochs, checkpoint_epoch, data_train, data_val, batch_size,
            hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff,
            "cpu", False)
    return g_i

def evaluate_branch(g_i, data_test, data_val, l, hid_g, z_dim):
    """Test statistics, plus the validation Sharpe used for selection."""
    stats, arrays = ttf_eval.evaluate_ttf(
        g_i, data_test, l, pred, hid_g, z_dim, B=1000)
    stats_val, _ = ttf_eval.evaluate_ttf(
        g_i, data_val, l, pred, hid_g, z_dim, B=1000)
    stats["sr_w_val"]  = stats_val["sr_w"]
    stats["pnl_w_val"] = stats_val["pnl_w"]
    return stats, arrays

def save_run(tag, g_i, arrays, stats):
    """Write the generator, the raw draws and the statistics to disk."""
    out = RUNS / tag
    out.mkdir(exist_ok=True)
    torch.save(g_i.state_dict(), out / "generator.pth")
    np.savez_compressed(out / "arrays.npz", **arrays)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))


LOSS_FNS = {
    "ForGAN":       FinGAN.TrainLoopForGAN,
    "MSE":          FinGAN.TrainLoopMainMSEnv,
    "PnL":          FinGAN.TrainLoopMainPnLnv,
    "PnL_STD":      FinGAN.TrainLoopMainPnLSTDnv,
    "PnL_MSE":      FinGAN.TrainLoopMainPnLMSEnv,
    "PnL_SR":       FinGAN.TrainLoopMainPnLSRnv,
    "PnL_MSE_STD":  FinGAN.TrainLoopMainPnLMSESTDnv,
    "PnL_MSE_SR":   FinGAN.TrainLoopMainPnLMSESRnv,
    "SR":           FinGAN.TrainLoopMainSRnv,
    "SR_MSE":       FinGAN.TrainLoopMainSRMSEnv,
}

def run_seed(seed, l, z_dim, hid_d, ngrad, n_epochs, hid_g=hid_g, scaling="reference"):
    """Train and evaluate all ten loss branches for one seed."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    data_train, data_val, data_test = load_splits(l)
    ref_mean, ref_std = scaling_params(data_train, batch_size, scaling)
    gen, disc, gen_opt, disc_opt = build_networks(
        l, z_dim, hid_g, hid_d, ref_mean, ref_std)
    criterion = nn.BCELoss()

    gen, disc, gen_opt, disc_opt, a, b, g, d = gradient_matching(
        gen, disc, gen_opt, disc_opt, criterion,
        data_train, l, z_dim, hid_g, hid_d, ngrad, batch_size)

    base_gen, base_disc = copy.deepcopy(gen), copy.deepcopy(disc)
    coeffs = (a, b, g, d)
    rows = []

    for name, train_fn in LOSS_FNS.items():
        print(f"  seed {seed}  {name}", flush=True)
        g_i = train_branch(train_fn, base_gen, base_disc, criterion, coeffs,
                           data_train, data_val, l, z_dim, hid_g, hid_d,
                           n_epochs, batch_size)

        stats, arrays = evaluate_branch(g_i, data_test, data_val, l, hid_g, z_dim)

        stats.update(dict(seed=seed, loss=name, l=l,
                          z_dim=z_dim, hid_g=hid_g, hid_d=hid_d,
                          ngrad=ngrad, n_epochs=n_epochs, scaling=scaling,
                          alpha=float(a), beta=float(b),
                          gamma=float(g), delta=float(d)))

        tag = (f"{name}_l{l}_z{z_dim}_hg{hid_g}_hd{hid_d}"f"_ng{ngrad}_ne{n_epochs}_{scaling}_seed{seed}")
        save_run(tag, g_i, arrays, stats)
        rows.append(stats)

    return rows

def run_config(name, **cfg):
    """
    Run one configuration across five seeds and write its own CSV.

    Each configuration writes separately, so a failure in one leaves the
    others intact. The timing print shows what the next configuration
    will cost.
    """
    t0 = time.time()
    print(f"\n{name}", flush=True)
    rows = [r for s in range(5) for r in run_seed(s, **cfg)]
    df = pd.DataFrame(rows)
    df["config"] = name
    df.to_csv(RUNS / f"results_{name}.csv", index=False)
    print(f"{name} done in {(time.time() - t0) / 60:.1f} min")
    return df


# Configuration names map to Table 1 in the thesis:
#   c1--c7   configurations 1--7
#   c8       input scaling test, reported separately
#   c9--c11  configurations 8--10

# Configuration 1: reference configuration, following Algorithm 1
df1 = run_config("c1", l=10, z_dim=8, hid_d=8, ngrad=25, n_epochs=100)

# # Configuration 2: longer training
df2 = run_config("c2", l=10, z_dim=8, hid_d=8, ngrad=100, n_epochs=500)

# # Configuration 3: increased noise dimension
df3 = run_config("c3", l=10, z_dim=32, hid_d=8, ngrad=100, n_epochs=500)

# # Configuration 4: increased discriminator hidden size
df4 = run_config("c4", l=10, z_dim=8, hid_d=64, ngrad=100, n_epochs=500)

# # Configuration 5: both, matching the published implementation defaults
df5 = run_config("c5", l=10, z_dim=32, hid_d=64, ngrad=100, n_epochs=500)

# # Configuration 6: longer condition window
df6 = run_config("c6", l=30, z_dim=32, hid_d=64, ngrad=100, n_epochs=500)

# # Configuration 7: extended training
df7 = run_config("c7", l=10, z_dim=32, hid_d=64, ngrad=100, n_epochs=1500)

# # Input scaling test: configuration 5 with full-split standardization
df8 = run_config("c8", l=10, z_dim=32, hid_d=64, ngrad=100, n_epochs=500,
                 scaling="full")

# # Configuration 8: increased generator hidden size
df9 = run_config("c9", l=10, z_dim=32, hid_d=64, ngrad=100, n_epochs=500, hid_g=64)

# # Configuration 9: extended training with the increased generator
df10 = run_config("c10", l=10, z_dim=32, hid_d=64, ngrad=100, n_epochs=1500, hid_g=64)

# # Configuration 10: increased generator with the original noise dimension
df11 = run_config("c11", l=10, z_dim=8,  hid_d=64, ngrad=100, n_epochs=500,  hid_g=64)