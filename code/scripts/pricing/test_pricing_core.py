"""
Check the pricing core against closed-form Black-76.

Under geometric Brownian motion the Monte Carlo price must match the
closed-form price to within Monte Carlo error. A failure here is a bug in
the pricing code, not in the generator.
"""

import numpy as np

from pricing_core import black76_call, price_strikes

rng = np.random.default_rng(0)

# --- setup ------------------------------------------------------------------
F_t   = 38.52       # settlement price on the valuation date
sigma = 0.40        # annualized volatility
h     = 20          # trading days to expiry
T     = h / 252.0
r     = 0.036
M     = 200_000     # large, so Monte Carlo error is small

strikes = np.round(F_t * np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6]), 2)

# --- simulate GBM returns ---------------------------------------------------
# Zero drift, matching the risk-neutral condition for a futures price.
# Variance per step is sigma^2 * T / h, so the h steps sum to sigma^2 * T.
step_sd = sigma * np.sqrt(T / h)
returns = rng.normal(-0.5 * step_sd**2, step_sd, size=(M, h))

# --- price ------------------------------------------------------------------
rows = price_strikes(returns, F_t, strikes, r, T)

print(f"F_t = {F_t}   sigma = {sigma}   T = {T:.4f}   M = {M:,}\n")
print(f"{'strike':>8} {'mny':>6} {'MC':>10} {'stderr':>9} "
      f"{'Black-76':>10} {'diff':>9} {'diff/se':>8} {'impl vol':>9}")

for row in rows:
    K = row["strike"]
    bs = black76_call(F_t, K, r, T, sigma)
    diff = row["price"] - bs
    z = diff / row["stderr"] if row["stderr"] > 0 else np.nan
    print(f"{K:8.2f} {row['moneyness']:6.2f} {row['price']:10.4f} "
          f"{row['stderr']:9.4f} {bs:10.4f} {diff:+9.4f} {z:+8.2f} "
          f"{row['implied_vol']:9.4f}")