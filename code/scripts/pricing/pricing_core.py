"""
Option pricing core.

Takes simulated returns and produces call prices and implied volatilities.
Independent of how the returns were generated, so the same code prices the
GAN, GBM and GARCH paths.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def terminal_prices(returns, F_t):
    """
    Compound simulated returns into settlement prices at expiry, Eq. 32.

    returns: array (M, h) of simulated log returns
    F_t:     settlement price on the valuation date
    """
    return F_t * np.exp(returns.sum(axis=1))


def martingale_correction(F_T, F_t):
    """
    Rescale so the sample mean equals F_t, Eq. 33.

    Multiplicative, so skewness and excess kurtosis are unchanged.
    """
    return F_t * F_T / F_T.mean()


def call_price(F_T, K, r, T):
    """Monte Carlo call price and its standard error."""
    payoff = np.maximum(F_T - K, 0.0)
    disc = np.exp(-r * T)
    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / np.sqrt(len(payoff))
    return price, stderr


def black76_call(F, K, r, T, sigma):
    """Closed-form Black-76 call price"""
    if sigma <= 0 or T <= 0:
        return np.exp(-r * T) * max(F - K, 0.0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def implied_vol(price, F, K, r, T, lo=1e-4, hi=5.0):
    """
    Invert Black-76 for sigma.

    Returns nan when the price lies outside the no-arbitrage bounds, where
    no implied volatility exists.
    """
    lower = np.exp(-r * T) * max(F - K, 0.0)
    upper = np.exp(-r * T) * F
    if not (lower < price < upper):
        return np.nan
    try:
        return brentq(lambda s: black76_call(F, K, r, T, s) - price, lo, hi)
    except ValueError:
        return np.nan


def price_strikes(returns, F_t, strikes, r, T):
    """
    Full pipeline for one valuation date: compound, correct, price every
    strike, invert to implied volatility.

    Returns a list of dicts, one per strike.
    """
    F_T = martingale_correction(terminal_prices(returns, F_t), F_t)

    out = []
    for K in strikes:
        price, stderr = call_price(F_T, K, r, T)
        out.append({
            "strike":      K,
            "moneyness":   K / F_t,
            "price":       price,
            "stderr":      stderr,
            "implied_vol": implied_vol(price, F_t, K, r, T),
        })
    return out