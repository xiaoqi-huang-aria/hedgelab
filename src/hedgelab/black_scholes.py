"""Vectorized Black-Scholes formulas (no dividends).

All functions broadcast over spot and tau (time to maturity), so the hedging
simulator can evaluate a whole (paths x dates) grid in one call. At tau = 0
they return the expiry limits: payoff for price, 1/0.5/0 for call delta, and 0
for gamma and vega.
"""

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _d1_d2(spot, strike, tau, rate, sigma):
    spot, strike, tau = np.broadcast_arrays(
        np.asarray(spot, float), np.asarray(strike, float), np.asarray(tau, float)
    )
    if np.any(tau < 0):
        raise ValueError("time to maturity must be nonnegative")
    live = tau > 0
    safe_tau = np.where(live, tau, 1.0)
    vol_sqrt_t = sigma * np.sqrt(safe_tau)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * safe_tau) / vol_sqrt_t
    return spot, strike, safe_tau, live, d1, d1 - vol_sqrt_t


def _pdf(x):
    return np.exp(-0.5 * x**2) / _SQRT_2PI


def price(spot, strike, tau, rate, sigma, is_call=True):
    spot, strike, safe_tau, live, d1, d2 = _d1_d2(spot, strike, tau, rate, sigma)
    df_strike = strike * np.exp(-rate * safe_tau)
    if is_call:
        value = spot * ndtr(d1) - df_strike * ndtr(d2)
        intrinsic = np.maximum(spot - strike, 0.0)
    else:
        value = df_strike * ndtr(-d2) - spot * ndtr(-d1)
        intrinsic = np.maximum(strike - spot, 0.0)
    return np.where(live, value, intrinsic)


def digital_call_price(spot, strike, tau, rate, sigma):
    """Price of a contract paying 1 if S_T >= K: exp(-r tau) N(d2)."""
    spot, strike, safe_tau, live, _, d2 = _d1_d2(spot, strike, tau, rate, sigma)
    return np.where(live, np.exp(-rate * safe_tau) * ndtr(d2), (spot >= strike).astype(float))


def delta(spot, strike, tau, rate, sigma, is_call=True):
    spot, strike, _, live, d1, _ = _d1_d2(spot, strike, tau, rate, sigma)
    expiry = np.where(spot > strike, 1.0, np.where(spot < strike, 0.0, 0.5))
    call_delta = np.where(live, ndtr(d1), expiry)
    return call_delta if is_call else call_delta - 1.0


def gamma(spot, strike, tau, rate, sigma):
    spot, _, safe_tau, live, d1, _ = _d1_d2(spot, strike, tau, rate, sigma)
    return np.where(live, _pdf(d1) / (spot * sigma * np.sqrt(safe_tau)), 0.0)


def vega(spot, strike, tau, rate, sigma):
    spot, _, safe_tau, live, d1, _ = _d1_d2(spot, strike, tau, rate, sigma)
    return np.where(live, spot * _pdf(d1) * np.sqrt(safe_tau), 0.0)
