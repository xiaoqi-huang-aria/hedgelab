"""Discrete delta hedging of a short European option.

The desk sells the option at its Black-Scholes price with implied volatility
sigma_i, then holds Delta(t_i, S_i; sigma_i) shares over [t_i, t_{i+1}] and
finances the position with a cash account at rate r. The price paths can come
from any model, so the realized dynamics may differ from the hedging model.

Sign convention: P&L is the seller's terminal P&L in time-T dollars,
    P&L = hedge portfolio at T - option payoff - transaction costs.

Taylor-expanding the sigma_i Black-Scholes value over one step and using the
Black-Scholes PDE gives the per-step approximation
    P&L_i ~ 1/2 Gamma_i S_i^2 (sigma_i^2 dt - (dS_i / S_i)^2),
so a seller who hedged at sigma_i makes money when realized variance comes in
below sigma_i^2, and loses when it comes in above.
"""

from dataclasses import dataclass

import numpy as np

from . import black_scholes as bs
from .instruments import Call, Option, Put


@dataclass(frozen=True)
class HedgeResult:
    pnl: np.ndarray        # (n_paths,) seller P&L net of costs, time-T dollars
    costs: np.ndarray      # (n_paths,) transaction costs, time-T dollars
    gamma_pnl: np.ndarray  # (n_paths,) sum of the per-step approximation above
    premium: np.ndarray    # (n_paths,) premium received at t = 0

    @property
    def gross_pnl(self):
        return self.pnl + self.costs

    def summary(self):
        n = self.pnl.size
        std = float(self.pnl.std(ddof=1))
        return {"mean": float(self.pnl.mean()), "std": std,
                "std_error": std / np.sqrt(n), "mean_cost": float(self.costs.mean())}


class DeltaHedger:
    def __init__(self, option: Option, rate, hedge_sigma, cost_rate=0.0):
        if option.american or not isinstance(option, (Call, Put)):
            raise NotImplementedError("DeltaHedger supports European calls and puts")
        if hedge_sigma <= 0 or cost_rate < 0:
            raise ValueError("need hedge_sigma > 0 and cost_rate >= 0")
        self.option = option
        self.rate = rate
        self.hedge_sigma = hedge_sigma
        self.cost_rate = cost_rate  # fraction of traded notional, e.g. 0.001 = 10 bp

    def _core_args(self):
        return dict(strike=self.option.strike, maturity=self.option.maturity, rate=self.rate,
                    hedge_sigma=self.hedge_sigma, is_call=isinstance(self.option, Call),
                    cost_rate=self.cost_rate)

    def run(self, t, S, backend="numpy", n_threads=0) -> HedgeResult:
        """Hedge on every date of grid t. S has shape (n_paths, len(t)).

        backend="cpp" runs the same computation in the compiled core (hedgelab._core),
        path by path on n_threads threads (0 = all cores).
        """
        t = np.asarray(t, float)
        S = np.atleast_2d(np.asarray(S, float))
        T = self.option.maturity
        if t.ndim != 1 or t.size < 2 or S.shape[1] != t.size:
            raise ValueError("t must be 1-D with at least 2 dates, matching S.shape[1]")
        if not (np.isclose(t[0], 0.0) and np.isclose(t[-1], T)):
            raise ValueError("the grid must run from 0 to the option's maturity")

        K, r, sig = self.option.strike, self.rate, self.hedge_sigma
        is_call = isinstance(self.option, Call)
        premium = bs.price(S[:, 0], K, T, r, sig, is_call)
        if backend == "cpp":
            from . import _core
            pnl, costs, gamma_pnl = _core.hedge_paths(t, S, n_threads=n_threads, **self._core_args())
            return HedgeResult(pnl=pnl, costs=costs, gamma_pnl=gamma_pnl, premium=premium)
        if backend != "numpy":
            raise ValueError("backend must be 'numpy' or 'cpp'")

        S_now, tau = S[:, :-1], T - t[:-1]
        deltas = bs.delta(S_now, K, tau, r, sig, is_call)        # (n_paths, n)

        # Self-financing portfolio in discounted units:
        #   e^{-rT} X_T = V_0 + sum_i Delta_i (S~_{i+1} - S~_i),  S~_i = e^{-r t_i} S_i
        discounted_S = S * np.exp(-r * t)
        hedge_value = np.exp(r * T) * (premium + np.sum(deltas * np.diff(discounted_S, axis=1), axis=1))

        # Proportional costs on every trade: initial buy, rebalances, final unwind.
        carry = np.exp(r * (T - t))                               # time-t dollars -> time-T dollars
        trades = np.diff(deltas, axis=1, prepend=0.0)
        costs = self.cost_rate * (np.sum(np.abs(trades) * S_now * carry[:-1], axis=1)
                                  + np.abs(deltas[:, -1]) * S[:, -1])

        pnl = hedge_value - self.option.payoff(S[:, -1]) - costs

        gammas = bs.gamma(S_now, K, tau, r, sig)
        returns = np.diff(S, axis=1) / S_now
        gamma_pnl = np.sum(0.5 * gammas * S_now**2 * (sig**2 * np.diff(t) - returns**2) * carry[1:], axis=1)

        return HedgeResult(pnl=pnl, costs=costs, gamma_pnl=gamma_pnl, premium=premium)

    def simulate_gbm(self, model, n_steps, n_paths, seed=0, drift=None, n_threads=0) -> HedgeResult:
        """Simulate GBM paths from a BlackScholesModel and hedge them, all in C++.

        Paths are generated and hedged one at a time, so memory stays O(n_steps)
        per thread. Results depend on seed but not on n_threads.
        """
        from . import _core
        mu = model.rate if drift is None else drift
        pnl, costs, gamma_pnl = _core.simulate_and_hedge(
            model.spot, mu, model.sigma, n_steps, n_paths, seed, n_threads=n_threads, **self._core_args())
        premium = np.full(n_paths, bs.price(model.spot, self.option.strike, self.option.maturity,
                                            self.rate, self.hedge_sigma, isinstance(self.option, Call)))
        return HedgeResult(pnl=pnl, costs=costs, gamma_pnl=gamma_pnl, premium=premium)
