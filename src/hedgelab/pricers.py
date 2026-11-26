"""Numerical methods. Each Pricer combines a Model with an Option to give a price.

- AnalyticPricer: closed-form Black-Scholes (European only), with Greeks.
- BinomialPricer: CRR tree with backward induction (European and American).
- MonteCarloPricer: exact terminal sampling, optional antithetic variates and
  an importance-sampling drift shift for rare-event payoffs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from . import black_scholes as bs
from .instruments import Call, DigitalCall, Option, Put
from .models import BlackScholesModel


class Pricer(ABC):
    def __init__(self, model: BlackScholesModel):
        self.model = model

    @abstractmethod
    def price(self, option: Option) -> float:
        ...


class AnalyticPricer(Pricer):
    def _args(self, option):
        if option.american:
            raise NotImplementedError("no closed form for American options")
        if not isinstance(option, (Call, Put)):
            raise NotImplementedError(f"no closed-form Greeks for {type(option).__name__}")
        m = self.model
        return m.spot, option.strike, option.maturity, m.rate, m.sigma

    def price(self, option):
        if isinstance(option, DigitalCall) and not option.american:
            m = self.model
            return float(bs.digital_call_price(m.spot, option.strike, option.maturity, m.rate, m.sigma))
        return float(bs.price(*self._args(option), is_call=isinstance(option, Call)))

    def delta(self, option):
        return float(bs.delta(*self._args(option), is_call=isinstance(option, Call)))

    def gamma(self, option):
        return float(bs.gamma(*self._args(option)))

    def vega(self, option):
        return float(bs.vega(*self._args(option)))


class BinomialPricer(Pricer):
    """Cox-Ross-Rubinstein tree; node i at step j has i down moves: S = S0 u^(j - 2i)."""

    def __init__(self, model, n_steps=500):
        super().__init__(model)
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        self.n_steps = n_steps

    def price(self, option):
        m, n = self.model, self.n_steps
        h = option.maturity / n
        u = np.exp(m.sigma * np.sqrt(h))
        d = 1.0 / u
        growth = np.exp(m.rate * h)
        if not d < growth < u:
            raise ValueError("no-arbitrage requires d < exp(r h) < u; increase n_steps")
        p = (growth - d) / (u - d)
        discount = 1.0 / growth

        def spots(j):  # j+1 prices at step j, highest first
            return m.spot * u ** (j - 2.0 * np.arange(j + 1))

        values = option.payoff(spots(n))
        for j in range(n - 1, -1, -1):
            values = discount * (p * values[:-1] + (1 - p) * values[1:])
            if option.american:
                values = np.maximum(values, option.payoff(spots(j)))
        return float(values[0])


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_samples: int

    def interval(self, z=1.96):
        return self.price - z * self.std_error, self.price + z * self.std_error


class MonteCarloPricer(Pricer):
    """European options by sampling S_T exactly.

    shift = theta samples B_T under Q^theta (B_T = W + theta T, W ~ N(0, T)) and
    reweights by the likelihood ratio L = exp(-theta W - theta^2 T / 2).
    """

    def __init__(self, model, n_paths=100_000, seed=None, antithetic=False, shift=0.0):
        super().__init__(model)
        self.n_paths = n_paths
        self.seed = seed
        self.antithetic = antithetic
        self.shift = shift

    def _weighted_discounted_payoffs(self, option, W):
        m, T, theta = self.model, option.maturity, self.shift
        S_T = m.spot * np.exp((m.rate - 0.5 * m.sigma**2) * T + m.sigma * (W + theta * T))
        L = np.exp(-theta * W - 0.5 * theta**2 * T)
        return np.exp(-m.rate * T) * L * option.payoff(S_T)

    def estimate(self, option) -> MCResult:
        if option.american:
            raise NotImplementedError("MonteCarloPricer handles European options only")
        rng = np.random.default_rng(self.seed)
        sqrt_T = np.sqrt(option.maturity)
        if self.antithetic:
            W = sqrt_T * rng.standard_normal(self.n_paths // 2)
            # Average each antithetic pair first: the pairs are i.i.d., the halves are not.
            samples = 0.5 * (self._weighted_discounted_payoffs(option, W)
                             + self._weighted_discounted_payoffs(option, -W))
        else:
            W = sqrt_T * rng.standard_normal(self.n_paths)
            samples = self._weighted_discounted_payoffs(option, W)
        return MCResult(float(samples.mean()),
                        float(samples.std(ddof=1) / np.sqrt(samples.size)),
                        samples.size)

    def price(self, option):
        return self.estimate(option).price
