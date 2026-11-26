"""Dynamics of the underlying.

A Model holds the market parameters and can simulate price paths. Pricers read
parameters from it; the hedging simulator only consumes the paths it produces,
so a different model (e.g. Heston) can generate the "real world" while the
hedger keeps using Black-Scholes deltas.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Model(ABC):
    spot: float
    rate: float

    @abstractmethod
    def simulate_paths(self, maturity, n_steps, n_paths, rng=None, drift=None):
        """Return t with shape (n_steps+1,) and S with shape (n_paths, n_steps+1).

        drift is the real-world drift mu; None means risk-neutral (mu = rate).
        """


@dataclass(frozen=True)
class BlackScholesModel(Model):
    sigma: float

    def __post_init__(self):
        if self.spot <= 0 or self.sigma <= 0:
            raise ValueError("spot and sigma must be positive")

    def paths_from_normals(self, z, maturity, drift=None):
        """Exact GBM on a uniform grid from standard normals z of shape (n_paths, n_steps)."""
        z = np.atleast_2d(np.asarray(z, float))
        mu = self.rate if drift is None else drift
        n_steps = z.shape[1]
        dt = maturity / n_steps
        log_increments = (mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
        t = np.linspace(0.0, maturity, n_steps + 1)
        S = np.empty((z.shape[0], n_steps + 1))
        S[:, 0] = self.spot
        S[:, 1:] = self.spot * np.exp(np.cumsum(log_increments, axis=1))
        return t, S

    def simulate_paths(self, maturity, n_steps, n_paths, rng=None, drift=None):
        if maturity <= 0 or n_steps < 1 or n_paths < 1:
            raise ValueError("need maturity > 0, n_steps >= 1, n_paths >= 1")
        rng = np.random.default_rng(rng)
        return self.paths_from_normals(rng.standard_normal((n_paths, n_steps)), maturity, drift)
