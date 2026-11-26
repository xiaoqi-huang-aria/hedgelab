"""Option contracts: what is paid, when, and whether early exercise is allowed.

An Option knows nothing about models or numerical methods. Pricers ask it for
its payoff, so adding a new contract means adding one subclass.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Option(ABC):
    strike: float
    maturity: float
    american: bool = False

    def __post_init__(self):
        if self.strike <= 0 or self.maturity <= 0:
            raise ValueError("strike and maturity must be positive")

    @abstractmethod
    def payoff(self, spot):
        """Exercise value at spot price(s); vectorized over numpy arrays."""


@dataclass(frozen=True)
class Call(Option):
    def payoff(self, spot):
        return np.maximum(np.asarray(spot, float) - self.strike, 0.0)


@dataclass(frozen=True)
class Put(Option):
    def payoff(self, spot):
        return np.maximum(self.strike - np.asarray(spot, float), 0.0)


@dataclass(frozen=True)
class DigitalCall(Option):
    """Pays 1 if S_T >= K; exp(rT) times its price is the risk-neutral P(S_T >= K)."""

    def payoff(self, spot):
        return (np.asarray(spot, float) >= self.strike).astype(float)
