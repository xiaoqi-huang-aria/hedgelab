"""Cross-checks: the C++ core must reproduce the NumPy reference implementation.

Same inputs -> same numbers (to floating-point rounding) for the deterministic parts;
statistical agreement for the parts that draw their own random numbers.
"""

import numpy as np
import pytest

_core = pytest.importorskip("hedgelab._core", reason="C++ core not built; see README")

from hedgelab import (AnalyticPricer, BinomialPricer, BlackScholesModel, Call,  # noqa: E402
                    DeltaHedger, Put)
from hedgelab import black_scholes as bs  # noqa: E402

R, SIGMA, T = 0.05, 0.2, 1.0
MODEL = BlackScholesModel(100.0, R, SIGMA)


@pytest.mark.parametrize("is_call", [True, False])
def test_vectorized_black_scholes_matches_numpy(is_call):
    rng = np.random.default_rng(0)
    spot = rng.uniform(50, 150, 1000)
    tau = rng.uniform(0, 2, 1000)
    tau[:10] = 0.0  # expiry limits too
    for cpp, ref in [(_core.bs_price(spot, 100.0, tau, R, SIGMA, is_call), bs.price(spot, 100.0, tau, R, SIGMA, is_call)),
                     (_core.bs_delta(spot, 100.0, tau, R, SIGMA, is_call), bs.delta(spot, 100.0, tau, R, SIGMA, is_call)),
                     (_core.bs_gamma(spot, 100.0, tau, R, SIGMA), bs.gamma(spot, 100.0, tau, R, SIGMA)),
                     (_core.bs_vega(spot, 100.0, tau, R, SIGMA), bs.vega(spot, 100.0, tau, R, SIGMA))]:
        np.testing.assert_allclose(cpp, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [4, 50, 1000])
@pytest.mark.parametrize("option", [Call(100.0, T), Put(100.0, T), Put(110.0, T, american=True),
                                    Call(95.0, 0.5, american=True)])
def test_crr_matches_numpy(option, n):
    cpp = _core.crr_price(100.0, option.strike, option.maturity, R, SIGMA, n,
                          isinstance(option, Call), option.american)
    assert cpp == pytest.approx(BinomialPricer(MODEL, n).price(option), abs=1e-10)


def test_crr_error_becomes_value_error():
    with pytest.raises(ValueError, match="no-arbitrage"):
        _core.crr_price(100.0, 100.0, 1.0, 0.5, 0.01, 1)


@pytest.mark.parametrize("option", [Call(100.0, T), Put(105.0, T)])
@pytest.mark.parametrize("cost_rate", [0.0, 0.001])
def test_hedge_paths_matches_numpy_on_identical_paths(option, cost_rate):
    t, S = BlackScholesModel(100.0, R, 0.25).simulate_paths(T, 100, 2000, rng=1)
    hedger = DeltaHedger(option, R, SIGMA, cost_rate=cost_rate)
    ref = hedger.run(t, S)
    cpp = hedger.run(t, S, backend="cpp")
    np.testing.assert_allclose(cpp.pnl, ref.pnl, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cpp.costs, ref.costs, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cpp.gamma_pnl, ref.gamma_pnl, rtol=1e-10, atol=1e-10)


def test_hedge_paths_rejects_bad_input():
    t, S = MODEL.simulate_paths(T, 10, 5, rng=0)
    with pytest.raises(ValueError):
        _core.hedge_paths(t, S[:, :-1], 100.0, T, R, SIGMA)
    with pytest.raises(ValueError):
        _core.hedge_paths(t[:-1] * 0.5, S[:, :-1], 100.0, T, R, SIGMA)  # grid stops at T/2


def test_monte_carlo_agrees_with_closed_form_and_is_thread_invariant():
    exact = AnalyticPricer(MODEL).price(Call(100.0, T))
    res = _core.mc_european(100.0, 100.0, T, R, SIGMA, True, 2_000_000, seed=3)
    assert abs(res.price - exact) < 4 * res.std_error
    one = _core.mc_european(100.0, 100.0, T, R, SIGMA, True, 100_000, seed=3, n_threads=1)
    many = _core.mc_european(100.0, 100.0, T, R, SIGMA, True, 100_000, seed=3, n_threads=8)
    assert (one.price, one.std_error) == (many.price, many.std_error)


def test_simulated_hedge_statistics_match_numpy():
    hedger = DeltaHedger(Call(100.0, T), R, SIGMA)
    cpp = hedger.simulate_gbm(BlackScholesModel(100.0, R, 0.25), 252, 40_000, seed=7)
    t, S = BlackScholesModel(100.0, R, 0.25).simulate_paths(T, 252, 40_000, rng=7)
    ref = hedger.run(t, S)
    # Different random numbers, same distribution: means agree within sampling error.
    se = np.hypot(cpp.summary()["std_error"], ref.summary()["std_error"])
    assert abs(cpp.pnl.mean() - ref.pnl.mean()) < 4 * se
    assert cpp.pnl.std() == pytest.approx(ref.pnl.std(), rel=0.03)
    assert np.corrcoef(cpp.pnl, cpp.gamma_pnl)[0, 1] > 0.98
    one = hedger.simulate_gbm(MODEL, 50, 3000, seed=1, n_threads=1)
    many = hedger.simulate_gbm(MODEL, 50, 3000, seed=1, n_threads=8)
    np.testing.assert_array_equal(one.pnl, many.pnl)
