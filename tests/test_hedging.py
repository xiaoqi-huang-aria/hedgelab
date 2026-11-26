import numpy as np
import pytest

from hedgelab import AnalyticPricer, BlackScholesModel, Call, DeltaHedger, Put

R, SIGMA, T = 0.05, 0.2, 1.0
CALL = Call(100.0, T)


def paths(sigma=SIGMA, n_steps=252, n_paths=20_000, seed=0, drift=None):
    return BlackScholesModel(100.0, R, sigma).simulate_paths(T, n_steps, n_paths, rng=seed, drift=drift)


@pytest.mark.parametrize("option", [CALL, Put(100.0, T), Call(110.0, T)])
def test_hedge_is_unbiased_when_vols_match_and_drift_is_risk_neutral(option):
    # Under Q the discounted hedge gains are a martingale, so E[P&L] = 0 exactly.
    result = DeltaHedger(option, R, SIGMA).run(*paths())
    s = result.summary()
    assert abs(s["mean"]) < 4 * s["std_error"]


def test_single_step_grid_is_static_hedge():
    t, S = paths(n_steps=1, n_paths=5)
    result = DeltaHedger(CALL, R, SIGMA).run(t, S)
    a = AnalyticPricer(BlackScholesModel(100.0, R, SIGMA))
    expected = (np.exp(R * T) * a.price(CALL) + a.delta(CALL) * (S[:, 1] - 100.0 * np.exp(R * T))
                - np.maximum(S[:, 1] - 100.0, 0.0))
    np.testing.assert_allclose(result.pnl, expected)


def test_hedging_error_std_scales_like_sqrt_dt():
    t, S = paths(n_steps=1600, n_paths=20_000, seed=1)
    hedger = DeltaHedger(CALL, R, SIGMA)
    steps = np.array([25, 50, 100, 200, 400, 800, 1600])
    stds = [hedger.run(t[:: 1600 // n], S[:, :: 1600 // n]).summary()["std"] for n in steps]
    slope = np.polyfit(np.log(T / steps), np.log(stds), 1)[0]
    assert slope == pytest.approx(0.5, abs=0.05)


def test_gamma_pnl_explains_vol_mismatch():
    # Realized 25% vs hedged at 20%: the seller loses roughly vega * (sigma_r - sigma_i).
    result = DeltaHedger(CALL, R, SIGMA).run(*paths(sigma=0.25))
    assert np.corrcoef(result.pnl, result.gamma_pnl)[0, 1] > 0.98
    assert result.pnl.mean() == pytest.approx(result.gamma_pnl.mean(), rel=0.02)
    vega = AnalyticPricer(BlackScholesModel(100.0, R, SIGMA)).vega(CALL)
    assert result.pnl.mean() == pytest.approx(-vega * 0.05 * np.exp(R * T), rel=0.05)


def test_transaction_costs_are_subtracted_exactly_and_grow_with_frequency():
    t, S = paths(n_steps=400, n_paths=2_000, seed=2)
    free = DeltaHedger(CALL, R, SIGMA).run(t, S)
    costly = DeltaHedger(CALL, R, SIGMA, cost_rate=0.001).run(t, S)
    np.testing.assert_allclose(costly.gross_pnl, free.pnl)
    assert np.all(costly.costs > 0)
    coarse = DeltaHedger(CALL, R, SIGMA, cost_rate=0.001).run(t[::40], S[:, ::40])
    assert coarse.costs.mean() < costly.costs.mean()


def test_rejects_bad_grids():
    hedger = DeltaHedger(CALL, R, SIGMA)
    t, S = paths(n_steps=10, n_paths=3)
    with pytest.raises(ValueError):
        hedger.run(t[:-1], S[:, :-1])          # stops before maturity
    with pytest.raises(ValueError):
        hedger.run(t, S[:, :-1])               # shape mismatch
    with pytest.raises(NotImplementedError):
        DeltaHedger(Call(100.0, T, american=True), R, SIGMA)
