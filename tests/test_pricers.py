import numpy as np
import pytest

from hedgelab import (AnalyticPricer, BinomialPricer, BlackScholesModel, Call, DigitalCall,
                    MonteCarloPricer, Put)
from hedgelab import black_scholes as bs

MODEL = BlackScholesModel(spot=100.0, rate=0.05, sigma=0.2)
CALL, PUT = Call(100.0, 1.0), Put(100.0, 1.0)


def test_black_scholes_reference_value():
    # Hull, Options, Futures and Other Derivatives: S=K=100, r=5%, sigma=20%, T=1
    assert AnalyticPricer(MODEL).price(CALL) == pytest.approx(10.450584, abs=1e-6)


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_put_call_parity(strike):
    a = AnalyticPricer(MODEL)
    lhs = a.price(Call(strike, 1.0)) - a.price(Put(strike, 1.0))
    assert lhs == pytest.approx(MODEL.spot - strike * np.exp(-MODEL.rate), abs=1e-10)


@pytest.mark.parametrize("option", [CALL, PUT, Call(90.0, 0.5), Put(115.0, 2.0)])
def test_greeks_match_finite_differences(option):
    a = AnalyticPricer(MODEL)
    h = 1e-3
    up = AnalyticPricer(BlackScholesModel(MODEL.spot + h, MODEL.rate, MODEL.sigma)).price(option)
    down = AnalyticPricer(BlackScholesModel(MODEL.spot - h, MODEL.rate, MODEL.sigma)).price(option)
    mid = a.price(option)
    assert a.delta(option) == pytest.approx((up - down) / (2 * h), abs=1e-6)
    assert a.gamma(option) == pytest.approx((up - 2 * mid + down) / h**2, abs=1e-4)
    hv = 1e-4  # central-difference error is O(hv^2); 1e-3 is too coarse for short-dated ITM vega
    vol_up = AnalyticPricer(BlackScholesModel(MODEL.spot, MODEL.rate, MODEL.sigma + hv)).price(option)
    vol_down = AnalyticPricer(BlackScholesModel(MODEL.spot, MODEL.rate, MODEL.sigma - hv)).price(option)
    assert a.vega(option) == pytest.approx((vol_up - vol_down) / (2 * hv), rel=1e-6)


def test_expiry_limits():
    spots = np.array([90.0, 100.0, 110.0])
    np.testing.assert_allclose(bs.price(spots, 100.0, 0.0, 0.05, 0.2), [0.0, 0.0, 10.0])
    np.testing.assert_allclose(bs.delta(spots, 100.0, 0.0, 0.05, 0.2), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(bs.gamma(spots, 100.0, 0.0, 0.05, 0.2), 0.0)


@pytest.mark.parametrize("option", [CALL, PUT])
def test_binomial_converges_to_black_scholes(option):
    exact = AnalyticPricer(MODEL).price(option)
    # CRR error is O(1/n) with oscillation; 2000 steps gets within 1 cent.
    assert BinomialPricer(MODEL, 2000).price(option) == pytest.approx(exact, abs=0.01)


@pytest.mark.parametrize("n, expected", [(4, 9.97052292), (10, 10.25340904),
                                         (25, 10.52096562), (50, 10.41069154)])
def test_binomial_european_call_reference_values(n, expected):
    # From an independent CRR implementation; n = 4 is small enough to check by hand.
    assert BinomialPricer(MODEL, n).price(CALL) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("n, expected", [(4, 5.882800), (10, 6.004259),
                                         (25, 6.146185), (50, 6.073728)])
def test_binomial_american_put_reference_values(n, expected):
    assert BinomialPricer(MODEL, n).price(Put(100.0, 1.0, american=True)) == pytest.approx(expected, abs=1e-6)


def test_american_call_without_dividends_equals_european():
    tree = BinomialPricer(MODEL, 500)
    assert tree.price(Call(100.0, 1.0, american=True)) == pytest.approx(tree.price(CALL), abs=1e-10)


def test_american_put_exceeds_european_and_intrinsic():
    tree = BinomialPricer(MODEL, 500)
    for strike in (90.0, 100.0, 130.0):
        american = tree.price(Put(strike, 1.0, american=True))
        assert american >= tree.price(Put(strike, 1.0)) - 1e-12
        assert american >= max(strike - MODEL.spot, 0.0) - 1e-12


@pytest.mark.parametrize("antithetic", [False, True])
def test_monte_carlo_within_four_standard_errors(antithetic):
    result = MonteCarloPricer(MODEL, 200_000, seed=11, antithetic=antithetic).estimate(CALL)
    assert abs(result.price - AnalyticPricer(MODEL).price(CALL)) < 4 * result.std_error


def test_antithetic_reduces_standard_error_at_equal_cost():
    plain = MonteCarloPricer(MODEL, 200_000, seed=3).estimate(CALL)
    anti = MonteCarloPricer(MODEL, 200_000, seed=3, antithetic=True).estimate(CALL)
    assert anti.std_error < plain.std_error


def test_importance_sampling_on_deep_otm_digital():
    # Deep out-of-the-money digital: P(S_T >= 150) with S0 = 100, sigma = 40%, T = 2
    model = BlackScholesModel(spot=100.0, rate=0.02, sigma=0.4)
    digital = DigitalCall(150.0, 2.0)
    exact = AnalyticPricer(model).price(digital)
    plain = MonteCarloPricer(model, 100_000, seed=5).estimate(digital)
    shifted = MonteCarloPricer(model, 100_000, seed=5, shift=0.6).estimate(digital)
    assert abs(shifted.price - exact) < 4 * shifted.std_error
    assert shifted.std_error < 0.7 * plain.std_error


def test_rejects_unsupported_combinations():
    with pytest.raises(NotImplementedError):
        AnalyticPricer(MODEL).price(Put(100.0, 1.0, american=True))
    with pytest.raises(NotImplementedError):
        MonteCarloPricer(MODEL).price(Put(100.0, 1.0, american=True))
    with pytest.raises(ValueError):
        Call(-1.0, 1.0)
