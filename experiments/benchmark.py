"""Speed of the C++ core against the NumPy reference and a pure-Python loop.

Every comparison runs the same computation (results are cross-checked in
tests/test_cpp_core.py). Times are the best of several runs;
the pure-Python loop is timed on a subset of paths and scaled up linearly.

    python experiments/benchmark.py
"""

import math
import os
import platform
import time

import numpy as np

from hedgelab import AnalyticPricer, BinomialPricer, BlackScholesModel, Call, DeltaHedger, MonteCarloPricer, Put
from hedgelab import _core

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2
MODEL = BlackScholesModel(S0, R, SIGMA)
CALL = Call(K, T)
N_THREADS = os.cpu_count() or 1


def best_time(fn, repeat=3):
    fn()  # warm-up
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def hedge_path_pure_python(t, s, strike, maturity, rate, sigma, cost_rate=0.0):
    """Same outputs as DeltaHedger for one call path (P&L, costs, gamma P&L), math module only."""
    def ncdf(x):
        return 0.5 * math.erfc(-x / math.sqrt(2.0))

    def bs_call(spot, tau):
        v = sigma * math.sqrt(tau)
        d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * tau) / v
        return spot * ncdf(d1) - strike * math.exp(-rate * tau) * ncdf(d1 - v)

    gains = traded = gamma_pnl = held = 0.0
    for i in range(len(t) - 1):
        tau = maturity - t[i]
        vol_sqrt_t = sigma * math.sqrt(tau)
        d1 = (math.log(s[i] / strike) + (rate + 0.5 * sigma * sigma) * tau) / vol_sqrt_t
        delta = ncdf(d1)
        gamma = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi) / (s[i] * vol_sqrt_t)
        gains += delta * (s[i + 1] * math.exp(-rate * t[i + 1]) - s[i] * math.exp(-rate * t[i]))
        traded += abs(delta - held) * s[i] * math.exp(rate * tau)
        ret = (s[i + 1] - s[i]) / s[i]
        gamma_pnl += 0.5 * gamma * s[i] ** 2 * (sigma**2 * (t[i + 1] - t[i]) - ret**2) * math.exp(rate * (maturity - t[i + 1]))
        held = delta
    cost = cost_rate * (traded + abs(held) * s[-1])
    pnl = math.exp(rate * maturity) * (bs_call(s[0], maturity) + gains) - max(s[-1] - strike, 0.0) - cost
    return pnl, cost, gamma_pnl


def row(case, impl, seconds, ref=None):
    speedup = f"{ref / seconds:,.1f}x" if ref else ""
    return f"| {case} | {impl} | {seconds * 1e3:,.1f} ms | {speedup} |"


def main():
    rows = []
    hedger = DeltaHedger(CALL, R, SIGMA)

    # 1. Hedge given paths: 20,000 paths x 252 daily steps.
    n_paths, n_steps = 20_000, 252
    t, S = MODEL.simulate_paths(T, n_steps, n_paths, rng=0)
    t_list, S_list = t.tolist(), S[:500].tolist()
    reference = hedger.run(t, S[:20])
    check = np.array([hedge_path_pure_python(t_list, p, K, T, R, SIGMA) for p in S_list[:20]])
    assert np.allclose(check[:, 0], reference.pnl, atol=1e-9), "pure-Python P&L disagrees with NumPy"
    assert np.allclose(check[:, 2], reference.gamma_pnl, atol=1e-9), "pure-Python gamma P&L disagrees"
    t0 = time.perf_counter()
    for p in S_list:
        hedge_path_pure_python(t_list, p, K, T, R, SIGMA)
    pure = (time.perf_counter() - t0) * n_paths / len(S_list)
    numpy_time = best_time(lambda: hedger.run(t, S))
    case = f"Delta hedge {n_paths:,} given paths x {n_steps} steps"
    rows += [row(case, "pure Python loop (scaled from 500 paths)", pure),
             row(case, "NumPy, vectorized", numpy_time),
             row(case, "C++, 1 thread", best_time(lambda: hedger.run(t, S, "cpp", 1)), numpy_time),
             row(case, f"C++, {N_THREADS} threads", best_time(lambda: hedger.run(t, S, "cpp", 0)), numpy_time)]
    cpp_one_thread_hedge = best_time(lambda: hedger.run(t, S, "cpp", 1))
    pure_vs_cpp = pure / cpp_one_thread_hedge

    # 2. Simulate and hedge 200,000 paths x 252 steps (NumPy in 20,000-path chunks to bound memory).
    n_big = 200_000

    def numpy_sim_and_hedge():
        for seed in range(n_big // 20_000):
            tc, Sc = MODEL.simulate_paths(T, n_steps, 20_000, rng=seed)
            hedger.run(tc, Sc)

    numpy_time = best_time(numpy_sim_and_hedge, repeat=2)
    case = f"Simulate + hedge {n_big:,} paths x {n_steps} steps"
    rows += [row(case, "NumPy, vectorized (20k-path chunks)", numpy_time),
             row(case, "C++, 1 thread", best_time(lambda: hedger.simulate_gbm(MODEL, n_steps, n_big, seed=1, n_threads=1), 2), numpy_time),
             row(case, f"C++, {N_THREADS} threads", best_time(lambda: hedger.simulate_gbm(MODEL, n_steps, n_big, seed=1), 2), numpy_time)]

    # 3. American put on a 5,000-step CRR tree.
    n_tree = 5_000
    put = Put(K, T, american=True)
    numpy_time = best_time(lambda: BinomialPricer(MODEL, n_tree).price(put))
    case = f"American put, CRR {n_tree:,} steps"
    rows += [row(case, "NumPy, vectorized per level", numpy_time),
             row(case, "C++", best_time(lambda: _core.crr_price(S0, K, T, R, SIGMA, n_tree, False, True)), numpy_time)]

    # 4. European call, 10 million Monte Carlo samples.
    n_mc = 10_000_000
    numpy_time = best_time(lambda: MonteCarloPricer(MODEL, n_mc, seed=1).estimate(CALL))
    case = f"European call MC, {n_mc // 10**6}M samples"
    rows += [row(case, "NumPy, vectorized", numpy_time),
             row(case, "C++, 1 thread", best_time(lambda: _core.mc_european(S0, K, T, R, SIGMA, True, n_mc, 1, False, 1)), numpy_time),
             row(case, f"C++, {N_THREADS} threads", best_time(lambda: _core.mc_european(S0, K, T, R, SIGMA, True, n_mc, 1, False, 0)), numpy_time)]
    exact = AnalyticPricer(MODEL).price(CALL)
    mc = _core.mc_european(S0, K, T, R, SIGMA, True, n_mc, 1)
    assert abs(mc.price - exact) < 4 * mc.std_error

    print(f"Machine: {platform.machine()} / {platform.system()} {platform.release()}, "
          f"{N_THREADS} hardware threads, Python {platform.python_version()}, NumPy {np.__version__}\n")
    print("| task | implementation | time | speed-up vs NumPy |\n|---|---|---|---|")
    print("\n".join(rows))
    print(f"\nC++ (1 thread) vs pure-Python loop on the hedging task: {pure_vs_cpp:,.0f}x")


if __name__ == "__main__":
    main()
