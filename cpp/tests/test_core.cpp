// Unit tests for the C++ core. A tiny CHECK harness keeps the build free of
// external test frameworks; run with `ctest --test-dir build` or ./build/test_hedgelab.

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "hedgelab/binomial.hpp"
#include "hedgelab/black_scholes.hpp"
#include "hedgelab/hedging.hpp"
#include "hedgelab/monte_carlo.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", file, line, expr);
    }
}

void check_near(double a, double b, double tol, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!(std::abs(a - b) <= tol)) {
        ++g_failures;
        std::fprintf(stderr, "%s:%d: CHECK_NEAR failed: %s  (%.12g vs %.12g, tol %.3g)\n", file, line, expr, a, b, tol);
    }
}

template <class F>
void check_throws(F f, const char* expr, const char* file, int line) {
    bool threw = false;
    try {
        f();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, expr, file, line);
}

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) check_near((a), (b), (tol), #a " ~ " #b, __FILE__, __LINE__)
#define CHECK_THROWS(stmt) check_throws([&] { stmt; }, #stmt, __FILE__, __LINE__)

using namespace hedgelab;

constexpr double S0 = 100.0, K = 100.0, T = 1.0, R = 0.05, SIGMA = 0.2;

void test_black_scholes() {
    CHECK_NEAR(bs::price(S0, K, T, R, SIGMA, true), 10.450583572185565, 1e-12);  // Hull's example
    for (double k : {80.0, 100.0, 120.0}) {
        const double parity = bs::price(S0, k, T, R, SIGMA, true) - bs::price(S0, k, T, R, SIGMA, false);
        CHECK_NEAR(parity, S0 - k * std::exp(-R * T), 1e-12);
    }
    const double h = 1e-3;
    for (bool is_call : {true, false}) {
        const double up = bs::price(S0 + h, K, T, R, SIGMA, is_call);
        const double mid = bs::price(S0, K, T, R, SIGMA, is_call);
        const double down = bs::price(S0 - h, K, T, R, SIGMA, is_call);
        CHECK_NEAR(bs::delta(S0, K, T, R, SIGMA, is_call), (up - down) / (2 * h), 1e-7);
        CHECK_NEAR(bs::gamma(S0, K, T, R, SIGMA), (up - 2 * mid + down) / (h * h), 1e-4);
    }
    const double hv = 1e-5;
    CHECK_NEAR(bs::vega(S0, K, T, R, SIGMA),
               (bs::price(S0, K, T, R, SIGMA + hv, true) - bs::price(S0, K, T, R, SIGMA - hv, true)) / (2 * hv), 1e-5);
    CHECK_NEAR(bs::price(110.0, K, 0.0, R, SIGMA, true), 10.0, 0.0);
    CHECK_NEAR(bs::delta(100.0, K, 0.0, R, SIGMA, true), 0.5, 0.0);
    CHECK_NEAR(bs::gamma(100.0, K, 0.0, R, SIGMA), 0.0, 0.0);
    CHECK_THROWS(bs::price(S0, K, -1.0, R, SIGMA, true));
}

void test_binomial() {
    // Reference values from an independent CRR implementation (n = 4 can be checked by hand).
    CHECK_NEAR(crr_price(S0, K, T, R, SIGMA, 4, true, false), 9.97052292, 1e-8);
    CHECK_NEAR(crr_price(S0, K, T, R, SIGMA, 50, true, false), 10.41069154, 1e-8);
    CHECK_NEAR(crr_price(S0, K, T, R, SIGMA, 4, false, true), 5.882800, 1e-6);
    CHECK_NEAR(crr_price(S0, K, T, R, SIGMA, 25, false, true), 6.146185, 1e-6);
    CHECK_NEAR(crr_price(S0, K, T, R, SIGMA, 2000, true, false), bs::price(S0, K, T, R, SIGMA, true), 0.01);
    // No dividends: early exercise of a call is never optimal.
    CHECK_NEAR(crr_price(S0, K, T, R, SIGMA, 500, true, true), crr_price(S0, K, T, R, SIGMA, 500, true, false), 1e-10);
    for (double k : {90.0, 100.0, 130.0}) {
        const double american = crr_price(S0, k, T, R, SIGMA, 500, false, true);
        CHECK(american >= crr_price(S0, k, T, R, SIGMA, 500, false, false) - 1e-12);
        CHECK(american >= std::max(k - S0, 0.0) - 1e-12);
    }
    CHECK_THROWS(crr_price(S0, K, T, 0.5, 0.01, 1, true, false));  // exp(r h) > u
    CHECK_THROWS(crr_price(S0, K, T, R, SIGMA, 0, true, false));
}

void test_monte_carlo() {
    const double exact = bs::price(S0, K, T, R, SIGMA, true);
    const MCResult plain = mc_european(S0, K, T, R, SIGMA, true, 1'000'000, 7, false, 0);
    CHECK(std::abs(plain.price - exact) < 4 * plain.std_error);
    const MCResult anti = mc_european(S0, K, T, R, SIGMA, true, 1'000'000, 7, true, 0);
    CHECK(std::abs(anti.price - exact) < 4 * anti.std_error);
    CHECK(anti.std_error < plain.std_error);
    CHECK(anti.n_samples == 500'000);
    // Per-block random streams: the result must not depend on the thread count.
    const MCResult one = mc_european(S0, K, T, R, SIGMA, false, 200'000, 3, false, 1);
    const MCResult many = mc_european(S0, K, T, R, SIGMA, false, 200'000, 3, false, 7);
    CHECK(one.price == many.price && one.std_error == many.std_error);
    CHECK_THROWS(mc_european(S0, K, T, R, -0.2, true, 1000, 1, false, 1));
}

double mean(const std::vector<PathPnL>& v) {
    double s = 0.0;
    for (const auto& x : v) s += x.pnl;
    return s / v.size();
}

double stdev(const std::vector<PathPnL>& v) {
    const double m = mean(v);
    double s = 0.0;
    for (const auto& x : v) s += (x.pnl - m) * (x.pnl - m);
    return std::sqrt(s / (v.size() - 1));
}

void test_hedging() {
    const DeltaHedger hedger({K, T, R, SIGMA, true, 0.0});

    // One interval: a static hedge, checkable by hand.
    const double t[2] = {0.0, T};
    const double path[2] = {S0, 112.0};
    PathPnL one;
    hedger.run_paths(t, 2, path, 1, &one, 1);
    const double expected = std::exp(R * T) * bs::price(S0, K, T, R, SIGMA, true)
                            + bs::delta(S0, K, T, R, SIGMA, true) * (112.0 - S0 * std::exp(R * T)) - 12.0;
    CHECK_NEAR(one.pnl, expected, 1e-12);

    // Matched vols and risk-neutral drift: E[P&L] = 0 exactly.
    std::vector<PathPnL> daily(40'000), weekly(40'000);
    hedger.simulate_and_run(S0, R, SIGMA, 252, daily.size(), 11, daily.data(), 0);
    CHECK(std::abs(mean(daily)) < 4 * stdev(daily) / std::sqrt(daily.size()));
    // Hedging error std ~ sqrt(dt): 252 vs 63 steps should be about a factor of 2.
    hedger.simulate_and_run(S0, R, SIGMA, 63, weekly.size(), 12, weekly.data(), 0);
    CHECK_NEAR(stdev(weekly) / stdev(daily), 2.0, 0.1);

    // Thread-count invariance.
    std::vector<PathPnL> a(1000), b(1000);
    hedger.simulate_and_run(S0, R, SIGMA, 50, 1000, 5, a.data(), 1);
    hedger.simulate_and_run(S0, R, SIGMA, 50, 1000, 5, b.data(), 6);
    bool same = true;
    for (size_t i = 0; i < a.size(); ++i) same = same && a[i].pnl == b[i].pnl;
    CHECK(same);

    // Costs are charged and subtracted.
    const DeltaHedger costly({K, T, R, SIGMA, true, 0.001});
    std::vector<PathPnL> c(1000);
    costly.simulate_and_run(S0, R, SIGMA, 50, 1000, 5, c.data(), 0);
    CHECK(c[0].cost > 0.0);
    CHECK_NEAR(c[0].pnl + c[0].cost, a[0].pnl, 1e-12);

    const double short_grid[2] = {0.0, 0.5};
    CHECK_THROWS(hedger.run_paths(short_grid, 2, path, 1, &one, 1));
    CHECK_THROWS(DeltaHedger({K, T, R, -0.1, true, 0.0}));
}

}  // namespace

int main() {
    test_black_scholes();
    test_binomial();
    test_monte_carlo();
    test_hedging();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
