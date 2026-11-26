#include "hedgelab/hedging.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "hedgelab/black_scholes.hpp"
#include "hedgelab/parallel.hpp"

namespace hedgelab {
namespace {
constexpr long long kPathsPerBlock = 256;
}

struct DeltaHedger::Grid {
    int n;                       // number of hedging intervals
    std::vector<double> dt;      // t[i+1] - t[i]
    std::vector<double> tau;     // maturity - t[i]
    std::vector<double> sqrt_tau;
    std::vector<double> disc;    // exp(-r t[i]):     time-t dollars -> time-0 dollars
    std::vector<double> carry;   // exp(r (T - t[i])): time-t dollars -> time-T dollars
};

DeltaHedger::DeltaHedger(const HedgeSpec& spec) : spec_(spec) {
    if (spec.strike <= 0.0 || spec.maturity <= 0.0)
        throw std::invalid_argument("strike and maturity must be positive");
    if (spec.hedge_sigma <= 0.0 || spec.cost_rate < 0.0)
        throw std::invalid_argument("need hedge_sigma > 0 and cost_rate >= 0");
}

DeltaHedger::Grid DeltaHedger::make_grid(const double* t, int n_dates) const {
    const double T = spec_.maturity;
    if (n_dates < 2) throw std::invalid_argument("need at least 2 dates");
    if (std::abs(t[0]) > 1e-8 || std::abs(t[n_dates - 1] - T) > 1e-8 + 1e-5 * T)
        throw std::invalid_argument("the grid must run from 0 to the option's maturity");
    for (int i = 0; i + 1 < n_dates; ++i)
        if (!(t[i + 1] > t[i])) throw std::invalid_argument("dates must be strictly increasing");

    Grid g;
    g.n = n_dates - 1;
    g.dt.resize(g.n);
    g.tau.resize(g.n);
    g.sqrt_tau.resize(g.n);
    g.disc.resize(n_dates);
    g.carry.resize(n_dates);
    for (int i = 0; i < n_dates; ++i) {
        g.disc[i] = std::exp(-spec_.rate * t[i]);
        g.carry[i] = std::exp(spec_.rate * (T - t[i]));
    }
    for (int i = 0; i < g.n; ++i) {
        g.dt[i] = t[i + 1] - t[i];
        g.tau[i] = std::max(T - t[i], 0.0);
        g.sqrt_tau[i] = std::sqrt(g.tau[i]);
    }
    return g;
}

PathPnL DeltaHedger::run_path(const Grid& g, const double* s) const {
    const double K = spec_.strike, r = spec_.rate, sigma = spec_.hedge_sigma, T = spec_.maturity;
    const double variance_rate = sigma * sigma;

    double gains = 0.0;      // sum_i Delta_i (S~_{i+1} - S~_i), discounted units
    double traded = 0.0;     // sum_i |Delta_i - Delta_{i-1}| S_i, carried to T
    double gamma_pnl = 0.0;
    double held = 0.0;       // shares held before today's trade
    for (int i = 0; i < g.n; ++i) {
        const double si = s[i];
        double delta, gamma;
        if (g.tau[i] > 0.0) {
            // d1 once for both Greeks: the hot loop's only log / erfc / exp.
            const double vol_sqrt_t = sigma * g.sqrt_tau[i];
            const double d1 = (std::log(si / K) + (r + 0.5 * variance_rate) * g.tau[i]) / vol_sqrt_t;
            delta = bs::norm_cdf(d1) - (spec_.is_call ? 0.0 : 1.0);
            gamma = bs::norm_pdf(d1) / (si * vol_sqrt_t);
        } else {
            delta = bs::delta(si, K, 0.0, r, sigma, spec_.is_call);
            gamma = 0.0;
        }
        gains += delta * (s[i + 1] * g.disc[i + 1] - si * g.disc[i]);
        traded += std::abs(delta - held) * si * g.carry[i];
        const double ret = (s[i + 1] - si) / si;
        gamma_pnl += 0.5 * gamma * si * si * (variance_rate * g.dt[i] - ret * ret) * g.carry[i + 1];
        held = delta;
    }

    const double s_T = s[g.n];
    const double cost = spec_.cost_rate * (traded + std::abs(held) * s_T);  // + final unwind
    const double premium = bs::price(s[0], K, T, r, sigma, spec_.is_call);
    const double hedge_value = std::exp(r * T) * (premium + gains);
    const double payoff = spec_.is_call ? std::max(s_T - K, 0.0) : std::max(K - s_T, 0.0);
    return {hedge_value - payoff - cost, cost, gamma_pnl};
}

void DeltaHedger::run_paths(const double* t, int n_dates, const double* paths, long long n_paths,
                            PathPnL* out, int n_threads) const {
    const Grid grid = make_grid(t, n_dates);
    parallel_blocks(n_paths, kPathsPerBlock, n_threads, [&](long long, long long begin, long long end) {
        for (long long p = begin; p < end; ++p) out[p] = run_path(grid, paths + p * n_dates);
    });
}

void DeltaHedger::simulate_and_run(double spot, double drift, double sigma, int n_steps, long long n_paths,
                                   std::uint64_t seed, PathPnL* out, int n_threads) const {
    if (spot <= 0.0 || sigma <= 0.0 || n_steps < 1 || n_paths < 1)
        throw std::invalid_argument("need spot > 0, sigma > 0, n_steps >= 1, n_paths >= 1");
    const double T = spec_.maturity;
    std::vector<double> t(n_steps + 1);
    for (int k = 0; k <= n_steps; ++k) t[k] = T * k / n_steps;
    const Grid grid = make_grid(t.data(), n_steps + 1);

    const double dt = T / n_steps;
    const double mean_step = (drift - 0.5 * sigma * sigma) * dt;
    const double vol_step = sigma * std::sqrt(dt);
    parallel_blocks(n_paths, kPathsPerBlock, n_threads, [&](long long b, long long begin, long long end) {
        auto rng = block_rng(seed, b);
        std::normal_distribution<double> normal;
        std::vector<double> path(n_steps + 1);
        for (long long p = begin; p < end; ++p) {
            path[0] = spot;
            for (int k = 0; k < n_steps; ++k) path[k + 1] = path[k] * std::exp(mean_step + vol_step * normal(rng));
            out[p] = run_path(grid, path.data());
        }
    });
}

}  // namespace hedgelab
