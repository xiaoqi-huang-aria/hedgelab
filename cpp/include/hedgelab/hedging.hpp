#pragma once

#include <cstdint>

namespace hedgelab {

struct HedgeSpec {
    double strike;
    double maturity;
    double rate;
    double hedge_sigma;      // implied vol used for the premium and the deltas
    bool is_call = true;
    double cost_rate = 0.0;  // proportional cost, fraction of traded notional
};

struct PathPnL {
    double pnl;        // seller P&L net of costs, time-T dollars
    double cost;       // transaction costs, time-T dollars
    double gamma_pnl;  // sum of 1/2 Gamma S^2 (sigma^2 dt - (dS/S)^2), carried to T
};

// Discrete delta hedge of a short European option: the same model, formulas and sign
// convention as hedgelab.hedging.DeltaHedger, evaluated one path at a time.
class DeltaHedger {
public:
    explicit DeltaHedger(const HedgeSpec& spec);

    // Paths are row-major (n_paths x n_dates) and observed on dates t[0..n_dates-1],
    // with t[0] = 0, t[n_dates-1] = maturity, strictly increasing.
    void run_paths(const double* t, int n_dates, const double* paths, long long n_paths,
                   PathPnL* out, int n_threads = 0) const;

    // Simulates GBM paths (exact log-normal steps on a uniform grid) and hedges each one
    // as it is generated: O(n_steps) memory per thread instead of O(paths x steps).
    void simulate_and_run(double spot, double drift, double sigma, int n_steps, long long n_paths,
                          std::uint64_t seed, PathPnL* out, int n_threads = 0) const;

private:
    struct Grid;  // per-date quantities shared by every path
    Grid make_grid(const double* t, int n_dates) const;
    PathPnL run_path(const Grid& grid, const double* s) const;

    HedgeSpec spec_;
};

}  // namespace hedgelab
