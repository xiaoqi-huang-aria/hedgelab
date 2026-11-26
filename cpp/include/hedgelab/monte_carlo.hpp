#pragma once

#include <cstdint>

namespace hedgelab {

struct MCResult {
    double price;
    double std_error;
    long long n_samples;  // antithetic: number of pairs
};

// European call/put by exact sampling of S_T under the risk-neutral measure.
// n_threads = 0 uses all hardware threads; the result does not depend on it.
MCResult mc_european(double spot, double strike, double maturity, double rate, double sigma,
                     bool is_call, long long n_paths, std::uint64_t seed, bool antithetic,
                     int n_threads = 0);

}  // namespace hedgelab
