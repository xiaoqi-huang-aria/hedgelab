#include "hedgelab/monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "hedgelab/parallel.hpp"

namespace hedgelab {
namespace {

// Running mean and sum of squared deviations (Welford). merge() combines two
// blocks exactly (Chan et al.), avoiding the cancellation in sum(x^2) - n mean^2.
struct Moments {
    long long n = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void add(double x) {
        ++n;
        const double d = x - mean;
        mean += d / static_cast<double>(n);
        m2 += d * (x - mean);
    }

    void merge(const Moments& other) {
        if (other.n == 0) return;
        const long long total = n + other.n;
        const double d = other.mean - mean;
        mean += d * static_cast<double>(other.n) / static_cast<double>(total);
        m2 += other.m2 + d * d * static_cast<double>(n) * static_cast<double>(other.n) / static_cast<double>(total);
        n = total;
    }
};

constexpr long long kBlockSize = 16384;

}  // namespace

MCResult mc_european(double spot, double strike, double maturity, double rate, double sigma,
                     bool is_call, long long n_paths, std::uint64_t seed, bool antithetic,
                     int n_threads) {
    if (spot <= 0.0 || strike <= 0.0 || maturity <= 0.0 || sigma <= 0.0)
        throw std::invalid_argument("spot, strike, maturity and sigma must be positive");
    const long long n_samples = antithetic ? n_paths / 2 : n_paths;
    if (n_samples < 2) throw std::invalid_argument("need at least 2 samples");

    const double drift = (rate - 0.5 * sigma * sigma) * maturity;
    const double vol = sigma * std::sqrt(maturity);
    const double discount = std::exp(-rate * maturity);
    auto discounted_payoff = [&](double z) {
        const double s = spot * std::exp(drift + vol * z);
        return discount * (is_call ? std::max(s - strike, 0.0) : std::max(strike - s, 0.0));
    };

    std::vector<Moments> blocks((n_samples + kBlockSize - 1) / kBlockSize);
    parallel_blocks(n_samples, kBlockSize, n_threads, [&](long long b, long long begin, long long end) {
        auto rng = block_rng(seed, b);
        std::normal_distribution<double> normal;
        Moments m;
        for (long long i = begin; i < end; ++i) {
            const double z = normal(rng);
            // Antithetic: average the pair first; pairs are i.i.d., the two halves are not.
            m.add(antithetic ? 0.5 * (discounted_payoff(z) + discounted_payoff(-z)) : discounted_payoff(z));
        }
        blocks[b] = m;
    });

    Moments total;
    for (const auto& m : blocks) total.merge(m);  // fixed order: deterministic result
    const double variance = total.m2 / static_cast<double>(total.n - 1);
    return {total.mean, std::sqrt(variance / static_cast<double>(total.n)), total.n};
}

}  // namespace hedgelab
