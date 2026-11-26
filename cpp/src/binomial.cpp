#include "hedgelab/binomial.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace hedgelab {

double crr_price(double spot, double strike, double maturity, double rate, double sigma,
                 int n_steps, bool is_call, bool american) {
    if (n_steps < 1) throw std::invalid_argument("n_steps must be positive");
    if (spot <= 0.0 || strike <= 0.0 || maturity <= 0.0 || sigma <= 0.0)
        throw std::invalid_argument("spot, strike, maturity and sigma must be positive");

    const double h = maturity / n_steps;
    const double log_u = sigma * std::sqrt(h);
    const double u = std::exp(log_u);
    const double d = 1.0 / u;
    const double growth = std::exp(rate * h);
    if (!(d < growth && growth < u))
        throw std::invalid_argument("no-arbitrage requires d < exp(r h) < u; increase n_steps");
    const double p = (growth - d) / (u - d);
    const double discount = 1.0 / growth;

    // spot * u^k for k = -n..n, so no pow() call inside the O(n^2) loop.
    std::vector<double> level_spot(2 * n_steps + 1);
    for (int k = -n_steps; k <= n_steps; ++k) level_spot[k + n_steps] = spot * std::exp(k * log_u);
    auto node_spot = [&](int j, int i) { return level_spot[j - 2 * i + n_steps]; };
    auto payoff = [&](double s) { return is_call ? std::max(s - strike, 0.0) : std::max(strike - s, 0.0); };

    std::vector<double> value(n_steps + 1);
    for (int i = 0; i <= n_steps; ++i) value[i] = payoff(node_spot(n_steps, i));

    // In-place backward induction: value[i] needs the old value[i] and value[i + 1],
    // and value[i + 1] is overwritten only after value[i] is computed.
    for (int j = n_steps - 1; j >= 0; --j) {
        for (int i = 0; i <= j; ++i) {
            const double continuation = discount * (p * value[i] + (1.0 - p) * value[i + 1]);
            value[i] = american ? std::max(continuation, payoff(node_spot(j, i))) : continuation;
        }
    }
    return value[0];
}

}  // namespace hedgelab
