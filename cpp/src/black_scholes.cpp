#include "hedgelab/black_scholes.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hedgelab::bs {
namespace {

constexpr double kInvSqrt2 = 0.70710678118654752440;
constexpr double kInvSqrt2Pi = 0.39894228040143267794;

void check_tau(double tau) {
    if (tau < 0.0) throw std::invalid_argument("time to maturity must be nonnegative");
}

double d1(double spot, double strike, double tau, double rate, double sigma) {
    return (std::log(spot / strike) + (rate + 0.5 * sigma * sigma) * tau) / (sigma * std::sqrt(tau));
}

}  // namespace

double norm_cdf(double x) { return 0.5 * std::erfc(-x * kInvSqrt2); }

double norm_pdf(double x) { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }

double price(double spot, double strike, double tau, double rate, double sigma, bool is_call) {
    check_tau(tau);
    if (tau == 0.0) return is_call ? std::max(spot - strike, 0.0) : std::max(strike - spot, 0.0);
    const double d_1 = d1(spot, strike, tau, rate, sigma);
    const double d_2 = d_1 - sigma * std::sqrt(tau);
    const double df_strike = strike * std::exp(-rate * tau);
    return is_call ? spot * norm_cdf(d_1) - df_strike * norm_cdf(d_2)
                   : df_strike * norm_cdf(-d_2) - spot * norm_cdf(-d_1);
}

double delta(double spot, double strike, double tau, double rate, double sigma, bool is_call) {
    check_tau(tau);
    double call_delta;
    if (tau == 0.0) {
        call_delta = spot > strike ? 1.0 : (spot < strike ? 0.0 : 0.5);
    } else {
        call_delta = norm_cdf(d1(spot, strike, tau, rate, sigma));
    }
    return is_call ? call_delta : call_delta - 1.0;
}

double gamma(double spot, double strike, double tau, double rate, double sigma) {
    check_tau(tau);
    if (tau == 0.0) return 0.0;
    return norm_pdf(d1(spot, strike, tau, rate, sigma)) / (spot * sigma * std::sqrt(tau));
}

double vega(double spot, double strike, double tau, double rate, double sigma) {
    check_tau(tau);
    if (tau == 0.0) return 0.0;
    return spot * norm_pdf(d1(spot, strike, tau, rate, sigma)) * std::sqrt(tau);
}

}  // namespace hedgelab::bs
