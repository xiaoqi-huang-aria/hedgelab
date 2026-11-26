#pragma once

// Black-Scholes formulas without dividends; tau is time to maturity. At tau = 0
// the functions return expiry limits: the payoff, call delta 1 / 0.5 / 0, and
// zero gamma and vega (same conventions as hedgelab.black_scholes).

namespace hedgelab::bs {

double norm_cdf(double x);
double norm_pdf(double x);

double price(double spot, double strike, double tau, double rate, double sigma, bool is_call);
double delta(double spot, double strike, double tau, double rate, double sigma, bool is_call);
double gamma(double spot, double strike, double tau, double rate, double sigma);
double vega(double spot, double strike, double tau, double rate, double sigma);

}  // namespace hedgelab::bs
