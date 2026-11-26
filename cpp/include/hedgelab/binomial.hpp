#pragma once

namespace hedgelab {

// Cox-Ross-Rubinstein tree priced by backward induction, O(n) memory and O(n^2)
// time. Node i at step j has i down moves: S = spot * u^(j - 2i), as in hedgelab.
// Throws std::invalid_argument unless d < exp(r h) < u.
double crr_price(double spot, double strike, double maturity, double rate, double sigma,
                 int n_steps, bool is_call, bool american);

}  // namespace hedgelab
