// pybind11 bindings: exposes the C++ core as hedgelab._core.
// Heavy calls release the GIL so other Python threads can run meanwhile.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "hedgelab/binomial.hpp"
#include "hedgelab/black_scholes.hpp"
#include "hedgelab/hedging.hpp"
#include "hedgelab/monte_carlo.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace hedgelab;

namespace {

// forcecast + c_style: numpy converts any input to a contiguous float64 copy if needed.
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::tuple to_arrays(const std::vector<PathPnL>& results) {
    const auto n = static_cast<py::ssize_t>(results.size());
    py::array_t<double> pnl(n), cost(n), gamma_pnl(n);
    auto p = pnl.mutable_unchecked<1>();
    auto c = cost.mutable_unchecked<1>();
    auto g = gamma_pnl.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < n; ++i) {
        p(i) = results[i].pnl;
        c(i) = results[i].cost;
        g(i) = results[i].gamma_pnl;
    }
    return py::make_tuple(pnl, cost, gamma_pnl);
}

py::tuple hedge_paths(const Array& t, const Array& paths, double strike, double maturity, double rate,
                      double hedge_sigma, bool is_call, double cost_rate, int n_threads) {
    if (t.ndim() != 1 || paths.ndim() != 2 || paths.shape(1) != t.shape(0))
        throw std::invalid_argument("t must be 1-D and paths 2-D with paths.shape[1] == len(t)");
    const DeltaHedger hedger({strike, maturity, rate, hedge_sigma, is_call, cost_rate});
    std::vector<PathPnL> out(static_cast<size_t>(paths.shape(0)));
    {
        py::gil_scoped_release release;
        hedger.run_paths(t.data(), static_cast<int>(t.shape(0)), paths.data(), paths.shape(0), out.data(), n_threads);
    }
    return to_arrays(out);
}

py::tuple simulate_and_hedge(double spot, double drift, double sigma, int n_steps, long long n_paths,
                             std::uint64_t seed, double strike, double maturity, double rate,
                             double hedge_sigma, bool is_call, double cost_rate, int n_threads) {
    const DeltaHedger hedger({strike, maturity, rate, hedge_sigma, is_call, cost_rate});
    if (n_paths < 1) throw std::invalid_argument("n_paths must be positive");
    std::vector<PathPnL> out(static_cast<size_t>(n_paths));
    {
        py::gil_scoped_release release;
        hedger.simulate_and_run(spot, drift, sigma, n_steps, n_paths, seed, out.data(), n_threads);
    }
    return to_arrays(out);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core of hedgelab: Black-Scholes, CRR tree, Monte Carlo and delta hedging";

    // py::vectorize turns scalar functions into numpy ufunc-style functions with broadcasting.
    m.def("bs_price", py::vectorize(bs::price), "spot"_a, "strike"_a, "tau"_a, "rate"_a, "sigma"_a, "is_call"_a = true);
    m.def("bs_delta", py::vectorize(bs::delta), "spot"_a, "strike"_a, "tau"_a, "rate"_a, "sigma"_a, "is_call"_a = true);
    m.def("bs_gamma", py::vectorize(bs::gamma), "spot"_a, "strike"_a, "tau"_a, "rate"_a, "sigma"_a);
    m.def("bs_vega", py::vectorize(bs::vega), "spot"_a, "strike"_a, "tau"_a, "rate"_a, "sigma"_a);

    m.def("crr_price", &crr_price, "spot"_a, "strike"_a, "maturity"_a, "rate"_a, "sigma"_a, "n_steps"_a,
          "is_call"_a = true, "american"_a = false, py::call_guard<py::gil_scoped_release>());

    py::class_<MCResult>(m, "MCResult")
        .def_readonly("price", &MCResult::price)
        .def_readonly("std_error", &MCResult::std_error)
        .def_readonly("n_samples", &MCResult::n_samples)
        .def("__repr__", [](const MCResult& r) {
            return "MCResult(price=" + std::to_string(r.price) + ", std_error=" + std::to_string(r.std_error)
                   + ", n_samples=" + std::to_string(r.n_samples) + ")";
        });
    m.def("mc_european", &mc_european, "spot"_a, "strike"_a, "maturity"_a, "rate"_a, "sigma"_a,
          "is_call"_a = true, "n_paths"_a = 1'000'000, "seed"_a = 0, "antithetic"_a = false, "n_threads"_a = 0,
          py::call_guard<py::gil_scoped_release>());

    m.def("hedge_paths", &hedge_paths, "t"_a, "paths"_a, "strike"_a, "maturity"_a, "rate"_a, "hedge_sigma"_a,
          "is_call"_a = true, "cost_rate"_a = 0.0, "n_threads"_a = 0,
          "Hedge given paths; returns (pnl, costs, gamma_pnl) arrays.");
    m.def("simulate_and_hedge", &simulate_and_hedge, "spot"_a, "drift"_a, "sigma"_a, "n_steps"_a, "n_paths"_a,
          "seed"_a, "strike"_a, "maturity"_a, "rate"_a, "hedge_sigma"_a, "is_call"_a = true, "cost_rate"_a = 0.0,
          "n_threads"_a = 0, "Simulate GBM paths and hedge them; returns (pnl, costs, gamma_pnl) arrays.");
}
