"""Delta-hedging P&L study for a short 1-year ATM call.

1. Hedging error vs rebalancing frequency: std(P&L) should scale like sqrt(dt).
2. Implied vs realized vol: P&L is explained path-by-path by the gamma P&L
   approximation, and on average equals exp(rT) [BS(sigma_i) - BS(sigma_r)].
3. Proportional transaction costs: hedging more often cuts risk but costs more;
   find the frequency that minimizes  mean cost + lambda * std(P&L).

Run from the repository root:
    python experiments/hedging_study.py            # writes figures/*.png
    python experiments/hedging_study.py --paths 2000  # quicker
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

from hedgelab import AnalyticPricer, BlackScholesModel, Call, DeltaHedger

S0, K, T, R, SIGMA_I = 100.0, 100.0, 1.0, 0.05, 0.20
FINE_STEPS = 1560                      # 52 weeks x 5 days x 6 intraday = divisible by 12, 52, 260
FREQUENCIES = [4, 12, 26, 52, 130, 260, 520, 1560]
COST_RATES = [0.0, 0.0005, 0.001, 0.002]
RISK_AVERSION = 1.0                    # lambda in  mean cost + lambda * std
OPTION = Call(K, T)

# Reference palette: categorical slots 1-4 (fixed order), chart chrome, ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlecolor": INK,
        "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "legend.frameon": False, "legend.labelcolor": INK_2,
    })


def end_label(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(10, dy), textcoords="offset points",
                va="center", fontsize=9, color=INK_2)


def frequency_study(t, S, out):
    hedger = DeltaHedger(OPTION, R, SIGMA_I)
    rows, pnls = [], {}
    for n in FREQUENCIES:
        k = FINE_STEPS // n
        res = hedger.run(t[::k], S[:, ::k])
        rows.append((n, T / n, *res.summary().values()))
        pnls[n] = res.pnl
    rows = np.array(rows)
    dt, std = rows[:, 1], rows[:, 3]
    slope, intercept = np.polyfit(np.log(dt), np.log(std), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(-6, 6, 121)
    for color, (n, label) in zip(SERIES, [(12, "monthly"), (52, "weekly"), (260, "daily")]):
        ax1.hist(pnls[n], bins=bins, histtype="step", lw=2, color=color, label=f"{label} (n = {n})")
    ax1.axvline(0, color=AXIS, lw=1)
    ax1.set(title=r"P&L distribution, $\sigma_i = \sigma_r = 20\%$",
            xlabel="Seller P&L at maturity ($)", ylabel="Paths")
    ax1.legend(loc="upper left")

    ax2.loglog(dt, std, "o-", color=SERIES[0], label="simulated std(P&L)")
    ax2.loglog(dt, np.exp(intercept) * dt**0.5, "--", lw=1.2, color=MUTED, label="slope 1/2 reference")
    ax2.set(title=f"Hedging error vs rebalancing interval (fitted slope {slope:.3f})",
            xlabel=r"Rebalancing interval $\Delta t$ (years)", ylabel="std of P&L ($)")
    ax2.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 3]))
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.legend(loc="upper left")
    for x, y, n in zip(dt, std, FREQUENCIES):
        if n in (4, 52, 1560):
            end_label(ax2, x, y, f"n={n}", dy=-10)
    fig.tight_layout()
    fig.savefig(out / "hedging_error_scaling.png", dpi=150)
    plt.close(fig)

    print(f"\n## 1. Rebalancing frequency (fitted log-log slope {slope:.3f})")
    print("| n | dt | mean P&L | std P&L | MC s.e. |\n|---|---|---|---|---|")
    for n, d, mean, s, se, _ in rows:
        print(f"| {int(n)} | {d:.5f} | {mean:+.4f} | {s:.4f} | {se:.4f} |")
    return slope


def vol_mismatch_study(z, out):
    n = 260
    k = FINE_STEPS // n
    z_coarse = z.reshape(z.shape[0], n, k).sum(axis=2) / np.sqrt(k)  # same Brownian path, daily grid
    hedger = DeltaHedger(OPTION, R, SIGMA_I)
    pricer_i = AnalyticPricer(BlackScholesModel(S0, R, SIGMA_I))
    vega = pricer_i.vega(OPTION)

    t, S = BlackScholesModel(S0, R, 0.25).paths_from_normals(z_coarse, T)
    res = hedger.run(t, S)
    corr = np.corrcoef(res.pnl, res.gamma_pnl)[0, 1]

    sigmas_r = np.linspace(0.10, 0.30, 9)
    means, ses, exact = [], [], []
    for sr in sigmas_r:
        # Common random numbers: the same normals drive every realized-vol scenario.
        t, S = BlackScholesModel(S0, R, sr).paths_from_normals(z_coarse, T)
        s = hedger.run(t, S).summary()
        means.append(s["mean"])
        ses.append(s["std_error"])
        exact.append(np.exp(R * T) * (pricer_i.price(OPTION)
                                      - AnalyticPricer(BlackScholesModel(S0, R, sr)).price(OPTION)))
    means, ses, exact = map(np.array, (means, ses, exact))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    m = min(3000, res.pnl.size)
    ax1.scatter(res.gamma_pnl[:m], res.pnl[:m], s=8, alpha=0.35, color=SERIES[0], linewidths=0)
    lim = np.percentile(np.r_[res.pnl, res.gamma_pnl], [0.5, 99.5])
    ax1.plot(lim, lim, "--", lw=1.2, color=MUTED)
    ax1.text(0.03, 0.95, "dashed line: y = x", transform=ax1.transAxes, va="top", fontsize=9, color=INK_2)
    ax1.set(xlim=lim, ylim=lim,
            title=rf"Gamma P&L explains realized P&L ($\sigma_r$ = 25%, corr {corr:.3f})",
            xlabel=r"$\sum_i \frac{1}{2}\Gamma_i S_i^2\,(\sigma_i^2\Delta t - (\Delta S_i/S_i)^2)$  (\$)",
            ylabel="Simulated seller P&L ($)")

    fine = np.linspace(sigmas_r[0], sigmas_r[-1], 200)
    exact_fine = [np.exp(R * T) * (pricer_i.price(OPTION)
                                   - AnalyticPricer(BlackScholesModel(S0, R, sr)).price(OPTION)) for sr in fine]
    ax2.plot(fine, exact_fine, color=SERIES[1], label=r"exact: $e^{rT}[BS(\sigma_i) - BS(\sigma_r)]$")
    ax2.plot(fine, -vega * (fine - SIGMA_I) * np.exp(R * T), "--", lw=1.5, color=SERIES[2],
             label=r"first order: $-\mathcal{V}\,(\sigma_r - \sigma_i)\,e^{rT}$")
    ax2.errorbar(sigmas_r, means, yerr=1.96 * ses, fmt="o", color=SERIES[0], ms=6,
                 mec=SURFACE, mew=1.5, capsize=0, label="simulated mean +/- 1.96 s.e.")
    ax2.axhline(0, color=AXIS, lw=1)
    ax2.axvline(SIGMA_I, color=AXIS, lw=1)
    ax2.set(title=r"Mean P&L vs realized vol ($\sigma_i$ = 20%, daily hedge)",
            xlabel=r"Realized volatility $\sigma_r$", ylabel="Mean seller P&L ($)")
    ax2.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "vol_mismatch.png", dpi=150)
    plt.close(fig)

    print(f"\n## 2. Implied 20% vs realized vol (daily hedge); corr(P&L, gamma P&L) at 25% = {corr:.4f}")
    print(f"mean P&L {res.pnl.mean():+.4f}, mean gamma P&L {res.gamma_pnl.mean():+.4f}")
    print("| sigma_r | simulated mean | MC s.e. | exact | first order |\n|---|---|---|---|---|")
    for sr, mu, se, ex in zip(sigmas_r, means, ses, exact):
        print(f"| {sr:.3f} | {mu:+.4f} | {se:.4f} | {ex:+.4f} | {-vega * (sr - SIGMA_I) * np.exp(R * T):+.4f} |")


def cost_study(t, S, out):
    table = {}
    for kappa in COST_RATES:
        hedger = DeltaHedger(OPTION, R, SIGMA_I, cost_rate=kappa)
        for n in FREQUENCIES:
            k = FINE_STEPS // n
            res = hedger.run(t[::k], S[:, ::k])
            s = res.summary()
            table[kappa, n] = (s["mean_cost"], float(res.gross_pnl.std(ddof=1)), s["std"], s["mean"])

    freq = np.array(FREQUENCIES)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    kappa_demo = 0.001
    cost = np.array([table[kappa_demo, n][0] for n in freq])
    risk = np.array([table[kappa_demo, n][1] for n in freq])
    ax1.semilogx(freq, risk, "o-", color=SERIES[0], label="hedging risk: std of gross P&L")
    ax1.semilogx(freq, cost, "o-", color=SERIES[1], label="expected transaction cost")
    end_label(ax1, freq[-1], risk[-1], "risk")
    end_label(ax1, freq[-1], cost[-1], "cost")
    ax1.set(title="The trade-off at 10 bp proportional cost",
            xlabel="Rebalances per year", ylabel="$ per option")
    ax1.legend(loc="upper center")

    print(f"\n## 3. Transaction costs: objective = mean cost + {RISK_AVERSION} x std(net P&L)")
    print("| cost (bp) | n* | mean cost at n* | std at n* | objective at n* |\n|---|---|---|---|---|")
    for color, kappa in zip(SERIES, COST_RATES):
        objective = np.array([-table[kappa, n][3] + RISK_AVERSION * table[kappa, n][2] for n in freq])
        best = int(np.argmin(objective))
        label = f"{kappa * 1e4:.0f} bp"
        ax2.semilogx(freq, objective, "o-", color=color, label=label)
        ax2.plot(freq[best], objective[best], "o", ms=12, mfc="none", mec=INK, mew=1.5)
        end_label(ax2, freq[-1], objective[-1], label)
        c, _, s, _ = table[kappa, freq[best]]
        print(f"| {kappa * 1e4:.0f} | {freq[best]} | {c:.4f} | {s:.4f} | {objective[best]:.4f} |")
    ax2.set(title=f"\u2212E[P&L] + {RISK_AVERSION:g} \u00d7 std(P&L)   (ring = optimum)",
            xlabel="Rebalances per year", ylabel="$ per option")
    ax2.legend(title="cost rate", loc="upper center", ncol=4, fontsize=8.5, title_fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "transaction_costs.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()

    rng = np.random.default_rng(args.seed)
    z = rng.standard_normal((args.paths, FINE_STEPS))
    t, S = BlackScholesModel(S0, R, SIGMA_I).paths_from_normals(z, T)   # drift = r

    frequency_study(t, S, args.out)
    vol_mismatch_study(z, args.out)
    cost_study(t, S, args.out)
    print(f"\nFigures written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
