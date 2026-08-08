"""Regenerate the README figures from the stored MCMC chain.

Usage
-----
    uv run python scripts/make_figures.py

Requires ``results/chain_m3.h5`` (produced by ``notebooks/01_fit.ipynb``).
Writes PNGs to ``figures/`` and prints the summary table used in the README.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bsrl_fit.diagnostics import (  # noqa: E402
    arviz_summary,
    corner_plot,
    credible_limits,
    load_chain,
)
from bsrl_fit.ppc import (  # noqa: E402
    draw_replications,
    plot_ppc_envelope,
    ppp_value,
    window_stat,
)
from bsrl_fit.simulate_data import expected_counts, simulate_profile  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
CHAIN_PATH = ROOT / "results" / "chain_m3.h5"

# --- ground truth used to generate the synthetic dataset ---------------
T_GRID = np.arange(0, 50, 0.05)
THETA_TRUE = [1e5, 0.3, 1e-2, 1e-3, 5.0, 10.0]
NAMES = ["N_main", "sigma", "f_sat", "f_ghost", "t0", "background"]
SEED = 42


def make_data():
    rng = np.random.default_rng(SEED)
    lam = expected_counts(THETA_TRUE, T_GRID)
    counts = simulate_profile(THETA_TRUE, T_GRID, rng)
    return lam, counts


def fig_data_overview(lam, counts):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax in (ax1, ax2):
        ax.plot(T_GRID, lam, lw=1.4, label="Expected counts/bin")
        ax.plot(T_GRID, counts, drawstyle="steps-mid", alpha=0.6, label="Poisson realisation")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Counts / bin")
        ax.legend()
    ax1.set_title("Linear scale — only the main bunch is visible")
    ax2.set_yscale("log")
    ax2.set_ylim(0.1, None)
    ax2.set_title("Log scale — satellite (7.5 ns) and ghost (30 ns) appear")
    fig.tight_layout()
    fig.savefig(FIGDIR / "data_overview.png", dpi=140)
    plt.close(fig)


def fig_corner(flat_samples):
    fig = corner_plot(
        flat_samples,
        NAMES,
        truth=THETA_TRUE,
        log_params=("f_sat", "f_ghost"),
    )
    fig.savefig(FIGDIR / "corner.png", dpi=110)
    plt.close(fig)


def fig_ppc(counts, flat_samples):
    rng = np.random.default_rng(SEED)
    y_rep = draw_replications(flat_samples, T_GRID, n_reps=200, rng=rng)

    med = np.median(y_rep, axis=0)
    resid = (counts - med) / np.sqrt(np.maximum(med, 1))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    plot_ppc_envelope(T_GRID, counts, y_rep, ax=ax1)
    ax1.set_yscale("log")
    ax1.set_ylim(0.5, None)
    ax1.set_title("Posterior predictive envelope (200 replications)")

    ax2.plot(T_GRID, resid, lw=0.6, color="k")
    for y in (-2, 0, 2):
        ax2.axhline(y, color="gray", lw=0.6, ls="--" if y else "-")
    ax2.set_xlabel("Time [ns]")
    ax2.set_ylabel("std. resid.")
    fig.tight_layout()
    fig.savefig(FIGDIR / "ppc.png", dpi=140)
    plt.close(fig)

    t0 = THETA_TRUE[4]
    stats = {
        "T1 main-peak max": (t0 - 3, t0 + 3, np.max),
        "T2 ghost-window sum": (t0 + 22, t0 + 28, np.sum),
    }
    out = {}
    for label, (lo, hi, red) in stats.items():
        T_obs = window_stat(counts, T_GRID, lo, hi, red)
        T_rep = np.array([window_stat(y, T_GRID, lo, hi, red) for y in y_rep])
        out[label] = ppp_value(T_obs, T_rep)
    return out


def main():
    FIGDIR.mkdir(exist_ok=True)
    lam, counts = make_data()
    samples, _ = load_chain(CHAIN_PATH)
    flat_samples = samples.reshape(-1, samples.shape[-1])

    fig_data_overview(lam, counts)
    fig_corner(flat_samples)
    ppp = fig_ppc(counts, flat_samples)

    print("chain shape (nsteps, nwalkers, ndim):", samples.shape)
    print(arviz_summary(samples, NAMES))
    for p in ("f_sat", "f_ghost"):
        print(p, credible_limits(flat_samples, NAMES, p))
    for k, v in ppp.items():
        print(f"{k}: ppp = {v:.2f}")


if __name__ == "__main__":
    main()
