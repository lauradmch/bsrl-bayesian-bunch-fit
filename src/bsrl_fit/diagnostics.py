"""MCMC diagnostics and plotting functions."""

import matplotlib.pyplot as plt
import numpy as np


def plot_traces(samples, param_names, log_params=()):
    """
    samples: array (nsteps, nwalkers, ndim)
    param_names: list of str, length ndim
    log_params: tuple of str, names to plot on log y-axis
    Returns: fig
    """

    nsteps, nwalkers, ndim = samples.shape
    fig, axes = plt.subplots(ndim, 1, figsize=(8, 2 * ndim), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], alpha=0.3, lw=0.5)
        ax.set_ylabel(param_names[i])
        if param_names[i] in log_params:
            ax.set_yscale("log")
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    return fig


def load_chain(path):
    """Load emcee chain from HDF backend, return (samples, log_prob)."""
    import emcee

    backend = emcee.backends.HDFBackend(path, read_only=True)
    return backend.get_chain(), backend.get_log_prob()


def corner_plot(flat_samples, param_names, truth=None, log_params=()):
    """
    flat_samples: (nsamples, ndim)
    param_names: list of str
    truth: array-like of true values (in *original* space), or None
    log_params: names to transform with log10 before plotting
    Returns: fig
    """
    import corner

    samples = flat_samples.copy()
    labels = list(param_names)
    truths = None if truth is None else list(truth)
    for name in log_params:
        i = param_names.index(name)
        samples[:, i] = np.log10(samples[:, i])
        labels[i] = f"log10({name})"
        if truths is not None:
            truths[i] = np.log10(truths[i])
    fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        truth_color="red",
        show_titles=True,
        title_fmt=".3g",
    )
    return fig


def arviz_summary(samples, param_names):
    """
    samples: array (nsteps, nwalkers, ndim)
    Returns: pandas DataFrame with mean, sd, hdi, ess_bulk, ess_tail, r_hat.
    """
    import arviz as az

    # arviz wants (chain, draw, *param); walkers are our chains
    posterior = {name: samples[:, :, i].T for i, name in enumerate(param_names)}
    return az.summary(posterior, round_to=4)


def credible_limits(flat_samples, param_names, param, levels=(0.16, 0.5, 0.84, 0.95)):
    """
    Return dict of {level: value} for the marginal posterior of `param`.
    Defaults give 68% CI (0.16, 0.84), median (0.5), and 95% upper limit (0.95).
    """
    i = param_names.index(param)
    q = np.quantile(flat_samples[:, i], levels)
    return dict(zip(levels, q, strict=True))
