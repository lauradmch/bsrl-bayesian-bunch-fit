"""
Posterior predictive checks (PPC) for the Bayesian regression model.
-> check the fit of the model to the observed data by comparing the observed data
to replicated data generated from the posterior predictive distribution.

"""

import matplotlib.pyplot as plt
import numpy as np

from .simulate_data import simulate_profile


def draw_replications(flat_samples, t_grid, n_reps, rng):
    """
    Draw random paramerter samples from the posterior predictive distribution,
    and simulate replicated datasets from the model using these samples.

    Parameters
    ----------
    flat_samples : array-like
        Posterior samples of the model parameters, flattened to a 2D array.
    t_grid : array-like
        Time points at which to simulate the profile.
    n_reps : int
        Number of replicated datasets to draw.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    y_rep : array-like
        Replicated data from the posterior predictive distribution.
    """
    # preallocate array for replicated data
    y_rep = np.empty((n_reps, len(t_grid)), dtype=int)
    # draw random indices from the posterior samples
    indices = rng.integers(low=0, high=flat_samples.shape[0], size=n_reps)
    # simulate replicated data for each draw
    for i, theta in enumerate(flat_samples[indices]):
        y_rep[i] = simulate_profile(theta, t_grid, rng)
    return y_rep


def window_stat(counts, t_grid, t_lo, t_hi, reducer):
    """
    Compute a scalar test statistic of the counts within a specified time window.
    -> Compare summary facts (mean or max values) of the simulated data to the real data.

    Parameters
    ----------
    counts : array-like
        Counts of events at each time point in t_grid.
    t_grid : array-like
        Time points corresponding to the counts.
    t_lo : float
        Lower bound of the time window.
    t_hi : float
        Upper bound of the time window.
    reducer : callable
        Function to reduce the counts within the window (e.g., np.sum, np.mean).

    Returns
    -------
    stat : float
        Scalar test statistic computed from the counts within the specified time window.
    """
    # create a boolean mask for the time window
    mask = (t_lo <= t_grid) & (t_grid <= t_hi)
    # apply the reducer function to the counts within the window (n.max or np.sum)
    return float(reducer(counts[mask]))


def ppp_value(T_obs, T_rep):
    """
    Compute the posterior predictive p-value = probability that
    a test statistic computed from simulated replicated data is more
    extreme than the same statistic computed from actual observed data.

    Parameters
    ----------
    T_obs : float
        Test statistic computed from the observed data.
    T_rep : array-like
        Test statistics computed from the replicated data.

    Returns
    -------
    ppp : float
        Posterior predictive p-value.
    """
    proportion = np.mean(T_rep >= T_obs)
    # return the minimum of the proportion and its complement for a two-sided p-value
    return min(proportion, 1 - proportion)


def plot_ppc_envelope(t_grid, counts, y_rep, ax=None, levels=(0.16, 0.5, 0.84)):
    """
    Plot the posterior predictive check (PPC) envelope.
    -> max or middle percentiles of replicated data generated
    against the real observed data.
    If real data falls outside the envelope, the model may be wrong.

    Parameters
    ----------
    t_grid : array-like
        Time points corresponding to the counts.
    counts : array-like
        Observed counts of events at each time point in t_grid.
    y_rep : array-like
        Replicated data from the posterior predictive distribution.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object with the PPC envelope plot.
    """

    per_bin = np.quantile(y_rep, q=levels, axis=0)
    low, median, high = per_bin

    fig, ax = plt.subplots() if ax is None else (None, ax)
    ax.fill_between(t_grid, low, high, alpha=0.3, label="68% PPC")
    ax.plot(t_grid, median, label="median")
    ax.step(t_grid, counts, where="mid", label="observed")
    ax.set_xlabel("Time")
    ax.set_ylabel("Counts")
    ax.legend()
    return ax
