import numpy as np

from bsrl_fit import expected_counts

S_N = 3e4
S_SIGMA = 2

# assumed bounds on t0 (ns)
T_MIN = 0
T_MAX = 50

def log_prior(theta):
    """
    Compute the log-prior density (up to an additive constant) of the parameters.

    Parameters
    ----------
    theta : tuple of 6 floats
        (N_main, sigma, f_sat, f_ghost, t0, background) where:
        N_main : float
            Expected total photons in the main bunch [counts].
        sigma : float
            RMS width of every Gaussian [ns].
        f_sat, f_ghost : float
            Satellite / ghost populations as a fraction of N_main [dimensionless].
        t0 : float
            Centre time of the main bunch [ns].
        background : float
            Flat dark-count rate [photons / ns].

    Returns
    -------
    log_prior : float
        Log-prior density.
    """
    N_main, sigma, f_sat, f_ghost, t0, background = theta

    # check parameter bounds
    if N_main < 0:      return -np.inf
    if sigma <= 0:      return -np.inf
    if not (1e-6 <= f_sat  <= 1e-1): return -np.inf
    if not (1e-6 <= f_ghost <= 1e-1): return -np.inf
    if not (T_MIN < t0 < T_MAX):     return -np.inf
    if background <= 0:  return -np.inf

    # sum of log-densities
    log_prior_value = (-N_main**2 / (2*S_N**2)      # half-normal on N_main
                   - sigma**2  / (2*S_SIGMA**2)   # half-normal on sigma
                   - np.log(f_sat)              # log-uniform on f_sat
                   - np.log(f_ghost))           # log-uniform on f_ghost
                   # t0 uniform  → +0 (skip)
                   # background  → +0 (skip)
    return log_prior_value

def log_likelihood(theta, t_grid, counts):
    """
    Compute the log-likelihood of the data given the parameters.

    Parameters
    ----------
    theta : tuple of 6 floats
        see log_prior for parameter definitions.
    t_grid : np.ndarray, shape (n_bins,)
        Uniformly spaced bin-centre times [ns].
    counts : np.ndarray, shape (n_bins,)
        Observed counts per bin [counts].

    Returns
    -------
    log_likelihood : float
        Log-likelihood.
    """
    # model-predicted rate in bins
    mu = expected_counts(theta, t_grid)
    # Poisson log-likelihood (ignoring constant term)
    return np.sum(counts * np.log(mu) - mu)

def log_posterior(theta, t_grid, counts):
    """
    Compute the log-posterior density (up to an additive constant) of the parameters.

    Parameters
    ----------
    theta : tuple of 6 floats
        see log_prior for parameter definitions.
    t_grid : np.ndarray, shape (n_bins,)
        Uniformly spaced bin-centre times [ns].
    counts : np.ndarray, shape (n_bins,)
        Observed counts per bin [counts].

    Returns
    -------
    log_posterior : float
        Log-posterior density.
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t_grid, counts)