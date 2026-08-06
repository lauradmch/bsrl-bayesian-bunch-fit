import numpy as np

SAT_OFFSET_NS = 2.5  # satellite offset within same RF bucket
GHOST_OFFSET_NS = 25.0  # ghost lives in next RF bucket (40 MHz spacing)


def _gaussian(t, mu, sigma):
    """
    Evaluate a Gaussian function at times t.

    Parameters
    ----------
    t : np.ndarray, shape (n_bins,)
        Times to evaluate the Gaussian at.
    mu : float
        Mean of the Gaussian.
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    g : np.ndarray, shape (n_bins,)
        Gaussian evaluated at t.
    """
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def expected_counts(theta, t_grid):
    """
    Compute the expected counts per bin for a 3-Gaussian + background model.

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
    t_grid : np.ndarray, shape (n_bins,)
        Uniformly spaced bin-centre times [ns].

    Returns
    -------
    lam : np.ndarray, shape (n_bins,)
        Expected counts per bin [counts].
    """
    # unpack parameters
    N_main, sigma, f_sat, f_ghost, t0, background = theta
    dt = t_grid[1] - t_grid[0]
    assert np.allclose(np.diff(t_grid), dt), "t_grid must be uniformly spaced"

    # compute noise-free rate (photons/ns)
    rate = (
        N_main * _gaussian(t_grid, t0, sigma)
        + N_main * f_sat * _gaussian(t_grid, t0 + SAT_OFFSET_NS, sigma)
        + N_main * f_ghost * _gaussian(t_grid, t0 + GHOST_OFFSET_NS, sigma)
        + background
    )

    # convert to photons per bin
    # avoid negative or zero rates
    return np.maximum(rate * dt, 1e-30)


def simulate_profile(theta, t_grid, rng):
    """
    Simulate Poisson photon counts from a 3-Gaussian + background bunch model.

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
    t_grid : np.ndarray, shape (n_bins,)
        Uniformly spaced bin-centre times [ns].
    rng : np.random.Generator
        Reproducible RNG (from np.random.default_rng(seed)).

    Returns
    -------
    counts : np.ndarray, shape (n_bins,), dtype int
        Simulated Poisson photon counts per bin [counts].
    """
    # compute expected counts per bin
    lam = expected_counts(theta, t_grid)

    return rng.poisson(lam)
