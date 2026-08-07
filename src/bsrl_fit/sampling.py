import emcee
import numpy as np

from bsrl_fit.model import log_posterior


def run_mcmc(
    theta0,
    t_grid,
    counts,
    nwalkers=32,
    nsteps=5000,
    seed=0,
    backend=None,
    moves=None,
    burn_restart=False,
    burn_steps=1000,
):
    """
    Run MCMC to sample from the posterior distribution of the parameters.

    Parameters
    ----------
    theta0 : tuple of 6 floats
        Initial guess for the parameters.
    t_grid : np.ndarray, shape (n_bins,)
        Uniformly spaced bin-centre times [ns].
    counts : np.ndarray, shape (n_bins,)
        Observed counts per bin [counts].
    nwalkers : int, optional
        Number of walkers. Default is 32.
    nsteps : int, optional
        Number of steps. Default is 5000.
    seed : int, optional
        Random seed for reproducibility. Default is 0.
    backend : emcee.backends.Backend, optional
        Backend to store the chain. Default is None.
    moves : list of emcee.moves.Move or (Move, weight) tuples, optional
        Move set passed to the sampler. Default None uses emcee's stretch move.
    burn_restart : bool, optional
        If True, run a short exploratory chain, recentre theta0 on the
        highest-log-posterior walker, then start the production run with a
        tighter walker ball. Robust against initial excursions to prior tails.
        Default False.
    burn_steps : int, optional
        Number of steps for the burn-in phase when burn_restart is True.
        Default 1000.

    Returns
    -------
    sampler : emcee.EnsembleSampler
        The MCMC sampler object after running.
    """
    rng = np.random.default_rng(seed)
    theta0 = np.asarray(theta0, dtype=float)
    assert (theta0 > 0).all(), "Components of initial guess must be positive."
    ndim = len(theta0)

    # --- optional burn-and-restart phase --------------------------------
    if burn_restart:
        # 1) build a throwaway in-memory sampler (no backend, no HDF5)
        p0_burn = theta0 + 1e-4 * theta0 * rng.standard_normal(size=(nwalkers, ndim))
        burn_sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior, args=(t_grid, counts), backend=None, moves=moves
        )
        burn_sampler.run_mcmc(p0_burn, burn_steps, progress=True)

        # 2) find the best walker across all (step, walker) pairs
        lp = burn_sampler.get_log_prob()
        best_step, best_walker = np.unravel_index(np.argmax(lp), lp.shape)
        theta0 = burn_sampler.get_chain()[best_step, best_walker]

        # 3) shrink the ball for the production run
        ball_scale = 1e-5
    else:
        ball_scale = 1e-4

    # --- production run --------------------------------------------------
    p0 = theta0 + ball_scale * theta0 * rng.standard_normal(size=(nwalkers, ndim))

    if backend is not None:
        backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_posterior,
        args=(t_grid, counts),
        backend=backend,
        moves=moves,
    )
    sampler.run_mcmc(p0, nsteps, progress=True)
    return sampler
