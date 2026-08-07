import numpy as np

from bsrl_fit.sampling import run_mcmc
from bsrl_fit.simulate_data import simulate_profile


def test_run_mcmc_shapes():
    theta = (1e5, 0.3, 0.01, 0.001, 5.0, 10.0)
    t_grid = np.linspace(0, 50, 100)
    counts = simulate_profile(theta, t_grid, rng=np.random.default_rng(0))

    sampler = run_mcmc(theta, t_grid, counts, nwalkers=16, nsteps=50, seed=0)

    chain = sampler.get_chain()
    assert chain.shape == (50, 16, 6)
    assert np.isfinite(sampler.get_log_prob()).all()
