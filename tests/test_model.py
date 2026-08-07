import numpy as np

from bsrl_fit import log_likelihood, log_posterior, log_prior, simulate_profile


def test_log_prior_rejects_negative_sigma():
    theta = (1e4, -1.0, 1e-3, 1e-3, 25.0, 0.1)
    assert log_prior(theta) == -np.inf


def test_log_prior_rejects_f_sat_out_of_range():
    theta = (1e4, 1.0, 1e-8, 1e-3, 25.0, 0.1)  # f_sat too small
    assert log_prior(theta) == -np.inf


def test_log_prior_finite_at_valid_theta():
    theta = (1e4, 1.0, 1e-3, 1e-3, 25.0, 0.1)
    assert np.isfinite(log_prior(theta))


def test_log_prior_rejects_zero_background():
    assert log_prior((1e4, 1.0, 1e-3, 1e-3, 25.0, 0.0)) == -np.inf


def test_log_likelihood_finite_on_simulated_data():
    rng = np.random.default_rng(0)
    t_grid = np.linspace(0, 50, 500)
    theta_true = (1e4, 1.0, 1e-3, 1e-3, 25.0, 0.1)
    counts = simulate_profile(theta_true, t_grid, rng)
    assert np.isfinite(log_likelihood(theta_true, t_grid, counts))


def test_log_posterior_peaks_near_truth():
    """
    The log-posterior at the true theta should exceed the log-posterior at a clearly wrong theta.
    """
    rng = np.random.default_rng(0)
    t_grid = np.linspace(0, 50, int(50 / 0.05) + 1)  # 1001 points, uniformly spaced
    theta_true = (1e4, 1.0, 1e-3, 1e-3, 25.0, 0.1)
    theta_wrong = (1e4, 5.0, 1e-3, 1e-3, 40.0, 0.1)  # wrong sigma AND t0
    counts = simulate_profile(theta_true, t_grid, rng)
    assert log_posterior(theta_true, t_grid, counts) > log_posterior(theta_wrong, t_grid, counts)
