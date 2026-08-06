import numpy as np

from bsrl_fit import expected_counts, simulate_profile

THETA = (1e5, 0.3, 1e-2, 1e-3, 5.0, 10.0)
T_GRID = np.arange(0.0, 50.0, 0.05)


def test_shape():
    lam = expected_counts(THETA, T_GRID)
    assert lam.shape == T_GRID.shape

    rng = np.random.default_rng(42)
    counts = simulate_profile(THETA, T_GRID, rng)
    assert counts.shape == T_GRID.shape

    # Check the output dtype is integer
    assert np.issubdtype(counts.dtype, np.integer)


def test_determinism():
    rng1 = np.random.default_rng(42)
    counts1 = simulate_profile(THETA, T_GRID, rng1)
    rng2 = np.random.default_rng(42)
    counts2 = simulate_profile(THETA, T_GRID, rng2)
    assert np.array_equal(counts1, counts2)

    rng3 = np.random.default_rng(999)
    counts3 = simulate_profile(THETA, T_GRID, rng3)
    assert not np.array_equal(counts1, counts3)


def test_poisson_mean():
    N = 1000
    rng = np.random.default_rng(42)
    lam = expected_counts(THETA, T_GRID)
    counts_trials = np.array([simulate_profile(THETA, T_GRID, rng) for _ in range(N)])
    mean = counts_trials.mean(axis=0)
    # Poisson: mean estimator has stddev sqrt(mu/N). Allow 5 sigma.
    tol = 5 * np.sqrt(lam / N)
    assert np.all(np.abs(mean - lam) <= tol)
