"""Tests for posterior predictive check utilities."""

import numpy as np
import pytest

from bsrl_fit.ppc import draw_replications, ppp_value, window_stat


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def fake_posterior():
    """Tight posterior around a plausible θ = (N, sigma, f_sat, f_ghost, t0, bg)."""
    n_draws = 500
    samples = np.empty((n_draws, 6))
    samples[:, 0] = 1e5  # N_main
    samples[:, 1] = 1.0  # sigma
    samples[:, 2] = 1e-2  # f_sat
    samples[:, 3] = 1e-3  # f_ghost
    samples[:, 4] = 5.0  # t0
    samples[:, 5] = 1.0  # background
    return samples


def test_draw_replications_shape_and_dtype(fake_posterior, rng):
    t_grid = np.linspace(0, 50, 500)
    n_reps = 20
    y_rep = draw_replications(fake_posterior, t_grid, n_reps, rng)
    assert y_rep.shape == (n_reps, len(t_grid))
    assert np.issubdtype(y_rep.dtype, np.integer)
    assert (y_rep >= 0).all()  # Poisson counts non-negative


def test_window_stat_sum_on_ones():
    counts = np.ones(100, dtype=int)
    t_grid = np.linspace(0, 10, 100)
    # window [2, 5] on a 100-point grid spanning [0, 10] → ~30 bins
    s = window_stat(counts, t_grid, 2.0, 5.0, np.sum)
    # compute expected number of bins in [2, 5]
    mask = (t_grid >= 2) & (t_grid <= 5)
    expected = mask.sum()
    assert s == expected


def test_window_stat_max():
    counts = np.array([0, 0, 5, 3, 0, 0])
    t_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert window_stat(counts, t_grid, 1.5, 3.5, np.max) == 5


def test_ppp_value_known():
    T_rep = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # proportion=(T_rep >= 3) = 3/5 = 0.6
    assert ppp_value(3.0, T_rep) == pytest.approx(0.4)


def test_ppp_value_extremes():
    T_rep = np.arange(100.0)  # values 0..99
    # T_obs above all → one-sided p = 0, two-sided min(0, 1) = 0
    assert ppp_value(200.0, T_rep) == pytest.approx(0.0)
    # T_obs below all → one-sided p = 1, two-sided min(1, 0) = 0
    assert ppp_value(-1.0, T_rep) == pytest.approx(0.0)
