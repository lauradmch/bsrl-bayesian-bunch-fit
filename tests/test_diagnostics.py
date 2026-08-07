"""Shape and behaviour tests for M4 diagnostics helpers."""

import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before any pyplot import

import numpy as np
import pytest

from bsrl_fit.diagnostics import (
    arviz_summary,
    corner_plot,
    credible_limits,
    plot_traces,
)

PARAM_NAMES = ["a", "b", "c"]


@pytest.fixture
def fake_samples():
    """(nsteps, nwalkers, ndim) = (200, 8, 3), reproducible."""
    rng = np.random.default_rng(0)
    return rng.normal(loc=[1.0, 2.0, 3.0], scale=0.1, size=(200, 8, 3))


def test_arviz_summary_shape(fake_samples):
    df = arviz_summary(fake_samples, PARAM_NAMES)
    assert list(df.index) == PARAM_NAMES
    for col in ("mean", "sd", "ess_bulk", "ess_tail", "r_hat"):
        assert col in df.columns


def test_credible_limits_keys_and_order(fake_samples):
    flat = fake_samples.reshape(-1, fake_samples.shape[-1])
    lims = credible_limits(flat, PARAM_NAMES, "b")
    assert set(lims.keys()) == {0.16, 0.5, 0.84, 0.95}
    # quantiles must be monotonically increasing
    assert lims[0.16] < lims[0.5] < lims[0.84] < lims[0.95]
    # posterior of "b" centered around 2.0
    assert abs(lims[0.5] - 2.0) < 0.05


def test_plot_traces_runs(fake_samples):
    fig = plot_traces(fake_samples, PARAM_NAMES)
    assert len(fig.axes) == 3


def test_corner_plot_runs(fake_samples):
    flat = fake_samples.reshape(-1, fake_samples.shape[-1])
    fig = corner_plot(flat, PARAM_NAMES, truth=[1.0, 2.0, 3.0])
    assert fig is not None
