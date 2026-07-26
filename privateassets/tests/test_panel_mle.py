"""Tests for the panel MLE of the common AR(1) smoothing coefficient."""

# packages
import numpy as np
import pytest
# qis / project
import qis
from privateassets.matf import (
    fisher_info_panel_ar1,
    fit_panel_ar1,
    panel_ar1_neg_log_likelihood,
)
from privateassets.tests.synthetic_data import make_ar1_panel


def test_panel_mle_is_consistent_on_a_long_panel():
    """With 400 observations per series the estimate lands on the true theta."""
    panel = make_ar1_panel(n_series=20, n_obs=400, theta=0.25)
    result = fit_panel_ar1(panel)
    assert result['theta_hat'] == pytest.approx(0.25, abs=0.01)
    assert result['se'] > 0.0
    assert result['n_series'] == 20


def test_short_panels_understate_theta_by_the_kendall_bias():
    """At realistic fund lengths the demeaned estimate is biased down, materially.

    Demeaning an AR(1) of length n biases the coefficient by about
    ``-(1 + 3 theta) / n``. At 80 quarterly observations and theta = 0.35 that is
    roughly 2.5 percentage points, and the bias flows straight into the
    volatility inflation ``1 / (1 - theta)``, understating unsmoothed risk.

    The test pins the direction and the magnitude so a future bias correction
    shows up here as a deliberate change rather than a silent one.
    """
    theta = 0.35
    n_obs = 80
    result = fit_panel_ar1(make_ar1_panel(n_series=20, n_obs=n_obs, theta=theta))
    kendall = theta - (1.0 + 3.0 * theta) / n_obs
    assert result['theta_hat'] < theta
    assert result['theta_hat'] == pytest.approx(kendall, abs=0.02)


def test_demeaning_is_what_causes_the_short_panel_bias():
    """The same panel without demeaning is close to unbiased at the same length."""
    theta, n_obs = 0.35, 80
    demeaned = fit_panel_ar1(make_ar1_panel(n_series=20, n_obs=n_obs, theta=theta))
    raw = fit_panel_ar1(make_ar1_panel(n_series=20, n_obs=n_obs, theta=theta, demean=False))
    assert raw['theta_hat'] == pytest.approx(theta, abs=0.02)
    assert raw['theta_hat'] > demeaned['theta_hat']


def test_pooling_beats_a_single_series():
    """The panel estimate sits closer to the truth than the median single-series fit.

    This is the whole reason the estimator exists: one fund contributes too few
    quarters to pin theta down.
    """
    true_theta = 0.30
    panel = make_ar1_panel(n_series=10, n_obs=60, theta=true_theta)
    result = fit_panel_ar1(panel)
    singles = [v for v in result['per_series_theta'].values() if v is not None]
    panel_error = abs(result['theta_hat'] - true_theta)
    median_single_error = float(np.median([abs(s - true_theta) for s in singles]))
    assert panel_error < median_single_error


def test_standard_error_shrinks_with_more_series():
    few = fit_panel_ar1(make_ar1_panel(n_series=3, n_obs=40, theta=0.2))
    many = fit_panel_ar1(make_ar1_panel(n_series=20, n_obs=40, theta=0.2))
    assert many['se'] < few['se']


def test_zero_theta_is_recovered_from_white_noise():
    """White noise carries no smoothing, so the estimate sits at zero."""
    panel = make_ar1_panel(n_series=12, n_obs=200, theta=0.0)
    assert fit_panel_ar1(panel)['theta_hat'] == pytest.approx(0.0, abs=0.03)


def test_likelihood_is_minimised_at_the_estimate():
    """The reported theta beats a grid around it, so the optimiser converged."""
    panel = make_ar1_panel(n_series=6, n_obs=50, theta=0.2)
    result = fit_panel_ar1(panel)
    at_hat = panel_ar1_neg_log_likelihood(result['theta_hat'], panel)
    grid = np.linspace(-0.9, 0.9, 61)
    assert at_hat <= min(panel_ar1_neg_log_likelihood(t, panel) for t in grid) + 1e-8


def test_fisher_information_is_positive():
    panel = make_ar1_panel(n_series=6, n_obs=50, theta=0.2)
    assert fisher_info_panel_ar1(0.2, panel) > 0.0


def test_short_series_are_skipped_not_fitted():
    panel = make_ar1_panel(n_series=4, n_obs=50, theta=0.2)
    panel.append(('too_short', np.array([0.01, -0.01])))
    result = fit_panel_ar1(panel)
    assert result['per_series_theta']['too_short'] is None
    assert result['n_series'] == 4


def test_bare_arrays_are_accepted():
    """A list of arrays is accepted and labelled positionally.

    Accuracy is covered by the consistency and bias tests above; this one pins
    the calling convention only.
    """
    labelled = make_ar1_panel(n_series=5, n_obs=50, theta=0.2)
    bare = [s for _, s in labelled]
    result = fit_panel_ar1(bare)
    assert result['theta_hat'] == pytest.approx(fit_panel_ar1(labelled)['theta_hat'])
    assert list(result['per_series_theta']) == [f'series_{i}' for i in range(5)]


def test_fit_rejects_an_empty_panel():
    with pytest.raises(ValueError, match='empty'):
        fit_panel_ar1([])


def test_fit_rejects_bounds_outside_the_stationary_region():
    panel = make_ar1_panel(n_series=3, n_obs=30, theta=0.2)
    with pytest.raises(ValueError, match='theta_bounds'):
        fit_panel_ar1(panel, theta_bounds=(-1.5, 0.9))


def test_estimated_theta_drives_the_qis_unsmoother():
    """The estimate is consumed by qis, and unsmoothing raises the variance.

    Inverting the moving average removes the smoothing that appraisal reporting
    introduces, so the unsmoothed series must be more volatile than the observed
    one for a positive theta.
    """
    import pandas as pd
    panel = make_ar1_panel(n_series=8, n_obs=80, theta=0.35)
    theta_hat = fit_panel_ar1(panel)['theta_hat']
    observed = pd.Series(panel[0][1], index=pd.date_range('2005-03-31', periods=80, freq='QE'))
    unsmoothed = qis.unsmooth_returns_glm(observed, ar_order=1, theta=theta_hat)
    assert unsmoothed.std() > observed.std()
