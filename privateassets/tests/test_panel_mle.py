"""Tests for the panel MLE of the common AR(1) smoothing coefficient."""

# packages
import numpy as np
import pytest
# qis / project
import qis
from privateassets.matf import (
    BiasCorrection,
    bootstrap_corrected_theta,
    fisher_info_panel_ar1,
    fit_panel_ar1,
    kendall_corrected_theta,
    panel_ar1_neg_log_likelihood,
    simulate_panel_ar1,
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


# --- bias correction ---------------------------------------------------------

def test_no_correction_is_the_default_and_changes_nothing():
    """A correction is never applied without being asked for."""
    panel = make_ar1_panel(n_series=8, n_obs=60, theta=0.3)
    result = fit_panel_ar1(panel)
    assert result['bias_correction'] == 'none'
    assert result['theta_hat'] == result['theta_raw']
    assert np.isnan(result['bias'])


def test_the_raw_estimate_is_always_reported():
    """A corrected result still carries what it was corrected from."""
    panel = make_ar1_panel(n_series=8, n_obs=60, theta=0.3)
    raw = fit_panel_ar1(panel)['theta_hat']
    for correction in (BiasCorrection.KENDALL, BiasCorrection.BOOTSTRAP):
        result = fit_panel_ar1(panel, bias_correction=correction, num_bias_draws=20)
        assert result['theta_raw'] == pytest.approx(raw)
        assert result['theta_hat'] == pytest.approx(raw - result['bias'])


def test_kendall_correction_inverts_its_own_bias_formula():
    """Applying the bias to the corrected value returns the raw estimate."""
    n = 50.0
    for theta_hat in (-0.2, 0.0, 0.25, 0.6):
        corrected = kendall_corrected_theta(theta_hat, n)
        assert corrected - (1.0 + 3.0 * corrected) / n == pytest.approx(theta_hat)


def test_kendall_correction_is_upward_and_shrinks_with_length():
    """The bias it removes is smaller on a longer panel."""
    short = kendall_corrected_theta(0.3, 40.0)
    long = kendall_corrected_theta(0.3, 400.0)
    assert short > long > 0.3


def test_kendall_correction_rejects_a_panel_too_short_to_define_it():
    with pytest.raises(ValueError, match='must exceed 3'):
        kendall_corrected_theta(0.3, 3.0)


def test_bootstrap_correction_reduces_the_bias():
    """Measured, not assumed: the corrected estimate lands nearer the truth.

    Averaged over replications, because a single draw is noisier than the bias
    being corrected.
    """
    true_theta, n_obs = 0.30, 40
    raw_errors, corrected_errors = [], []
    for replication in range(12):
        panel = make_ar1_panel(n_series=12, n_obs=n_obs, theta=true_theta,
                               seed=2000 + replication)
        raw_errors.append(fit_panel_ar1(panel)['theta_hat'] - true_theta)
        corrected_errors.append(
            fit_panel_ar1(panel, bias_correction=BiasCorrection.BOOTSTRAP,
                          num_bias_draws=30, seed=5)['theta_hat'] - true_theta)
    assert np.mean(raw_errors) < -0.02  # the raw estimate understates
    assert abs(np.mean(corrected_errors)) < abs(np.mean(raw_errors))


def test_the_measured_bias_is_negative_where_the_estimator_understates():
    panel = make_ar1_panel(n_series=12, n_obs=40, theta=0.35)
    result = fit_panel_ar1(panel, bias_correction=BiasCorrection.BOOTSTRAP,
                           num_bias_draws=30, seed=5)
    assert result['bias'] < 0.0
    assert result['theta_hat'] > result['theta_raw']


def test_the_bootstrap_correction_is_reproducible():
    panel = make_ar1_panel(n_series=8, n_obs=50, theta=0.3)
    kwargs = dict(bias_correction=BiasCorrection.BOOTSTRAP, num_bias_draws=25, seed=42)
    assert (fit_panel_ar1(panel, **kwargs)['theta_hat']
            == fit_panel_ar1(panel, **kwargs)['theta_hat'])


def test_a_different_seed_gives_a_different_correction():
    panel = make_ar1_panel(n_series=8, n_obs=50, theta=0.3)
    a = fit_panel_ar1(panel, bias_correction=BiasCorrection.BOOTSTRAP,
                      num_bias_draws=25, seed=1)['theta_hat']
    b = fit_panel_ar1(panel, bias_correction=BiasCorrection.BOOTSTRAP,
                      num_bias_draws=25, seed=2)['theta_hat']
    assert a != b


def test_correction_leaves_a_long_panel_almost_alone():
    """There is little bias to remove when the series are long."""
    panel = make_ar1_panel(n_series=10, n_obs=400, theta=0.3)
    result = fit_panel_ar1(panel, bias_correction=BiasCorrection.BOOTSTRAP,
                           num_bias_draws=25, seed=3)
    assert abs(result['bias']) < 0.02


def test_simulated_panels_match_the_shape_they_were_asked_for():
    rng = np.random.default_rng(0)
    panel = simulate_panel_ar1(0.3, sigmas=[0.02, 0.05], lengths=[30, 45], rng=rng)
    assert [len(values) for _, values in panel] == [30, 45]
    assert all(abs(values.mean()) < 1e-12 for _, values in panel)  # demeaned


def test_simulated_series_start_stationary():
    """A zero start would leave a transient that reads as bias and gets removed.

    The first quarter of a simulated series should be no less variable than the
    last, which a zero-started recursion would violate.
    """
    rng = np.random.default_rng(1)
    panel = simulate_panel_ar1(0.6, sigmas=[0.03] * 40, lengths=[80] * 40, rng=rng)
    values = np.vstack([v for _, v in panel])
    assert values[:, :20].std() == pytest.approx(values[:, -20:].std(), rel=0.15)


def test_bootstrap_correction_rejects_a_non_positive_draw_count():
    with pytest.raises(ValueError, match='num_draws must be positive'):
        bootstrap_corrected_theta(0.3, [0.02], [40], num_draws=0)


def test_a_correction_on_an_unusable_panel_is_refused():
    with pytest.raises(ValueError, match='long enough'):
        fit_panel_ar1([('a', np.array([0.01, -0.01]))],
                      bias_correction=BiasCorrection.KENDALL)
