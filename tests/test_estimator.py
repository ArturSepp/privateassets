"""
End-to-end tests for the MATF estimator.

The panel these run on is built so the funds earn exactly what the MATF
benchmark portfolio earns, times ``exp(alpha)``, and the reported NAVs are then
smoothed by a known AR(1). Recovering alpha therefore requires the return
reconstruction, the unsmoothing, the loading fit and the deflator all to be
right at once, which is the only test that covers the seams between them.
"""

# packages
import dataclasses
import numpy as np
import pandas as pd
import pytest
# qis / project
from privateassets.matf import MatfResult, SignConstraint, estimate_matf_alpha
from tests.synthetic_data import (FACTOR_NAMES, make_cash_flows,
                                                make_factor_driven_panel, make_factor_levels,
                                                make_rf_quarterly)

pytest.importorskip('factorlasso', reason='the loading fit needs the [factors] extra')

TRUE_BETA = np.array([0.70, 0.00, 0.50, 0.00])
TRUE_ALPHA = 0.04
TRUE_THETA = 0.30

FIXED_BETA = pd.Series(TRUE_BETA, index=FACTOR_NAMES)


def _panel(alpha: float = TRUE_ALPHA, theta: float = TRUE_THETA):
    return make_factor_driven_panel(beta=TRUE_BETA, annual_alpha=alpha, theta=theta)


def test_recovers_an_injected_alpha():
    """The headline number lands on the alpha that was put in."""
    result = estimate_matf_alpha(*_panel())
    assert isinstance(result, MatfResult)
    assert result.cap_weighted_alpha == pytest.approx(TRUE_ALPHA, abs=0.015)
    assert result.n_mature == 4


def test_alpha_responds_one_for_one_to_injected_alpha():
    """Doubling the injected alpha moves the estimate by the same amount.

    Stronger than a level test: it isolates the estimator's response from any
    constant offset caused by the loadings being shrunk.
    """
    low = estimate_matf_alpha(*_panel(alpha=0.04))
    high = estimate_matf_alpha(*_panel(alpha=0.08))
    assert high.cap_weighted_alpha - low.cap_weighted_alpha == pytest.approx(0.04, abs=0.006)


def test_a_panel_with_no_alpha_prices_near_zero():
    result = estimate_matf_alpha(*_panel(alpha=0.0))
    assert abs(result.cap_weighted_alpha) < 0.015


def test_alpha_is_exact_when_the_loadings_are_known():
    """Given the true beta, the residual is the injected alpha and little else."""
    cf, navs, levels, rf = _panel()
    result = estimate_matf_alpha(cf, navs, levels, rf, beta=FIXED_BETA)
    assert result.cap_weighted_alpha == pytest.approx(TRUE_ALPHA, abs=0.008)
    assert result.betas is None
    assert result.provenance['beta_was_supplied'] is True


def test_no_look_ahead_end_to_end():
    """Removing data after a vintage closes leaves that vintage's alpha unchanged.

    The strongest available statement that the deflator is point-in-time: if a
    later observation could reach an earlier valuation, deleting it would move
    the number. Loadings are held fixed because the fit is in-sample by design
    and would otherwise move under the test.
    """
    cf, navs, levels, rf = _panel()
    last_flow = cf.loc[cf['vintage_label'] == 'Fund 1', 'date'].max()
    cutoff = last_flow + pd.Timedelta(days=5)

    full = estimate_matf_alpha(cf, navs, levels, rf, beta=FIXED_BETA)
    truncated = estimate_matf_alpha(cf[cf['date'] <= cutoff], navs[navs['date'] <= cutoff],
                                    levels.loc[:cutoff], rf.loc[:cutoff], beta=FIXED_BETA)

    alpha_full = full.vintage_alpha.set_index('vintage_label').loc['Fund 1', 'matf_alpha']
    alpha_truncated = truncated.vintage_alpha.set_index('vintage_label').loc['Fund 1', 'matf_alpha']
    assert alpha_full == pytest.approx(alpha_truncated, abs=1e-12)


def test_the_covariance_path_is_point_in_time():
    """Every shared covariance matrix is identical under truncation."""
    cf, navs, levels, rf = _panel()
    cutoff = pd.Timestamp('2016-12-31')
    full = estimate_matf_alpha(cf, navs, levels, rf, beta=FIXED_BETA)
    truncated = estimate_matf_alpha(cf[cf['date'] <= cutoff], navs[navs['date'] <= cutoff],
                                    levels.loc[:cutoff], rf.loc[:cutoff], beta=FIXED_BETA)
    shared = set(full.sigma_by_period) & set(truncated.sigma_by_period)
    assert len(shared) > 20
    for date in shared:
        assert np.allclose(full.sigma_by_period[date], truncated.sigma_by_period[date], atol=0)


def test_loadings_land_on_the_factors_that_drive_the_panel():
    """Equity and Credit carry the exposure; the other two do not."""
    result = estimate_matf_alpha(*_panel())
    beta = result.betas.beta
    assert beta['Equity'] > 0.3
    assert beta['Credit'] > 0.2
    assert beta['Equity'] > beta['Rates']
    assert beta['Credit'] > beta['Commodities']


def test_a_smoothed_panel_gives_a_positive_theta():
    """Appraisal smoothing is detected, and more of it reads as more."""
    lightly = estimate_matf_alpha(*_panel(theta=0.05))
    heavily = estimate_matf_alpha(*_panel(theta=0.45))
    assert heavily.theta > lightly.theta
    assert heavily.theta > 0.0


def test_theta_is_attenuated_by_the_j_curve():
    """The estimate sits below the truth, and the direction is documented.

    Modified Dietz returns are noisiest while capital is still being called, when
    the denominator is dominated by the call itself. That measurement noise
    attenuates an autoregressive estimate, on top of the Kendall demeaning bias.
    Understating theta understates the volatility uplift 1/(1-theta).
    """
    result = estimate_matf_alpha(*_panel(theta=TRUE_THETA))
    assert 0.0 < result.theta < TRUE_THETA


def test_a_supplied_theta_is_used_rather_than_estimated():
    cf, navs, levels, rf = _panel()
    result = estimate_matf_alpha(cf, navs, levels, rf, theta=0.25)
    assert result.theta == 0.25
    assert np.isnan(result.theta_se)  # not estimated, so it has no standard error
    assert result.provenance['theta'] == 0.25


def test_unsmoothing_raises_the_dispersion_of_the_pooled_series():
    result = estimate_matf_alpha(*_panel())
    assert result.unsmoothed_returns.std() > result.pooled_returns.std()


def test_every_intermediate_is_returned_not_printed(capsys):
    """A caller can reach each stage's output without re-reading a spreadsheet."""
    result = estimate_matf_alpha(*_panel(), num_bootstrap=12, block_size=8)
    assert not capsys.readouterr().out
    assert len(result.pooled_returns) > 20
    assert len(result.unsmoothed_returns) > 20
    assert result.betas.beta.notna().all()
    assert result.beta_bootstrap is not None
    assert len(result.beta_bootstrap.draws) > 0
    assert set(result.vintage_alpha.columns) >= {'vintage_label', 'contributions', 'DPI',
                                                 'matf_alpha'}
    assert len(result.sigma_by_period) > 20


def test_the_resampled_interval_brackets_the_point_estimate():
    result = estimate_matf_alpha(*_panel(), num_bootstrap=25, block_size=8, seed=3)
    boot = result.beta_bootstrap
    for factor in ('Equity', 'Credit'):
        assert boot.lower[factor] <= boot.point[factor] <= boot.upper[factor]


def test_the_same_seed_reproduces_the_interval():
    cf, navs, levels, rf = _panel()
    kwargs = dict(num_bootstrap=15, block_size=8, seed=11)
    a = estimate_matf_alpha(cf, navs, levels, rf, **kwargs)
    b = estimate_matf_alpha(cf, navs, levels, rf, **kwargs)
    pd.testing.assert_frame_equal(a.beta_bootstrap.draws, b.beta_bootstrap.draws)


def test_bootstrap_is_skipped_by_default():
    assert estimate_matf_alpha(*_panel()).beta_bootstrap is None


def test_provenance_carries_what_a_published_number_needs():
    """Versions and specification, because a resampled result is not reproducible without them."""
    result = estimate_matf_alpha(*_panel(), num_bootstrap=10, block_size=8, seed=7)
    provenance = result.provenance
    assert provenance['qis_version'] != 'not installed'
    assert provenance['factorlasso_version'] != 'not installed'
    assert provenance['seed'] == 7
    assert provenance['freq'] == 'QE'
    assert provenance['covar_span_months'] == 60
    assert provenance['betas_are_in_sample'] is True
    assert provenance['covariance_is_point_in_time'] is True
    assert provenance['no_interpolation'] is True
    assert provenance['reporting_months'] == 3.0


def test_the_maturity_gate_excludes_unrealised_vintages():
    """Raising the DPI threshold removes vintages from the aggregate."""
    cf, navs, levels, rf = _panel()
    loose = estimate_matf_alpha(cf, navs, levels, rf, beta=FIXED_BETA, dpi_threshold=0.5)
    strict = estimate_matf_alpha(cf, navs, levels, rf, beta=FIXED_BETA, dpi_threshold=2.2)
    assert strict.n_mature < loose.n_mature


def test_unpriced_vintages_are_reported_as_nan_not_dropped():
    """A vintage the deflator cannot reach keeps its row, with a NaN alpha."""
    cf, navs, levels, rf = _panel()
    late_levels = levels.loc[pd.Timestamp('2014-01-01'):]
    result = estimate_matf_alpha(cf, navs, late_levels, rf, beta=FIXED_BETA)
    assert len(result.vintage_alpha) == cf['vintage_label'].nunique()
    assert result.vintage_alpha['matf_alpha'].isna().any()


def test_result_is_an_immutable_snapshot():
    result = estimate_matf_alpha(*_panel())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.cap_weighted_alpha = 0.0


def test_a_mixed_frequency_panel_is_rejected():
    """The frequency policy holds end to end, not only in the returns module."""
    cf, navs, levels, rf = _panel()
    thinned = navs[~((navs['vintage_label'] == 'Fund 2')
                     & (navs['date'] == navs['date'].unique()[10]))]
    with pytest.raises(ValueError, match='skip a QE period'):
        estimate_matf_alpha(cf, thinned, levels, rf)


def test_a_factor_panel_that_does_not_overlap_is_rejected():
    cf, navs, levels, rf = _panel()
    with pytest.raises(ValueError, match='overlaps the reporting window'):
        estimate_matf_alpha(cf, navs, levels.loc[:'2001-06-30'], rf)


def test_a_supplied_beta_naming_unknown_factors_is_rejected():
    cf, navs, levels, rf = _panel()
    bad = pd.Series([1.0], index=['NotAFactor'])
    with pytest.raises(ValueError, match='absent from factor_levels'):
        estimate_matf_alpha(cf, navs, levels, rf, beta=bad)


def test_sign_constraints_reach_the_loading_fit():
    cf, navs, levels, rf = _panel()
    result = estimate_matf_alpha(cf, navs, levels, rf,
                                 sign_constraints={'Equity': SignConstraint.NEG},
                                 fit_kwargs={'auto_sign_constraints': False})
    assert result.betas.beta['Equity'] <= 1e-8


def test_runs_on_the_unrelated_synthetic_panel():
    """The J-curve panel is not factor-driven, so this only checks it completes."""
    cf, navs = make_cash_flows()
    levels = make_factor_levels()
    quarter_ends = pd.DatetimeIndex(levels.resample('QE').last().index)
    result = estimate_matf_alpha(cf, navs, levels, make_rf_quarterly(quarter_ends))
    assert len(result.vintage_alpha) == 5
    assert np.isfinite(result.cap_weighted_alpha)


def test_the_bias_correction_closes_part_of_the_theta_gap_and_names_the_rest():
    """Two biases push theta down, and only one of them is correctable.

    Removing the demeaning bias moves the estimate towards the truth but does
    not reach it. What remains is measurement error in the reported returns:
    modified Dietz is noisiest while capital is still being called, and error in
    a regressor attenuates its coefficient. Simulating from the fitted model
    reproduces the fitted persistence, not the true one, so no small-sample
    correction recovers it.

    On this panel the measurement-error component is several times the
    small-sample one, which is the ordering worth knowing before choosing what
    to fix next.
    """
    from privateassets.matf import BiasCorrection
    cf, navs, levels, rf = _panel(theta=TRUE_THETA)
    raw = estimate_matf_alpha(cf, navs, levels, rf, beta=FIXED_BETA)
    corrected = estimate_matf_alpha(cf, navs, levels, rf, beta=FIXED_BETA,
                                    bias_correction=BiasCorrection.BOOTSTRAP)

    assert corrected.theta_raw == pytest.approx(raw.theta)
    assert raw.theta < corrected.theta < TRUE_THETA          # closer, not there
    small_sample = corrected.theta - corrected.theta_raw
    measurement_error = TRUE_THETA - corrected.theta
    assert measurement_error > small_sample                   # and it is the larger one


def test_the_correction_is_recorded_in_provenance():
    from privateassets.matf import BiasCorrection
    result = estimate_matf_alpha(*_panel(), bias_correction=BiasCorrection.KENDALL)
    assert result.provenance['bias_correction'] == 'kendall'
    assert result.theta != result.theta_raw


def test_a_supplied_theta_ignores_the_correction():
    from privateassets.matf import BiasCorrection
    cf, navs, levels, rf = _panel()
    result = estimate_matf_alpha(cf, navs, levels, rf, theta=0.25,
                                 bias_correction=BiasCorrection.BOOTSTRAP)
    assert result.theta == 0.25 and result.theta_raw == 0.25
