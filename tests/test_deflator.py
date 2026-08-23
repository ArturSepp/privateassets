"""Tests for the MATF multi-factor deflator and its point-in-time covariance."""

# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
from privateassets.matf import (
    closest_or_default_sigma,
    factor_log_levels_panel,
    factor_monthly_log_returns,
    matf_deflator,
    rolling_factor_covar,
)
from tests.synthetic_data import make_factor_levels, make_rf_quarterly


def _panel():
    """Cumulative log factor levels and cumulative log rf on a quarter-end grid."""
    levels = make_factor_levels()
    quarter_ends = pd.DatetimeIndex(levels.resample('QE').last().index)
    quarterly = np.log(levels.resample('QE').last()).diff().fillna(0.0)
    cum_log_factor = factor_log_levels_panel(quarterly).values
    rf_q = make_rf_quarterly(quarter_ends)
    cum_log_rf = np.log1p(rf_q).cumsum().values
    return levels, quarter_ends, cum_log_factor, cum_log_rf


def test_zero_beta_deflator_is_the_risk_free_accrual():
    """With no factor exposure the benchmark portfolio is cash.

    This is the only case with a closed form independent of the covariance, so
    it pins the rf leg without relying on the factor leg being right.
    """
    levels, quarter_ends, cum_log_factor, cum_log_rf = _panel()
    n_factors = cum_log_factor.shape[1]
    t0 = quarter_ends[20]
    cf_dates = list(quarter_ends[[20, 30, 44]])
    out = matf_deflator(cf_dates=cf_dates, t0=t0,
                        cum_log_factor=cum_log_factor, cum_log_rf=cum_log_rf,
                        quarter_ends=quarter_ends, beta=np.zeros(n_factors),
                        sigma_default=np.zeros((n_factors, n_factors)))
    idx0 = 20
    expected = [float(np.exp(cum_log_rf[i] - cum_log_rf[idx0])) for i in (20, 30, 44)]
    assert out == pytest.approx(expected)


def test_deflator_at_the_origin_is_one_without_rf():
    """A zero horizon and a zero rate leave the deflator at unity."""
    levels, quarter_ends, cum_log_factor, _ = _panel()
    n_factors = cum_log_factor.shape[1]
    zero_rf = np.zeros(len(quarter_ends))
    t0 = quarter_ends[30]
    out = matf_deflator(cf_dates=[t0], t0=t0,
                        cum_log_factor=cum_log_factor, cum_log_rf=zero_rf,
                        quarter_ends=quarter_ends, beta=np.zeros(n_factors),
                        sigma_default=np.zeros((n_factors, n_factors)))
    assert out[0] == pytest.approx(1.0)


def test_unit_beta_on_one_factor_reproduces_that_factor_plus_rf():
    """Beta on a single factor and zero covariance recovers rf plus the factor return."""
    levels, quarter_ends, cum_log_factor, cum_log_rf = _panel()
    n_factors = cum_log_factor.shape[1]
    beta = np.zeros(n_factors)
    beta[0] = 1.0
    t0, i0, i1 = quarter_ends[12], 12, 40
    out = matf_deflator(cf_dates=[quarter_ends[i1]], t0=t0,
                        cum_log_factor=cum_log_factor, cum_log_rf=cum_log_rf,
                        quarter_ends=quarter_ends, beta=beta,
                        sigma_default=np.zeros((n_factors, n_factors)))
    expected = np.exp((cum_log_rf[i1] - cum_log_rf[i0])
                      + (cum_log_factor[i1, 0] - cum_log_factor[i0, 0]))
    assert out[0] == pytest.approx(expected)


def test_dates_before_the_panel_are_nan_not_pinned_to_the_first_quarter():
    """A pre-panel date has no defined horizon, so it must not return a number.

    Clipping the index to zero would silently pin the date to the first quarter
    end and hand back a finite deflator for a horizon that does not exist.
    """
    levels, quarter_ends, cum_log_factor, cum_log_rf = _panel()
    n_factors = cum_log_factor.shape[1]
    early = quarter_ends[0] - pd.Timedelta(days=400)
    out = matf_deflator(cf_dates=[early, quarter_ends[10]], t0=quarter_ends[5],
                        cum_log_factor=cum_log_factor, cum_log_rf=cum_log_rf,
                        quarter_ends=quarter_ends, beta=np.zeros(n_factors),
                        sigma_default=np.zeros((n_factors, n_factors)))
    assert np.isnan(out[0])
    assert np.isfinite(out[1])


def test_origin_before_the_panel_makes_every_date_nan():
    levels, quarter_ends, cum_log_factor, cum_log_rf = _panel()
    n_factors = cum_log_factor.shape[1]
    out = matf_deflator(cf_dates=list(quarter_ends[5:8]),
                        t0=quarter_ends[0] - pd.Timedelta(days=400),
                        cum_log_factor=cum_log_factor, cum_log_rf=cum_log_rf,
                        quarter_ends=quarter_ends, beta=np.zeros(n_factors),
                        sigma_default=np.zeros((n_factors, n_factors)))
    assert np.isnan(out).all()


def test_jensen_terms_cancel_when_beta_sums_a_single_unit_exposure():
    """For a single unit exposure diag(Sigma) equals beta' Sigma beta, so the
    convexity correction nets to zero regardless of the variance level."""
    levels, quarter_ends, cum_log_factor, cum_log_rf = _panel()
    n_factors = cum_log_factor.shape[1]
    beta = np.zeros(n_factors)
    beta[1] = 1.0
    sigma = np.diag([0.01, 0.04, 0.02, 0.03])
    with_variance = matf_deflator(cf_dates=[quarter_ends[50]], t0=quarter_ends[10],
                                  cum_log_factor=cum_log_factor, cum_log_rf=cum_log_rf,
                                  quarter_ends=quarter_ends, beta=beta, sigma_default=sigma)
    without = matf_deflator(cf_dates=[quarter_ends[50]], t0=quarter_ends[10],
                            cum_log_factor=cum_log_factor, cum_log_rf=cum_log_rf,
                            quarter_ends=quarter_ends, beta=beta,
                            sigma_default=np.zeros((n_factors, n_factors)))
    assert with_variance[0] == pytest.approx(without[0])


def test_deflator_rejects_a_mismatched_beta():
    levels, quarter_ends, cum_log_factor, cum_log_rf = _panel()
    with pytest.raises(ValueError, match='beta must have shape'):
        matf_deflator(cf_dates=[quarter_ends[10]], t0=quarter_ends[5],
                      cum_log_factor=cum_log_factor, cum_log_rf=cum_log_rf,
                      quarter_ends=quarter_ends, beta=np.zeros(2),
                      sigma_default=np.zeros((4, 4)))


def test_deflator_requires_some_covariance():
    levels, quarter_ends, cum_log_factor, cum_log_rf = _panel()
    with pytest.raises(ValueError, match='sigma_by_quarter'):
        matf_deflator(cf_dates=[quarter_ends[10]], t0=quarter_ends[5],
                      cum_log_factor=cum_log_factor, cum_log_rf=cum_log_rf,
                      quarter_ends=quarter_ends, beta=np.zeros(4))


def test_rolling_covariance_is_point_in_time():
    """Truncating the panel after a date leaves that date's matrix unchanged.

    A covariance that changed when later data was removed would be looking ahead.
    """
    levels = make_factor_levels()
    full = rolling_factor_covar(levels)
    cutoff = sorted(full)[40]
    truncated = rolling_factor_covar(levels.loc[:cutoff])
    assert np.allclose(full[cutoff], truncated[cutoff], rtol=1e-8)


def test_rolling_covariance_returns_one_square_matrix_per_rebalancing_date():
    levels = make_factor_levels()
    sigmas = rolling_factor_covar(levels)
    assert len(sigmas) > 0
    assert all(s.shape == (4, 4) for s in sigmas.values())
    assert all(np.allclose(s, s.T) for s in sigmas.values())


def test_rolling_covariance_rejects_a_non_positive_span():
    levels = make_factor_levels()
    with pytest.raises(ValueError, match='span_months must be positive'):
        rolling_factor_covar(levels, span_months=0)


def test_closest_sigma_falls_back_to_the_preceding_quarter():
    a, b = pd.Timestamp('2010-03-31'), pd.Timestamp('2010-06-30')
    sigmas = {a: np.eye(2)}
    default = np.zeros((2, 2))
    assert np.allclose(closest_or_default_sigma(sigmas, b, default), np.eye(2))
    assert np.allclose(closest_or_default_sigma(sigmas, pd.Timestamp('2009-12-31'), default),
                       default)


def test_monthly_log_returns_rejects_a_non_datetime_index():
    with pytest.raises(ValueError, match='DatetimeIndex'):
        factor_monthly_log_returns(pd.DataFrame({'a': [1.0, 2.0]}))


def test_a_longer_span_gives_a_smoother_covariance_path():
    """A longer EWMA span varies less through time, which is what span means."""
    levels = make_factor_levels()
    def path_variation(span):
        sigmas = rolling_factor_covar(levels, span_months=span)
        series = np.array([np.trace(s) for _, s in sorted(sigmas.items())])
        return float(np.std(np.diff(series)))
    assert path_variation(120) < path_variation(24)
