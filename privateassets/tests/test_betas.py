"""Tests for the sign-constrained, cluster-shrunk factor-loading estimator."""

# packages
import dataclasses
import numpy as np
import pandas as pd
import pytest
# qis / project
from privateassets.matf import (
    FactorBetas,
    SignConstraint,
    fit_factor_betas,
    matf_deflator,
)
from privateassets.tests.synthetic_data import FACTOR_NAMES, make_factor_levels

factorlasso = pytest.importorskip('factorlasso',
                                  reason='factor loadings need the [factors] extra')

TRUE_BETA = np.array([0.60, -0.20, 0.90, 0.10])


def _panel(noise: float = 0.01, seed: int = 7, beta: np.ndarray = TRUE_BETA):
    """Quarterly factor returns and an asset built from a known loading vector."""
    levels = make_factor_levels(start='2004-12-31', end='2024-12-31')
    factors = np.log(levels.resample('QE').last()).diff().dropna()
    rng = np.random.default_rng(seed)
    asset = pd.Series(factors.values @ beta + rng.normal(0.0, noise, len(factors)),
                      index=factors.index, name='asset')
    return asset, factors


def test_recovers_a_known_loading_vector():
    """On a panel built from a known beta the estimate lands near it."""
    asset, factors = _panel(noise=0.005)
    fit = fit_factor_betas(asset, factors, span=None)
    assert isinstance(fit, FactorBetas)
    assert list(fit.beta.index) == FACTOR_NAMES
    assert fit.beta['Equity'] == pytest.approx(0.60, abs=0.15)
    assert fit.beta['Credit'] == pytest.approx(0.90, abs=0.15)


def test_shrinkage_pulls_towards_zero_not_away():
    """A heavier group penalty moves the loading vector towards the origin.

    This is the property that makes the estimator usable on a short collinear
    panel, so it is pinned rather than assumed.
    """
    asset, factors = _panel()
    light = fit_factor_betas(asset, factors, reg_lambda=1e-6, span=None)
    heavy = fit_factor_betas(asset, factors, reg_lambda=1e-1, span=None)
    assert np.linalg.norm(heavy.beta.values) < np.linalg.norm(light.beta.values)


def test_positive_sign_constraint_is_respected():
    """A factor with a truly negative loading is held at or above zero."""
    asset, factors = _panel()
    fit = fit_factor_betas(asset, factors,
                           sign_constraints={'Rates': SignConstraint.POS},
                           auto_sign_constraints=False, span=None)
    assert fit.beta['Rates'] >= -1e-8


def test_negative_sign_constraint_is_respected():
    """A factor with a truly positive loading is held at or below zero."""
    asset, factors = _panel()
    fit = fit_factor_betas(asset, factors,
                           sign_constraints={'Equity': SignConstraint.NEG},
                           auto_sign_constraints=False, span=None)
    assert fit.beta['Equity'] <= 1e-8


def test_zero_constraint_excludes_the_factor():
    """A zeroed factor carries no loading, which is how a collinear factor is dropped."""
    asset, factors = _panel()
    fit = fit_factor_betas(asset, factors,
                           sign_constraints={'Commodities': SignConstraint.ZERO},
                           auto_sign_constraints=False, span=None)
    assert fit.beta['Commodities'] == pytest.approx(0.0, abs=1e-8)


def test_prior_shifts_the_estimate_towards_itself():
    """Shrinking towards a non-zero prior moves the loading that way.

    The prior is an economic view, so it is a caller argument with no default.
    """
    asset, factors = _panel()
    neutral = fit_factor_betas(asset, factors, reg_lambda=1e-2, span=None)
    with_prior = fit_factor_betas(asset, factors, reg_lambda=1e-2, span=None,
                                  prior_beta={'Commodities': 1.5})
    assert with_prior.beta['Commodities'] > neutral.beta['Commodities']


def test_alpha_is_near_zero_when_the_asset_is_pure_factor_exposure():
    """An asset that is exactly a factor basket has no intercept to find."""
    asset, factors = _panel(noise=0.001)
    fit = fit_factor_betas(asset, factors, span=None)
    assert abs(fit.alpha_const) < 0.01


def test_alpha_recovers_an_injected_intercept():
    """Adding a constant to the asset shows up in alpha, roughly one for one."""
    asset, factors = _panel(noise=0.001)
    base = fit_factor_betas(asset, factors, span=None)
    lifted = fit_factor_betas(asset + 0.02, factors, span=None)
    assert lifted.alpha_const - base.alpha_const == pytest.approx(0.02, abs=0.005)


def test_ragged_inputs_are_aligned_on_the_intersection():
    asset, factors = _panel()
    asset_short = asset.iloc[4:]
    fit = fit_factor_betas(asset_short, factors, span=None)
    assert fit.n_obs == len(asset_short) - 12


def test_result_is_an_immutable_snapshot():
    asset, factors = _panel()
    fit = fit_factor_betas(asset, factors, span=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fit.beta = None


def test_rejects_a_sign_constraint_on_an_unknown_factor():
    asset, factors = _panel()
    with pytest.raises(ValueError, match='absent from factor_returns'):
        fit_factor_betas(asset, factors, sign_constraints={'NotAFactor': SignConstraint.POS})


def test_rejects_a_panel_shorter_than_the_warmup():
    asset, factors = _panel()
    with pytest.raises(ValueError, match='warmup_period'):
        fit_factor_betas(asset.iloc[:8], factors, warmup_period=12)


def test_rejects_the_wrong_container_types():
    asset, factors = _panel()
    with pytest.raises(ValueError, match='must be a Series'):
        fit_factor_betas(asset.to_frame(), factors)
    with pytest.raises(ValueError, match='must be a DataFrame'):
        fit_factor_betas(asset, factors['Equity'])


def test_insample_covariance_is_labelled_and_not_point_in_time():
    """The returned covariance uses the whole panel, which is why it is named so.

    Feeding it to a deflator applied at a date is look-ahead. The name is the
    guard, so this test asserts the name exists and that the matrix is in fact
    full-sample.
    """
    asset, factors = _panel()
    fit = fit_factor_betas(asset, factors, span=None)
    assert hasattr(fit, 'sigma_quarterly_insample')
    early = fit_factor_betas(asset.iloc[:40], factors.iloc[:40], span=None)
    assert not np.allclose(fit.sigma_quarterly_insample.values,
                           early.sigma_quarterly_insample.values)


def test_betas_feed_the_deflator_end_to_end():
    """The estimator's output is accepted by the deflator without adaptation.

    This is the gap v0.1.0 shipped with: a deflator that needed a beta nothing
    in the package produced.
    """
    asset, factors = _panel()
    fit = fit_factor_betas(asset, factors, span=None)

    quarter_ends = pd.DatetimeIndex(factors.index)
    cum_log_factor = factors.cumsum().values
    cum_log_rf = np.log1p(pd.Series(0.005, index=quarter_ends)).cumsum().values

    deflators = matf_deflator(cf_dates=list(quarter_ends[[10, 30, 50]]),
                              t0=quarter_ends[10],
                              cum_log_factor=cum_log_factor,
                              cum_log_rf=cum_log_rf,
                              quarter_ends=quarter_ends,
                              beta=fit.beta.values,
                              sigma_default=fit.sigma_quarterly_insample.values)
    assert np.all(np.isfinite(deflators))
    assert deflators[0] == pytest.approx(np.exp(0.5 * (fit.beta.values @ np.diag(
        fit.sigma_quarterly_insample.values) - fit.beta.values
        @ fit.sigma_quarterly_insample.values @ fit.beta.values) * (1 / 365.25 * 4)), rel=0.05)
