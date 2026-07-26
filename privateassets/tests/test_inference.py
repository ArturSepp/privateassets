"""Tests for the block-resampling inference layer."""

# packages
import dataclasses
import numpy as np
import pandas as pd
import pytest
# qis / project
import qis
from privateassets.matf import BetaBootstrap, bootstrap_factor_betas
from privateassets.tests.synthetic_data import make_factor_levels

pytest.importorskip('factorlasso', reason='resampling refits need the [factors] extra')

TRUE_BETA = np.array([0.60, -0.20, 0.90, 0.10])


def _panel(noise: float = 0.02, seed: int = 3):
    levels = make_factor_levels(start='2004-12-31', end='2024-12-31')
    factors = np.log(levels.resample('QE').last()).diff().dropna()
    rng = np.random.default_rng(seed)
    asset = pd.Series(factors.values @ TRUE_BETA + rng.normal(0.0, noise, len(factors)),
                      index=factors.index, name='asset')
    return asset, factors


def test_bootstrap_returns_a_populated_distribution():
    asset, factors = _panel()
    out = bootstrap_factor_betas(asset, factors, num_samples=40, block_size=8, seed=1)
    assert isinstance(out, BetaBootstrap)
    assert len(out.draws) + out.num_failed == 40
    assert list(out.draws.columns) == list(factors.columns)


def test_interval_brackets_the_point_estimate():
    """The observed-sample fit sits inside its own resampled interval."""
    asset, factors = _panel()
    out = bootstrap_factor_betas(asset, factors, num_samples=60, block_size=8, seed=2)
    for factor in ('Equity', 'Credit'):
        assert out.lower[factor] <= out.point[factor] <= out.upper[factor]


def test_interval_covers_the_true_loading():
    asset, factors = _panel(noise=0.01)
    out = bootstrap_factor_betas(asset, factors, num_samples=60, block_size=8, seed=3)
    assert out.lower['Credit'] <= 0.90 <= out.upper['Credit']


def test_same_seed_reproduces_the_interval():
    """A published interval must be reproducible from its seed."""
    asset, factors = _panel()
    kw = dict(num_samples=30, block_size=8)
    a = bootstrap_factor_betas(asset, factors, seed=7, **kw)
    b = bootstrap_factor_betas(asset, factors, seed=7, **kw)
    pd.testing.assert_frame_equal(a.draws, b.draws)


def test_different_seed_gives_a_different_sample():
    asset, factors = _panel()
    kw = dict(num_samples=30, block_size=8)
    a = bootstrap_factor_betas(asset, factors, seed=7, **kw)
    b = bootstrap_factor_betas(asset, factors, seed=8, **kw)
    assert not a.draws.equals(b.draws)


def test_resampling_is_paired_across_the_asset_and_its_factors():
    """A shared index keeps each quarter's asset return with its own factors.

    Resampling the two sides independently would break the covariance the
    loading measures, and the estimate would collapse towards zero. Compare a
    paired bootstrap against one that shuffles the asset against fixed factors.
    """
    asset, factors = _panel(noise=0.01)
    paired = bootstrap_factor_betas(asset, factors, num_samples=30, block_size=8, seed=4)

    rng = np.random.default_rng(0)
    scrambled = asset.copy()
    scrambled[:] = asset.values[rng.permutation(len(asset))]
    broken = bootstrap_factor_betas(scrambled, factors, num_samples=30, block_size=8, seed=4)

    assert abs(paired.draws['Credit'].mean()) > abs(broken.draws['Credit'].mean())


def test_provenance_is_recorded():
    """The qis version is part of the result: STATIONARY draws changed at 5.1.0."""
    asset, factors = _panel()
    out = bootstrap_factor_betas(asset, factors, num_samples=10, block_size=8)
    assert out.qis_version != 'unknown'
    assert out.seed == 1
    assert out.ci == (5.0, 95.0)


def test_bootstrap_type_is_configurable():
    asset, factors = _panel()
    kw = dict(num_samples=20, block_size=8, seed=5)
    stationary = bootstrap_factor_betas(asset, factors,
                                        bootstrap_type=qis.BootstrapType.STATIONARY, **kw)
    iid = bootstrap_factor_betas(asset, factors,
                                 bootstrap_type=qis.BootstrapType.IID, **kw)
    assert not stationary.draws.equals(iid.draws)


def test_share_at_zero_reports_a_binding_constraint():
    """Under a binding sign constraint the mass sits on the boundary.

    Reported as a share of draws, deliberately not as a p-value: the quantity is
    monotone in how binding the constraint is, not in the strength of evidence.
    """
    from privateassets.matf import SignConstraint
    asset, factors = _panel()
    out = bootstrap_factor_betas(asset, factors, num_samples=30, block_size=8, seed=6,
                                 fit_kwargs={'sign_constraints': {'Rates': SignConstraint.POS},
                                             'auto_sign_constraints': False})
    assert out.share_at_zero['Rates'] > 0.5
    assert 0.0 <= out.share_at_zero.min() <= out.share_at_zero.max() <= 1.0


def test_result_is_immutable():
    asset, factors = _panel()
    out = bootstrap_factor_betas(asset, factors, num_samples=10, block_size=8)
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.point = None


def test_rejects_invalid_arguments():
    asset, factors = _panel()
    with pytest.raises(ValueError, match='num_samples must be positive'):
        bootstrap_factor_betas(asset, factors, num_samples=0)
    with pytest.raises(ValueError, match='block_size must be positive'):
        bootstrap_factor_betas(asset, factors, block_size=0)
    with pytest.raises(ValueError, match='ci must be an increasing pair'):
        bootstrap_factor_betas(asset, factors, ci=(95.0, 5.0))
    with pytest.raises(ValueError, match='more than block_size'):
        bootstrap_factor_betas(asset.iloc[:5], factors, block_size=12)
