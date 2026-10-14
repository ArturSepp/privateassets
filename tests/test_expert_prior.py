"""Synthetic regressions for the selected-factor expert prior."""

# packages
import dataclasses
import inspect
import numpy as np
import pandas as pd
import pytest
# qis / project
from privateassets import ExpertPrior, PriorDiagnostics
from privateassets.matf import SignConstraint, estimate_matf_alpha, fit_factor_betas
from tests.synthetic_data import make_collinear_factor_panel, make_factor_driven_panel

factorlasso = pytest.importorskip('factorlasso', reason='expert priors need the [factors] extra')


def _floor_panel():
    """A second correlated factor explains the asset, leaving the expert under its floor."""
    return make_collinear_factor_panel(rho=0.95, beta=np.array([0., 0., 1., 0.]),
                                       n_quarters=120, residual_vol=0.005, seed=19)[:2]


def test_no_prior_reproduces_the_unmodified_anchor():
    """Fails if prior support changes the previously recorded default loading fit."""
    asset, factors, _ = make_collinear_factor_panel()
    fit = fit_factor_betas(asset, factors, expert_prior=None)
    np.testing.assert_allclose(fit.beta.values, [0.8165657654648387, -0.00010855937833621083,
                                               0.3601913634970741, -1.4066459264354485e-19],
                               atol=1e-10, rtol=0)
    assert fit.alpha_const == pytest.approx(-0.009611572623667852, abs=1e-10, rel=0)
    assert fit.r2 == pytest.approx(0.9048916746971367, abs=1e-10, rel=0)
    assert fit.n_obs == 48
    assert fit.prior is None


@pytest.mark.parametrize('span', [None, 18])
def test_floor_matches_aligned_statistics_and_independent_hac(span):
    """Fails on wrong alignment, weights, HAC bandwidth, centre or floor extraction."""
    asset, factors = _floor_panel()
    asset = asset.iloc[2:].copy()
    asset.iloc[7] = np.nan
    fit = fit_factor_betas(asset, factors, span=span, expert_prior=ExpertPrior('Equity'))
    aligned = pd.concat([asset, factors], axis=1, join='inner').dropna()
    y, x = aligned.iloc[:, 0], aligned[['Equity']]
    reference = factorlasso.compute_expert_prior_statistics(x, y, span=span, hac_lags=1,
                                                           min_periods=12).loc['Equity']
    prior = fit.prior
    assert isinstance(prior, PriorDiagnostics)
    assert prior.centre == pytest.approx(reference.reference_beta, abs=1e-8, rel=0)
    assert prior.se_hac == pytest.approx(reference.se_hac, abs=1e-8, rel=0)
    assert prior.floor == pytest.approx(max(0, abs(prior.centre) - prior.se_hac), abs=1e-8, rel=0)
    assert prior.n_obs == len(aligned)

    # Independent test reference: direct normal equations and Bartlett score products,
    # not factorlasso.compute_expert_prior_statistics, the routine used by the fit itself.
    design = np.column_stack([np.ones(len(x)), x.values])
    weights = np.ones(len(x)) if span is None else (1 - 2 / (span + 1)) ** np.arange(len(x)-1, -1, -1)
    bread = np.linalg.inv(design.T @ (weights[:, None] * design))
    coefficient = bread @ (design.T @ (weights * y.values))
    scores = (weights * (y.values - design @ coefficient))[:, None] * design
    meat = scores.T @ scores
    lag_product = scores[1:].T @ scores[:-1]
    meat += 0.5 * (lag_product + lag_product.T)
    covariance = bread @ meat @ bread * len(x) / (len(x) - 2)
    assert prior.centre == pytest.approx(coefficient[1], abs=1e-8, rel=0)
    assert prior.se_hac == pytest.approx(np.sqrt(covariance[1, 1]), abs=1e-8, rel=0)


@pytest.mark.parametrize('direction', [1, -1])
def test_floor_changes_the_solve_in_both_directions(direction):
    """Fails if the floor is omitted, applied after fitting, or oriented positively only."""
    asset, factors = _floor_panel()
    asset = direction * asset
    plain = fit_factor_betas(asset, factors, span=None, auto_sign_constraints=False)
    fit = fit_factor_betas(asset, factors, span=None, auto_sign_constraints=False,
                           expert_prior=ExpertPrior('Equity'))
    assert fit.prior.imposed and fit.prior.floor > 0
    assert direction * plain.beta['Equity'] < fit.prior.floor - 0.1
    assert direction * fit.beta['Equity'] >= fit.prior.floor - 1e-9
    assert direction * fit.beta['Equity'] == pytest.approx(fit.prior.floor, abs=1e-5)


def test_free_and_zero_codes_are_distinct():
    """Fails if FREE is encoded as zero and silently excludes a negative loading."""
    asset, factors = _floor_panel()
    free = fit_factor_betas(-asset, factors, span=None, auto_sign_constraints=False,
                            sign_constraints={'Credit': SignConstraint.FREE})
    zero = fit_factor_betas(-asset, factors, span=None, auto_sign_constraints=False,
                            sign_constraints={'Credit': SignConstraint.ZERO})
    assert free.beta['Credit'] < -0.5
    assert zero.beta['Credit'] == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize('kwargs', [dict(factor=''), dict(factor='  '), dict(factor=1),
                                  dict(bound_n_std=-1), dict(bound_n_std=np.inf),
                                  dict(bound_n_std=np.nan), dict(bound_n_std=True),
                                  dict(hac_lags=True), dict(hac_lags=-1), dict(hac_lags=1.5)])
def test_prior_settings_reject_invalid_values(kwargs):
    """Fails if invalid prior settings reach the optimizer instead of construction validation."""
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        ExpertPrior(**{'factor': 'Equity', **kwargs})


def test_unknown_expert_factor_is_named():
    """Fails if a missing selected factor is silently replaced with an automatic selection."""
    asset, factors = _floor_panel()
    with pytest.raises(ValueError, match='not a column'):
        fit_factor_betas(asset, factors, expert_prior=ExpertPrior('not a column'))


def test_overrides_compose_with_the_ols_centre(monkeypatch):
    """Fails if a zero-filled frame cancels OLS centres or another-factor override is lost."""
    asset, factors = _floor_panel()
    models = []
    original = factorlasso.LassoModel.fit

    def capture(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        models.append(model)
        return result

    monkeypatch.setattr(factorlasso.LassoModel, 'fit', capture)
    prior = ExpertPrior('Equity')
    plain = fit_factor_betas(asset, factors, span=None, expert_prior=prior, auto_sign_constraints=False)
    fit_factor_betas(asset, factors, span=None, expert_prior=prior,
                     prior_beta={'Rates': 0.5}, auto_sign_constraints=False)
    base, override = [m.effective_beta_prior_.loc['asset'] for m in models]
    assert base['Equity'] == pytest.approx(plain.prior.centre)
    assert (base.drop('Equity') == 0).all()
    assert override['Rates'] == 0.5
    assert override['Equity'] == pytest.approx(base['Equity'])


@pytest.mark.parametrize('kwargs', [dict(prior_beta={'Equity': 0.25}),
                                  dict(prior_beta={'Equity': 0.0}),
                                  dict(sign_constraints={'Equity': SignConstraint.NEG}),
                                  dict(sign_constraints={'Equity': SignConstraint.ZERO})])
def test_manual_prior_and_hard_signs_take_precedence(kwargs):
    """Fails on converting an unreported count to int or imposing a contradictory expert bound."""
    asset, factors = _floor_panel()
    fit = fit_factor_betas(asset, factors, span=None, expert_prior=ExpertPrior('Equity'), **kwargs)
    assert not fit.prior.imposed
    assert fit.prior.floor == 0
    assert fit.prior.centre > 0
    assert np.isnan(fit.prior.se_hac)
    assert fit.prior.n_obs is None
    if 'sign_constraints' in kwargs:
        assert fit.beta['Equity'] <= 1e-9


def test_missing_diagnostics_warn_without_inventing_a_floor(monkeypatch):
    """Fails if an empty upstream diagnostics table crashes or invents usable uncertainty."""
    asset, factors = _floor_panel()
    original = factorlasso.LassoModel.fit

    def drop_diagnostics(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        model.prior_bound_diagnostics_ = pd.DataFrame()
        return result

    monkeypatch.setattr(factorlasso.LassoModel, 'fit', drop_diagnostics)
    with pytest.warns(UserWarning, match='diagnostics unavailable') as warnings:
        fit = fit_factor_betas(asset, factors, span=None, expert_prior=ExpertPrior('Equity'))
    assert len(warnings) == 1
    assert np.isnan(fit.prior.centre) and np.isnan(fit.prior.se_hac)
    assert fit.prior.n_obs is None and fit.prior.floor == 0 and not fit.prior.imposed


def test_old_dependency_reports_its_version(monkeypatch):
    """Fails if an old optional dependency produces an opaque constructor error."""
    asset, factors = _floor_panel()

    class OldLasso:
        pass

    monkeypatch.setattr(factorlasso, 'LassoModel', OldLasso)
    with pytest.raises(RuntimeError, match=r'factorlasso>=1.0.0; installed'):
        fit_factor_betas(asset, factors, expert_prior=ExpertPrior('Equity'))


def test_estimator_records_the_prior_in_provenance():
    """Fails if estimator forwarding or serializable prior provenance is omitted."""
    prior = ExpertPrior('Equity')
    result = estimate_matf_alpha(*make_factor_driven_panel(), expert_prior=prior,
                                 fit_kwargs={'span': None})
    assert result.provenance['expert_prior'] == dataclasses.asdict(prior)
    assert result.betas.prior.factor == prior.factor


def test_public_exports_and_positional_compatibility():
    """Fails if exports are missing or the new parameter shifts existing positional calls."""
    from privateassets import matf
    assert matf.ExpertPrior is ExpertPrior and matf.PriorDiagnostics is PriorDiagnostics
    for function, following in [(fit_factor_betas, 'reg_lambda'), (estimate_matf_alpha, 'covar_span_months')]:
        parameters = inspect.signature(function).parameters
        assert list(parameters)[list(parameters).index('prior_beta') + 1] == following
        assert parameters['expert_prior'].kind == inspect.Parameter.KEYWORD_ONLY
        assert 'expert_prior:' in function.__doc__
