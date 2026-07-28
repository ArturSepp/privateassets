"""
privateassets.matf._estimator — the MATF estimator, end to end.

Composes the pieces into one call and returns everything it computed:

    marks and cash flows
      -> reported returns per vintage           (_returns)
      -> pooled manager series                  (_returns)
      -> smoothing coefficient theta            (_panel_mle)
      -> unsmoothed series                      (qis.unsmooth_returns_glm)
      -> factor loadings beta                   (_betas)
      -> point-in-time factor covariance        (_deflator)
      -> per-vintage deflator and alpha         (_deflator, _pme)
      -> capital-weighted aggregate             (here)
      -> resampled interval on beta             (_inference, optional)

Two properties the precursor did not have.

**Nothing is printed and nothing is written to disk.** Every intermediate a
caller might want — the loadings, the covariance in force, the per-vintage
alphas, the interval — is a field on the returned object. A confidence interval
that can only be read out of a spreadsheet is not a result a program can use.

**The covariance is point-in-time.** The deflator at a cash-flow date uses the
covariance estimated from returns observed by that date. A single full-sample
matrix applied to every flow deflates a 2001 contribution with a covariance that
knows about 2008, which is correct for a descriptive exhibit and wrong the moment
the deflator is described as a portfolio an investor could have held.

The loadings remain in-sample by construction: one beta over the whole panel,
applied to every vintage. That is the identification design and it is recorded in
``MatfResult.provenance`` so it cannot be forgotten when the number is quoted.
"""

# packages
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
# qis / project
import qis
from privateassets.matf._betas import FactorBetas, SignConstraint, fit_factor_betas
from privateassets.matf._deflator import (factor_log_levels_panel, matf_deflator,
                                          rolling_factor_covar)
from privateassets.matf._inference import BetaBootstrap, bootstrap_factor_betas
from privateassets.matf._panel_mle import BiasCorrection, fit_panel_ar1
from privateassets.matf._pme import cf_with_terminal_for_vintage, vintage_direct_alpha
from privateassets.matf._returns import (infer_reporting_frequency, nav_implied_returns,
                                         pool_vintage_returns)

DEFAULT_DPI_THRESHOLD = 0.8  # a vintage enters the aggregate above this realised multiple
DEFLATOR_FREQ = 'QE'  # the factor panel grid the deflator runs on, whatever the reporting frequency


@dataclass(frozen=True)
class MatfResult:
    """Everything the estimator computed, in one immutable snapshot.

    Attributes:
        vintage_alpha: one row per vintage with contributions, distributions,
            DPI, and the annualised MATF alpha. NaN alpha means the vintage could
            not be priced, normally because it starts before the covariance
            burn-in.
        cap_weighted_alpha: contribution-weighted alpha across mature vintages.
        equal_weighted_alpha: unweighted mean across the same vintages.
        n_mature: how many vintages cleared the DPI gate and priced.
        betas: the factor-loading fit, or None when loadings were supplied.
        beta_bootstrap: resampled interval on the loadings, when requested.
        theta: the AR(1) smoothing coefficient used, corrected when asked.
        theta_raw: the uncorrected estimate, or the supplied value.
        theta_se: asymptotic standard error of the raw estimate. A bias
            correction moves the point estimate without narrowing it.
        pooled_returns: the pooled manager series before unsmoothing.
        unsmoothed_returns: the series the loadings were fitted on.
        sigma_by_period: point-in-time factor covariance, quarterly units.
        provenance: package versions, seed, and the specification choices a
            published number has to be quoted with.
    """

    vintage_alpha: pd.DataFrame
    cap_weighted_alpha: float
    equal_weighted_alpha: float
    n_mature: int
    betas: Optional[FactorBetas]
    beta_bootstrap: Optional[BetaBootstrap]
    theta: float
    theta_raw: float
    theta_se: float
    pooled_returns: pd.Series
    unsmoothed_returns: pd.Series
    sigma_by_period: Dict[pd.Timestamp, np.ndarray]
    provenance: Dict[str, Any] = field(default_factory=dict)


def estimate_matf_alpha(cf: pd.DataFrame,
                        navs: pd.DataFrame,
                        factor_levels: pd.DataFrame,  # daily EXCESS factor index levels
                        rf_rate: pd.Series,  # simple risk-free yield per quarter, quarter-end index
                        freq: str = 'QE',  # frequency the funds report on
                        sign_constraints: Optional[Dict[str, SignConstraint]] = None,  # None: all free
                        prior_beta: Optional[Dict[str, float]] = None,  # None: shrink towards zero
                        covar_span_months: int = 60,  # EWMA span for the factor covariance
                        dpi_threshold: float = DEFAULT_DPI_THRESHOLD,  # maturity gate for the aggregate
                        beta: Optional[pd.Series] = None,  # None: fit it; otherwise price against this
                        theta: Optional[float] = None,  # None: estimate it from the panel
                        bias_correction: BiasCorrection = BiasCorrection.NONE,  # for the estimated theta
                        num_bootstrap: int = 0,  # 0 skips the resampled interval
                        block_size: int = 12,  # mean resample block, in reporting periods
                        seed: int = 1,
                        sigma_default: Optional[np.ndarray] = None,  # covariance before the first estimate
                        fit_kwargs: Optional[Dict[str, Any]] = None,  # passed to fit_factor_betas
                        ) -> MatfResult:
    """estimate multi-factor money-weighted alpha for a panel of vintages.

    Args:
        cf: tidy cash-flow frame. Contributions negative.
        navs: tidy NAV frame.
        factor_levels: factor index levels, excess of the risk-free rate, at
            daily or higher frequency.
        rf_rate: simple risk-free yield per quarter, on a quarter-end index.
        freq: frequency the funds report on. The deflator always runs on the
            factor panel's quarterly grid, so this governs the return
            reconstruction and the loading fit only.
        sign_constraints: admissible sign per factor.
        prior_beta: shrinkage target per factor. None shrinks towards zero.
        beta: price against these loadings instead of fitting them, indexed by
            factor name. Supplying them skips the fit and the resampling, which
            is how a production loading vector is applied to a new panel, and how
            the deflator path is tested without the in-sample fit moving under it.
        covar_span_months: EWMA span for the point-in-time factor covariance.
        dpi_threshold: a vintage enters the aggregate when its DPI exceeds this.
        theta: fix the smoothing coefficient instead of estimating it. Passing a
            value taken from another dataset makes the result depend on data not
            supplied here, so record it if you do.
        bias_correction: correct the small-sample bias in the estimated
            coefficient. Ignored when ``theta`` is supplied. The default leaves
            the estimate alone, and the raw value is reported either way.
        num_bootstrap: resample draws for the interval on the loadings. Zero
            skips it, which is much faster.
        block_size: mean resample block length, in reporting periods.
        seed: seed for the resampling.
        sigma_default: covariance to use before the first point-in-time
            estimate. None leaves those vintages unpriced rather than lending
            them a matrix estimated later.
        fit_kwargs: forwarded to :func:`~privateassets.matf.fit_factor_betas`.

    Returns:
        A :class:`MatfResult`.

    Raises:
        ValueError: if the panel does not report at one frequency, if the factor
            panel does not overlap the reporting window, or if no vintage can be
            priced.
    """
    returns_panel, capital = nav_implied_returns(cf, navs, freq=freq, require_regular=True)
    pooled = pool_vintage_returns(returns_panel, capital)
    if pooled.empty:
        raise ValueError("no reporting period carries a usable return for any vintage")

    theta_se = float('nan')
    theta_raw = theta
    if theta is None:
        series_list = []
        for vintage in returns_panel.columns:
            values = returns_panel[vintage].dropna().values
            if len(values) >= 3:
                series_list.append((str(vintage), values - values.mean()))
        if not series_list:
            raise ValueError("no vintage has enough observations to estimate theta")
        panel_fit = fit_panel_ar1(series_list, bias_correction=bias_correction, seed=seed)
        theta = float(panel_fit['theta_hat'])
        theta_se = float(panel_fit['se'])
        theta_raw = float(panel_fit['theta_raw'])

    unsmoothed = qis.unsmooth_returns_glm(pooled, ar_order=1, theta=theta)
    unsmoothed = pd.Series(unsmoothed, index=pooled.index, name='unsmoothed').dropna()

    factor_returns = np.log(factor_levels.resample(freq).last()).diff().dropna(how='any')
    aligned_index = unsmoothed.index.intersection(factor_returns.index)
    if len(aligned_index) < 3:
        raise ValueError(f"factor panel overlaps the reporting window in only "
                         f"{len(aligned_index)} periods; check the factor levels cover it")

    betas: Optional[FactorBetas] = None
    if beta is None:
        betas = fit_factor_betas(unsmoothed.loc[aligned_index],
                                 factor_returns.loc[aligned_index],
                                 sign_constraints=sign_constraints,
                                 prior_beta=prior_beta,
                                 **(fit_kwargs or {}))
        beta = betas.beta
    else:
        beta = pd.Series(beta)
        unknown = [f for f in beta.index if f not in factor_returns.columns]
        if unknown:
            raise ValueError(f"beta names factors absent from factor_levels: {unknown}")

    beta_bootstrap = None
    if num_bootstrap > 0 and betas is not None:
        beta_bootstrap = bootstrap_factor_betas(unsmoothed.loc[aligned_index],
                                                factor_returns.loc[aligned_index],
                                                num_samples=num_bootstrap,
                                                block_size=block_size,
                                                seed=seed,
                                                fit_kwargs={'sign_constraints': sign_constraints,
                                                            'prior_beta': prior_beta,
                                                            **(fit_kwargs or {})})

    sigma_by_period = rolling_factor_covar(factor_levels, span_months=covar_span_months,
                                           rebalancing_freq=DEFLATOR_FREQ)

    quarter_ends, cum_log_factor, cum_log_rf = _deflator_panel(factor_levels, rf_rate,
                                                               list(beta.index))
    rows = _price_vintages(cf, navs, quarter_ends, cum_log_factor, cum_log_rf,
                           beta.values, sigma_by_period, sigma_default)
    vintage_alpha = pd.DataFrame(rows).sort_values('first_call').reset_index(drop=True)

    mature = vintage_alpha[(vintage_alpha['DPI'] > dpi_threshold)
                           & vintage_alpha['matf_alpha'].notna()]
    weights = mature['contributions']
    cap_weighted = (float((mature['matf_alpha'] * weights).sum() / weights.sum())
                    if weights.sum() > 0 else float('nan'))
    equal_weighted = float(mature['matf_alpha'].mean()) if len(mature) else float('nan')

    return MatfResult(
        vintage_alpha=vintage_alpha,
        cap_weighted_alpha=cap_weighted,
        equal_weighted_alpha=equal_weighted,
        n_mature=len(mature),
        betas=betas,
        beta_bootstrap=beta_bootstrap,
        theta=theta,
        theta_raw=theta_raw,
        theta_se=theta_se,
        pooled_returns=pooled,
        unsmoothed_returns=unsmoothed,
        sigma_by_period=sigma_by_period,
        provenance=_provenance(freq=freq, covar_span_months=covar_span_months,
                               dpi_threshold=dpi_threshold, seed=seed,
                               num_bootstrap=num_bootstrap, theta=theta,
                               bias_correction=bias_correction.value,
                               n_periods=len(aligned_index),
                               beta_was_supplied=betas is None,
                               reporting_months=infer_reporting_frequency(navs).median()),
    )


def _deflator_panel(factor_levels: pd.DataFrame,
                    rf_rate: pd.Series,
                    factors: list,
                    ) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """cumulative log excess factor returns and log risk-free, on the quarterly grid."""
    quarterly = np.log(factor_levels[factors].resample(DEFLATOR_FREQ).last()).diff().fillna(0.0)
    quarter_ends = pd.DatetimeIndex(quarterly.index)
    cum_log_factor = factor_log_levels_panel(quarterly).values
    rf_aligned = pd.Series(rf_rate).reindex(quarter_ends).fillna(0.0)
    return quarter_ends, cum_log_factor, np.log1p(rf_aligned).cumsum().values


def _price_vintages(cf: pd.DataFrame,
                    navs: pd.DataFrame,
                    quarter_ends: pd.DatetimeIndex,
                    cum_log_factor: np.ndarray,
                    cum_log_rf: np.ndarray,
                    beta: np.ndarray,
                    sigma_by_period: Dict[pd.Timestamp, np.ndarray],
                    sigma_default: Optional[np.ndarray],
                    ) -> list:
    """per-vintage MATF alpha against the point-in-time deflator."""
    rows = []
    for vintage in sorted(cf['vintage_label'].unique()):
        cf_g = cf[cf['vintage_label'] == vintage]
        nav_g = navs[navs['vintage_label'] == vintage]
        cf_v, rvpi_nav, _, dates = cf_with_terminal_for_vintage(cf_g, nav_g)

        deflators = matf_deflator(cf_dates=dates, t0=dates[0],
                                  cum_log_factor=cum_log_factor, cum_log_rf=cum_log_rf,
                                  quarter_ends=quarter_ends, beta=beta,
                                  sigma_by_quarter=sigma_by_period,
                                  sigma_default=sigma_default)
        alpha = vintage_direct_alpha(cf_v, rvpi_nav, dates, deflators)

        contributions = -cf_v.loc[cf_v['amount'] < 0, 'amount'].sum()
        distributions = cf_v.loc[cf_v['amount'] > 0, 'amount'].sum()
        rows.append({
            'vintage_label': vintage,
            'first_call': cf_v.loc[cf_v['amount'] < 0, 'date'].min(),
            'contributions': contributions,
            'distributions': distributions,
            'rvpi_nav': rvpi_nav,
            'DPI': distributions / contributions if contributions > 0 else np.nan,
            'matf_alpha': alpha,
        })
    return rows


def _provenance(**spec: Any) -> Dict[str, Any]:
    """package versions and specification, recorded with the result.

    A resampled or unsmoothed number is not reproducible without the version that
    produced it: ``qis.BootstrapType.STATIONARY`` changed its wrapping at
    ``qis 5.1.0``.
    """
    from importlib.metadata import PackageNotFoundError, version

    def _version(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:  # pragma: no cover
            return 'not installed'

    return {
        'qis_version': _version('qis'),
        'factorlasso_version': _version('factorlasso'),
        'privateassets_version': _version('privateassets'),
        'betas_are_in_sample': True,  # unless beta_was_supplied
        'no_interpolation': True,
        'covariance_is_point_in_time': True,
        **spec,
    }
