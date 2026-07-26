"""
privateassets.matf — multi-factor, money-weighted PME.

Estimates risk-adjusted alpha and systematic factor exposures directly from
private-asset cash flows. Generalises Direct Alpha, KS-PME and GPME from a
single benchmark to a tradable multi-factor deflator.

Two layers:

- classical single-benchmark measures (``ks_pme``, ``direct_alpha``,
  ``long_nickels_pme``, ``xirr``) and the vintage statistics around them
- the MATF deflator (``matf_deflator``) with its point-in-time factor
  covariance, and ``vintage_direct_alpha``, which turns any deflator path into
  an annualised alpha

Unsmoothing lives in ``qis``. Estimate the AR(1) coefficient here with
``fit_panel_ar1``, then apply it with
``qis.unsmooth_returns_glm(returns, ar_order=1, theta=theta_hat)``.

Importing this module has no side effects and reads nothing from disk.
"""
from privateassets.matf._pme import (
    CF_COLUMNS,
    NAV_COLUMNS,
    cap_weighted_aggregates,
    cf_with_terminal_for_vintage,
    compute_vintage_stats,
    direct_alpha,
    ks_pme,
    load_cash_flows,
    load_navs,
    long_nickels_pme,
    vintage_direct_alpha,
    xirr,
)
from privateassets.matf._deflator import (
    DEFAULT_BURNIN_MONTHS,
    DEFAULT_COVAR_SPAN_MONTHS,
    build_rolling_sigma,
    closest_or_default_sigma,
    factor_log_levels_panel,
    factor_monthly_log_returns,
    matf_deflator,
    rolling_ewma_quarterly_covar,
)
from privateassets.matf._panel_mle import (
    MIN_OBS_FOR_THETA,
    MIN_OBS_PER_VINTAGE_MLE,
    fisher_info_panel_ar1,
    fit_panel_ar1,
    panel_ar1_neg_log_likelihood,
)

__all__ = [
    # cash-flow containers
    'CF_COLUMNS',
    'NAV_COLUMNS',
    'load_cash_flows',
    'load_navs',
    'cf_with_terminal_for_vintage',
    # single-benchmark measures
    'xirr',
    'ks_pme',
    'direct_alpha',
    'long_nickels_pme',
    'compute_vintage_stats',
    'cap_weighted_aggregates',
    # multi-factor deflator
    'matf_deflator',
    'vintage_direct_alpha',
    'factor_monthly_log_returns',
    'factor_log_levels_panel',
    'rolling_ewma_quarterly_covar',
    'build_rolling_sigma',
    'closest_or_default_sigma',
    'DEFAULT_COVAR_SPAN_MONTHS',
    'DEFAULT_BURNIN_MONTHS',
    # unsmoothing coefficient
    'fit_panel_ar1',
    'panel_ar1_neg_log_likelihood',
    'fisher_info_panel_ar1',
    'MIN_OBS_FOR_THETA',
    'MIN_OBS_PER_VINTAGE_MLE',
]
