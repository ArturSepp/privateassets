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
- sign-constrained, cluster-shrunk factor loadings (``fit_factor_betas``), which
  produce the ``beta`` the deflator needs. Requires the ``[factors]`` extra.

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
    DEFAULT_COVAR_SPAN_MONTHS,
    closest_or_default_sigma,
    factor_log_levels_panel,
    factor_monthly_log_returns,
    horizon_indices,
    matf_deflator,
    rolling_factor_covar,
)
from privateassets.matf._betas import (
    DEFAULT_REG_LAMBDA,
    DEFAULT_SPAN,
    DEFAULT_SPAN_FREQ,
    DEFAULT_WARMUP_PERIOD,
    FactorBetas,
    SignConstraint,
    fit_factor_betas,
)
from privateassets.matf._benchmarks import (
    MIN_ANNUAL_VARIANCE,
    QUARTERS_PER_YEAR,
    kn16_gpme_deflator,
    kn16_sdf_params,
    kn24_benchmark_deflator,
)
from privateassets.matf._estimator import (
    DEFAULT_DPI_THRESHOLD,
    DEFLATOR_FREQ,
    MatfResult,
    estimate_matf_alpha,
)
from privateassets.matf._inference import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_ZERO_TOLERANCE,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SEED,
    BetaBootstrap,
    bootstrap_factor_betas,
)
from privateassets.matf._returns import (
    FREQUENCY_BY_MONTHS,
    PERIODS_PER_YEAR_BY_MONTHS,
    infer_reporting_frequency,
    nav_implied_returns,
    pool_vintage_returns,
    split_by_reporting_frequency,
)
from privateassets.matf._panel_mle import (
    DEFAULT_BIAS_DRAWS,
    MIN_OBS_FOR_THETA,
    MIN_OBS_PER_VINTAGE_MLE,
    BiasCorrection,
    bootstrap_corrected_theta,
    kendall_corrected_theta,
    simulate_panel_ar1,
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
    'rolling_factor_covar',
    'closest_or_default_sigma',
    'horizon_indices',
    'DEFAULT_COVAR_SPAN_MONTHS',
    # the estimator, end to end
    'estimate_matf_alpha',
    'MatfResult',
    'DEFAULT_DPI_THRESHOLD',
    'DEFLATOR_FREQ',
    # reported returns
    'nav_implied_returns',
    'pool_vintage_returns',
    'infer_reporting_frequency',
    'split_by_reporting_frequency',
    'FREQUENCY_BY_MONTHS',
    'PERIODS_PER_YEAR_BY_MONTHS',
    # single-factor benchmarks the MATF deflator displaces
    'kn24_benchmark_deflator',
    'kn16_sdf_params',
    'kn16_gpme_deflator',
    'QUARTERS_PER_YEAR',
    'MIN_ANNUAL_VARIANCE',
    # resampling inference
    'bootstrap_factor_betas',
    'BetaBootstrap',
    'DEFAULT_NUM_SAMPLES',
    'DEFAULT_BLOCK_SIZE',
    'DEFAULT_SEED',
    'DEFAULT_ZERO_TOLERANCE',
    # factor loadings
    'fit_factor_betas',
    'FactorBetas',
    'SignConstraint',
    'DEFAULT_REG_LAMBDA',
    'DEFAULT_SPAN',
    'DEFAULT_WARMUP_PERIOD',
    'DEFAULT_SPAN_FREQ',
    # unsmoothing coefficient
    'fit_panel_ar1',
    'panel_ar1_neg_log_likelihood',
    'fisher_info_panel_ar1',
    'MIN_OBS_FOR_THETA',
    'MIN_OBS_PER_VINTAGE_MLE',
    'BiasCorrection',
    'kendall_corrected_theta',
    'bootstrap_corrected_theta',
    'simulate_panel_ar1',
    'DEFAULT_BIAS_DRAWS',
]
