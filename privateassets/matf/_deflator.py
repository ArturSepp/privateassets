"""
privateassets.matf._deflator — the MATF multi-factor deflator.

A single-benchmark PME divides fund cash flows by the return of one index. The
MATF deflator divides them by the return of a *tradable multi-factor portfolio*
with loadings beta, which is what generalises Direct Alpha, KS-PME and GPME to a
multi-factor setting:

    R^b_h = exp{ rf_h + beta' r_h
                 + (1/2) h (beta' diag(Sigma))
                 - (1/2) h (beta' Sigma beta) }

The two Jensen terms convert the log-return of the levered factor basket into
the arithmetic return an investor holding it would have earned. Factor inputs
are excess log returns, so rf enters once and only once.

Sigma is estimated point-in-time: at each quarter end the covariance uses only
the factor returns observed by that date, so the deflator at a cash-flow date
carries no information from after it.
"""

# packages
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
# qis / project
import qis

DEFAULT_COVAR_SPAN_MONTHS = 60  # EWMA span in months; the documented production setting


def factor_monthly_log_returns(levels: pd.DataFrame) -> pd.DataFrame:
    """end-of-month log returns of a daily factor level panel.

    Args:
        levels: daily factor levels, indexed by date.

    Returns:
        Monthly log returns, indexed by month end, first month dropped.

    Raises:
        ValueError: if ``levels`` is not a DataFrame with a DatetimeIndex.
    """
    if not isinstance(levels, pd.DataFrame):
        raise ValueError(f"levels must be a DataFrame, got {type(levels)!r}")
    if not isinstance(levels.index, pd.DatetimeIndex):
        raise ValueError(f"levels must carry a DatetimeIndex, got {type(levels.index)!r}")
    month_end = levels.resample('ME').last()
    return np.log(month_end / month_end.shift(1)).dropna(how='all')


def factor_log_levels_panel(log_returns: pd.DataFrame) -> pd.DataFrame:
    """cumulative log levels of a factor return panel, starting at zero."""
    return log_returns.cumsum()


def horizon_indices(cf_dates: List[pd.Timestamp],
                    t0: pd.Timestamp,
                    quarter_ends: pd.DatetimeIndex,
                    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """map cash-flow dates onto the factor panel's quarter-end grid.

    Every deflator in this package shares this convention, so that a KN and a
    MATF deflator evaluated on the same cash flows differ in their pricing
    kernel and in nothing else.

    A date is mapped to the last quarter end at or before it, the ``asof``
    convention the factor panel is reported on. A date before the first quarter
    end is marked out of panel rather than pinned to index zero: pinning
    fabricates a horizon that does not exist.

    Args:
        cf_dates: dates to locate.
        t0: horizon origin.
        quarter_ends: quarter-end dates the cumulative arrays are indexed by.

    Returns:
        Tuple of the per-date panel index, the origin's panel index (negative
        when the origin precedes the panel), the ACT/365.25 horizon in years,
        and a boolean mask of dates that fall inside the panel.
    """
    quarters_ns = np.asarray([pd.Timestamp(q).value for q in quarter_ends], dtype=np.int64)
    cf_ns = np.asarray([pd.Timestamp(t).value for t in cf_dates], dtype=np.int64)
    idx_t = np.searchsorted(quarters_ns, cf_ns, side='right') - 1
    idx_t0 = int(np.searchsorted(quarters_ns, pd.Timestamp(t0).value, side='right') - 1)
    horizon_years = np.asarray([max((pd.Timestamp(t) - pd.Timestamp(t0)).days, 1) / 365.25
                                for t in cf_dates])
    return idx_t, idx_t0, horizon_years, idx_t >= 0


def rolling_factor_covar(levels: pd.DataFrame,
                         span_months: int = DEFAULT_COVAR_SPAN_MONTHS,  # EWMA span in months
                         returns_freq: str = 'ME',  # frequency the returns are estimated at
                         rebalancing_freq: str = 'QE',  # frequency the matrices are sampled at
                         demean: bool = True,  # subtract the rolling EWMA mean
                         time_period: Optional[qis.TimePeriod] = None,  # None: the whole sample
                         ) -> Dict[pd.Timestamp, np.ndarray]:
    """point-in-time EWMA factor covariance, one matrix per rebalancing date.

    Delegates to ``qis.estimate_rolling_ewma_covar``, so this package carries one
    covariance convention and it is the stack's. Estimation frequency and
    sampling frequency are separate arguments because the first sets the
    sampling error and the second sets how often the deflator can update.

    At each rebalancing date the estimate uses only returns observed by then, so
    a deflator evaluated at a cash-flow date carries no later information.

    Matrices are returned in **quarterly** units. ``qis`` annualises, so they are
    divided by 4; the deflator then scales by the horizon in quarters.

    Args:
        levels: daily factor levels.
        span_months: EWMA span in months.
        returns_freq: frequency the returns are estimated at.
        rebalancing_freq: frequency the matrices are sampled at.
        demean: subtract the rolling EWMA mean.
        time_period: window to restrict the output to. None uses the whole sample.

    Returns:
        Dict mapping rebalancing date to a quarterly covariance matrix.

    Raises:
        ValueError: if ``span_months`` is not positive.
    """
    if span_months <= 0:
        raise ValueError(f"span_months must be positive, got {span_months!r}")
    covars = qis.estimate_rolling_ewma_covar(prices=levels,
                                             time_period=time_period,
                                             returns_freq=returns_freq,
                                             rebalancing_freq=rebalancing_freq,
                                             span=span_months,
                                             demean=demean,
                                             apply_an_factor=True)
    out: Dict[pd.Timestamp, np.ndarray] = {}
    for date, sigma_annualised in covars.items():
        key = date.tz_localize(None) if getattr(date, 'tz', None) else date
        out[key] = sigma_annualised.values / 4.0  # annual to quarterly units
    return out


def closest_or_default_sigma(sigma_by_quarter: Dict[pd.Timestamp, np.ndarray],
                             quarter_end: pd.Timestamp,
                             default: np.ndarray,  # used when no quarter end precedes the date
                             ) -> np.ndarray:
    """covariance for a quarter end, falling back to the most recent one before it.

    Args:
        sigma_by_quarter: covariance by quarter end.
        quarter_end: quarter end to look up.
        default: used when no estimated quarter end precedes ``quarter_end``.

    Returns:
        The covariance matrix in force at ``quarter_end``.
    """
    qe = quarter_end.tz_localize(None) if getattr(quarter_end, 'tz', None) else quarter_end
    if qe in sigma_by_quarter:
        return sigma_by_quarter[qe]
    preceding = sorted(d for d in sigma_by_quarter if d <= qe)
    if preceding:
        return sigma_by_quarter[preceding[-1]]
    return default


def matf_deflator(cf_dates: List[pd.Timestamp],
                  t0: pd.Timestamp,  # horizon origin, normally the first capital call
                  cum_log_factor: np.ndarray,  # (T, M) cumulative log excess factor returns at quarter ends
                  cum_log_rf: np.ndarray,  # (T,) cumulative log(1 + rf) at quarter ends
                  quarter_ends: pd.DatetimeIndex,
                  beta: np.ndarray,  # (M,) factor loadings
                  sigma_by_quarter: Optional[Dict[pd.Timestamp, np.ndarray]] = None,  # None: use default
                  sigma_default: Optional[np.ndarray] = None,  # quarterly covariance, used before burn-in
                  ) -> np.ndarray:
    """MATF benchmark-portfolio deflator evaluated at a vintage's cash-flow dates.

    Cash-flow dates are mapped to the last quarter end at or before them, which
    is the ``asof`` convention the factor panel is reported on. A date before the
    first quarter end returns NaN rather than being pinned to the first
    quarter — pinning silently fabricates a horizon.

    Args:
        cf_dates: dates to evaluate the deflator at.
        t0: horizon origin, normally the vintage's first capital call.
        cum_log_factor: cumulative log excess factor returns at ``quarter_ends``.
        cum_log_rf: cumulative log(1 + rf) at ``quarter_ends``.
        quarter_ends: quarter-end dates the cumulative arrays are indexed by.
        beta: factor loadings.
        sigma_by_quarter: point-in-time quarterly covariance by quarter end. When
            None, ``sigma_default`` is used at every date.
        sigma_default: quarterly covariance used before the covariance burn-in.

    Returns:
        Deflator value at each date in ``cf_dates``, NaN where the date or the
        origin falls before the first quarter end.

    Raises:
        ValueError: if the shapes do not align, or if both ``sigma_by_quarter``
            and ``sigma_default`` are None.
    """
    if sigma_by_quarter is None and sigma_default is None:
        raise ValueError("pass sigma_by_quarter, sigma_default, or both")
    beta = np.asarray(beta, dtype=float)
    n_quarters, n_factors = np.shape(cum_log_factor)
    if beta.shape != (n_factors,):
        raise ValueError(f"beta must have shape ({n_factors},), got {beta.shape}")
    if len(cum_log_rf) != n_quarters:
        raise ValueError(f"cum_log_rf must have length {n_quarters}, got {len(cum_log_rf)}")
    if len(quarter_ends) != n_quarters:
        raise ValueError(f"quarter_ends must have length {n_quarters}, got {len(quarter_ends)}")

    quarters_ns = np.asarray([pd.Timestamp(q).value for q in quarter_ends], dtype=np.int64)
    cf_ns = np.asarray([pd.Timestamp(t).value for t in cf_dates], dtype=np.int64)
    t0_ns = pd.Timestamp(t0).value

    # 'asof' semantics: the largest quarter end at or before the date.
    idx_t = np.searchsorted(quarters_ns, cf_ns, side='right') - 1
    idx_t0 = int(np.searchsorted(quarters_ns, t0_ns, side='right') - 1)

    out = np.full(len(cf_dates), np.nan, dtype=float)
    if idx_t0 < 0:
        return out  # the origin precedes the panel, so no horizon is defined
    in_panel = idx_t >= 0

    horizon_years = np.asarray([max((pd.Timestamp(t) - pd.Timestamp(t0)).days, 1) / 365.25
                                for t in cf_dates])
    horizon_quarters = horizon_years * 4.0

    for j in range(len(cf_dates)):
        if not in_panel[j]:
            continue
        if sigma_by_quarter is None:
            sigma = sigma_default
        else:
            sigma = closest_or_default_sigma(sigma_by_quarter,
                                             pd.Timestamp(quarter_ends[idx_t[j]]),
                                             sigma_default)
            if sigma is None:
                continue
        h = horizon_quarters[j]
        r_h = cum_log_factor[idx_t[j]] - cum_log_factor[idx_t0]
        rf_h = cum_log_rf[idx_t[j]] - cum_log_rf[idx_t0]
        if np.any(np.isnan(r_h)) or np.isnan(rf_h):
            continue
        log_rb = (rf_h
                  + beta @ r_h
                  + 0.5 * h * (beta @ np.diag(sigma))
                  - 0.5 * h * float(beta @ sigma @ beta))
        out[j] = np.exp(log_rb)
    return out
