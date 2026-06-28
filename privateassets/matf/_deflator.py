"""
Rolling-Sigma multi-factor deflator.

Builds the multi-factor PME deflator
    R^b_h = exp{ rf*h + beta'r_h + (1/2)(beta'diag(Sigma_h) - beta'Sigma_h beta) }
with a time-varying factor covariance Sigma_h. At each quarter-end the
covariance is an EWMA estimate on a trailing window of monthly factor returns
(span 36 months), so the deflator at a cash flow date uses only information
available by that date.

Specification:
  - Monthly factor returns: end-of-month level to log excess return
  - EWMA span: 36 months (about a 3-year half-life)
  - Quarterly covariance: monthly Sigma times 3
  - Burn-in: 36 months before the first cash flow date

Author: A. Sepp / the desk
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from factorlasso import compute_ewm_covar

from privateassets.matf._pipeline import (
    cf_with_terminal_for_vintage,
    vintage_da_fast,
)


def factor_monthly_log_returns(levels_daily: pd.DataFrame) -> pd.DataFrame:
    """End-of-month log returns from daily levels."""
    me = levels_daily.resample('ME').last()
    return np.log(me / me.shift(1)).dropna(how='all')


def rolling_ewma_quarterly_covar(
    levels_daily: pd.DataFrame,
    quarter_ends: pd.DatetimeIndex,
    span_months: int = 36,
    burnin_months: int = 36,
) -> Dict[pd.Timestamp, np.ndarray]:
    """
    Build a dict mapping each quarter-end to an EWMA covariance matrix
    estimated on monthly factor returns with span = span_months.

    The returned Σ are SCALED TO A QUARTERLY HORIZON: monthly Σ × 3
    (the standard variance-time-scaling under iid). This is consistent
    with how the deflator uses Σ_h (h = 1 quarter for adjacent cash flow
    dates within the same quarter).
    """
    monthly_log = factor_monthly_log_returns(levels_daily)

    sigmas = {}
    for q in quarter_ends:
        # Monthly returns up to and including this quarter end
        m_end = q.to_period('M').to_timestamp(how='end').normalize()
        window = monthly_log.loc[:m_end].dropna(how='any')
        if len(window) < burnin_months:
            continue
        # EWMA covariance on monthly excess log returns, demean within the window
        Xv = window.values
        Xv_dm = Xv - Xv.mean(axis=0)
        sigma_monthly = compute_ewm_covar(a=Xv_dm, span=span_months)
        sigma_quarterly = sigma_monthly * 3.0  # variance-time scaling
        sigmas[q] = sigma_quarterly
    return sigmas


def matf_deflator_rolling(
    cf_dates: list,
    t0: pd.Timestamp,
    cum_log_factor_q: np.ndarray,    # (T, M) cumulative log factor returns at QE
    cum_log_rf_q: np.ndarray,        # (T,) cumulative log(1+rf) at QE
    quarters_q: pd.DatetimeIndex,
    quarters_ns: np.ndarray,
    beta_vec: np.ndarray,
    sigma_h_dict: Dict[pd.Timestamp, np.ndarray],
    sigma_h_default: np.ndarray,
) -> np.ndarray:
    """
    Deflator with quarter-specific Σ_h pulled from sigma_h_dict.
    For each cash flow date d, find the most recent quarter-end ≤ d,
    look up Σ_h for that quarter (or use sigma_h_default if missing
    due to burn-in), then evaluate Eq 3.
    """
    cf_ns = np.array([pd.Timestamp(t).value for t in cf_dates], dtype=np.int64)
    t0_ns = pd.Timestamp(t0).value
    n_q = len(quarters_ns)

    idx_t = np.clip(np.searchsorted(quarters_ns, cf_ns, side='right') - 1,
                    0, n_q - 1)
    idx_t0 = int(np.clip(np.searchsorted(quarters_ns, t0_ns, side='right') - 1,
                         0, n_q - 1))

    rh = cum_log_factor_q[idx_t] - cum_log_factor_q[idx_t0]   # (n_cf, M)
    rfh = cum_log_rf_q[idx_t] - cum_log_rf_q[idx_t0]           # (n_cf,)

    h_years = np.array([max((pd.Timestamp(t) - pd.Timestamp(t0)).days, 1) / 365.25
                        for t in cf_dates])
    h_q = h_years * 4.0

    out = np.empty(len(cf_dates), dtype=float)
    for j, t in enumerate(cf_dates):
        q_end = quarters_q[idx_t[j]]
        sigma = sigma_h_dict.get(q_end, sigma_h_default)
        diag_h = np.diag(sigma)
        quad = float(beta_vec @ sigma @ beta_vec)
        rb_j = (rfh[j]
                + rh[j] @ beta_vec
                + 0.5 * h_q[j] * (beta_vec @ diag_h)
                - 0.5 * h_q[j] * quad)
        out[j] = np.exp(rb_j)
    return out


def per_vintage_da_with_rolling_covar(
    cf: pd.DataFrame,
    navs: pd.DataFrame,
    asof: pd.Timestamp,
    cum_log_factor_q: np.ndarray,
    cum_log_rf_q: np.ndarray,
    quarters_q: pd.DatetimeIndex,
    quarters_ns: np.ndarray,
    beta_arr: np.ndarray,
    sigma_h_dict: Dict[pd.Timestamp, np.ndarray],
    sigma_h_default: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for v in sorted(cf['vintage_label'].unique()):
        cf_g = cf[cf['vintage_label'] == v]
        nav_g = navs[navs['vintage_label'] == v]
        cf_v, rvpi_nav, _, dates = cf_with_terminal_for_vintage(cf_g, nav_g, asof)
        defl = matf_deflator_rolling(
            dates, dates[0], cum_log_factor_q, cum_log_rf_q,
            quarters_q, quarters_ns, beta_arr, sigma_h_dict, sigma_h_default)
        da = vintage_da_fast(cf_v, rvpi_nav, dates, defl)
        contrib = -cf_v.loc[cf_v['amount'] < 0, 'amount'].sum()
        distrib = cf_v.loc[cf_v['amount'] > 0, 'amount'].sum()
        dpi = distrib / contrib if contrib > 0 else float('nan')
        rows.append({'vintage_label': v, 'contribution': contrib, 'DPI': dpi,
                     'DA': da})
    return pd.DataFrame(rows)


def per_vintage_da_with_constant_covar(
    cf: pd.DataFrame,
    navs: pd.DataFrame,
    asof: pd.Timestamp,
    cum_log_factor_q: np.ndarray,
    cum_log_rf_q: np.ndarray,
    quarters_q: pd.DatetimeIndex,
    quarters_ns: np.ndarray,
    beta_arr: np.ndarray,
    sigma_h_const: np.ndarray,
) -> pd.DataFrame:
    rows = []
    diag_h_q = np.diag(sigma_h_const)
    quad = float(beta_arr @ sigma_h_const @ beta_arr)
    for v in sorted(cf['vintage_label'].unique()):
        cf_g = cf[cf['vintage_label'] == v]
        nav_g = navs[navs['vintage_label'] == v]
        cf_v, rvpi_nav, _, dates = cf_with_terminal_for_vintage(cf_g, nav_g, asof)
        cf_ns = np.array([pd.Timestamp(t).value for t in dates], dtype=np.int64)
        n_q = len(quarters_ns)
        idx_t = np.clip(np.searchsorted(quarters_ns, cf_ns, side='right') - 1,
                        0, n_q - 1)
        idx_t0 = idx_t[0]
        rh = cum_log_factor_q[idx_t] - cum_log_factor_q[idx_t0]
        rfh = cum_log_rf_q[idx_t] - cum_log_rf_q[idx_t0]
        h_years = np.array([
            max((pd.Timestamp(t) - pd.Timestamp(dates[0])).days, 1) / 365.25
            for t in dates])
        h_q = h_years * 4.0
        rb = (rfh + rh @ beta_arr
              + 0.5 * h_q * (beta_arr @ diag_h_q)
              - 0.5 * h_q * quad)
        defl = np.exp(rb)
        da = vintage_da_fast(cf_v, rvpi_nav, dates, defl)
        contrib = -cf_v.loc[cf_v['amount'] < 0, 'amount'].sum()
        distrib = cf_v.loc[cf_v['amount'] > 0, 'amount'].sum()
        dpi = distrib / contrib if contrib > 0 else float('nan')
        rows.append({'vintage_label': v, 'contribution': contrib, 'DPI': dpi,
                     'DA': da})
    return pd.DataFrame(rows)
