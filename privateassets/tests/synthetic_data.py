"""
Seeded synthetic private-asset panels for the test suite.

Draws a panel carrying the defects real LP panels carry: irregular cash-flow
dates, a J-curve, unrealised residual NAVs, funds of different lengths, and a
factor panel that starts after the first fund does. Imports numpy and pandas
only, never the library under test, so a golden pinned to it stays valid.
"""

# packages
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

SEED = 20260726
FACTOR_NAMES = ['Equity', 'Rates', 'Credit', 'Commodities']


def make_factor_levels(start: str = '1999-12-31',
                       end: str = '2024-12-31',
                       seed: int = SEED,
                       ) -> pd.DataFrame:
    """daily factor level panel starting at 100, drawn from a seeded GBM."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n_days, n_factors = len(dates), len(FACTOR_NAMES)
    vols = np.array([0.15, 0.05, 0.08, 0.18]) / np.sqrt(260.0)
    drifts = np.array([0.05, 0.01, 0.03, 0.02]) / 260.0
    shocks = rng.normal(size=(n_days, n_factors)) * vols + drifts
    levels = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    return pd.DataFrame(levels, index=dates, columns=FACTOR_NAMES)


def make_rf_quarterly(quarter_ends: pd.DatetimeIndex,
                      rate: float = 0.02,  # annualised simple rate
                      ) -> pd.Series:
    """flat quarterly simple risk-free yield on a quarter-end grid."""
    return pd.Series(rate / 4.0, index=quarter_ends, name='rf')


def make_cash_flows(n_funds: int = 5,
                    seed: int = SEED,
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """tidy cash-flow and NAV frames for a small panel of funds.

    Each fund draws capital over its first three years, distributes over the
    following seven, and carries a residual NAV. The youngest fund is
    deliberately unrealised so the DPI maturity gate has something to exclude.

    Returns:
        Tuple of (cash flows, NAVs) in the tidy long format the measures consume.
    """
    rng = np.random.default_rng(seed)
    cf_rows: List[Dict] = []
    nav_rows: List[Dict] = []

    for i in range(n_funds):
        fund = f'Synthetic Fund {i + 1}, L.P.'
        label = f'Synthetic Fund {i + 1}'
        vintage_start = pd.Timestamp('2001-06-30') + pd.DateOffset(years=3 * i)
        commitment = 100.0 * (1.0 + 0.2 * i)

        call_dates = pd.date_range(vintage_start, periods=8, freq='QE')
        call_weights = rng.dirichlet(np.ones(8) * 2.0)
        for d, w in zip(call_dates, call_weights, strict=True):
            cf_rows.append({'fund': fund, 'vintage_label': label, 'date': d,
                            'amount': -commitment * w, 'kind': 'contribution'})

        n_dists = 12 if i < n_funds - 1 else 3  # the youngest fund stays unrealised
        dist_dates = pd.date_range(vintage_start + pd.DateOffset(years=3),
                                   periods=n_dists, freq='2QE')
        multiple = 1.6 if i < n_funds - 1 else 0.3
        dist_weights = rng.dirichlet(np.ones(n_dists) * 2.0)
        for d, w in zip(dist_dates, dist_weights, strict=True):
            cf_rows.append({'fund': fund, 'vintage_label': label, 'date': d,
                            'amount': commitment * multiple * w, 'kind': 'distribution'})

        nav_dates = pd.date_range(vintage_start, dist_dates[-1], freq='QE')
        peak = commitment * 0.9
        for k, d in enumerate(nav_dates):
            shape = np.sin(np.pi * (k + 1) / (len(nav_dates) + 1))
            nav_rows.append({'fund': fund, 'vintage_label': label, 'date': d,
                             'nav': float(peak * shape * (1.0 + 0.05 * rng.standard_normal()))})

    cf = pd.DataFrame(cf_rows).sort_values(['fund', 'date']).reset_index(drop=True)
    navs = pd.DataFrame(nav_rows).sort_values(['fund', 'date']).reset_index(drop=True)
    return cf, navs


def make_ar1_panel(n_series: int = 8,
                   n_obs: int = 60,
                   theta: float = 0.25,  # true AR(1) smoothing coefficient
                   seed: int = SEED,
                   burnin: int = 200,  # draws discarded so the series starts stationary
                   demean: bool = True,  # subtract the sample mean, as production does
                   ) -> List[Tuple[str, np.ndarray]]:
    """AR(1) series with a common theta and per-series residual variance.

    The burn-in matters: starting the recursion at zero leaves a transient in
    which the series is less persistent than its own parameter, which reads as
    estimator bias when it is a property of the draw.
    """
    rng = np.random.default_rng(seed)
    out: List[Tuple[str, np.ndarray]] = []
    for i in range(n_series):
        sigma = 0.02 * (1.0 + 0.5 * i / n_series)
        r = np.zeros(n_obs + burnin)
        for t in range(1, n_obs + burnin):
            r[t] = theta * r[t - 1] + rng.normal(0.0, sigma)
        series = r[burnin:]
        out.append((f'series_{i}', series - series.mean() if demean else series))
    return out
