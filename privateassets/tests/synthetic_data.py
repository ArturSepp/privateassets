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


def make_factor_driven_panel(beta: np.ndarray = None,  # true factor loadings
                             annual_alpha: float = 0.04,  # true annualised alpha
                             theta: float = 0.30,  # true appraisal-smoothing coefficient
                             n_funds: int = 4,
                             n_quarters: int = 44,  # about eleven years per fund
                             start: str = '2006-03-31',
                             noise: float = 0.01,
                             seed: int = SEED,
                             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """a panel whose funds genuinely earn a known alpha over a known factor basket.

    Built so the whole estimator chain can be validated end to end: the true
    return is ``beta' f + alpha``, it is then smoothed by an AR(1) of coefficient
    ``theta`` the way an appraisal process smooths it, and the reported NAVs are
    the smoothed path. Recovering alpha therefore requires the unsmoothing, the
    loading fit and the deflator all to be right together.

    Returns:
        Tuple of (cash flows, NAVs, factor levels, quarterly risk-free yield).
    """
    if beta is None:
        beta = np.array([0.70, 0.00, 0.50, 0.00])
    rng = np.random.default_rng(seed)

    levels = make_factor_levels(start='2000-12-31', end='2026-12-31', seed=seed)
    quarter_ends = pd.DatetimeIndex(levels.resample('QE').last().index)
    factor_q = np.log(levels.resample('QE').last()).diff().fillna(0.0)
    rf = make_rf_quarterly(quarter_ends, rate=0.02)

    cf_rows, nav_rows = [], []
    for i in range(n_funds):
        label = f'Fund {i + 1}'
        first = pd.Timestamp(start) + pd.DateOffset(years=2 * i)
        periods = pd.date_range(first, periods=n_quarters, freq='QE')
        periods = periods[periods.isin(quarter_ends)]
        commitment = 100.0 * (1.0 + 0.3 * i)

        # The fund earns exactly what the MATF benchmark portfolio earns, times
        # exp(alpha). Factor inputs are excess log returns, so the risk-free leg
        # is added back and the Jensen terms convert the levered log basket into
        # the arithmetic return a holder would realise. Built this way, the
        # estimator must return `annual_alpha` and nothing else.
        sigma_q = np.cov(factor_q.values, rowvar=False)
        jensen = 0.5 * (beta @ np.diag(sigma_q) - float(beta @ sigma_q @ beta))
        log_rf = np.log1p(rf.reindex(periods).values)
        log_gross = (log_rf
                     + factor_q.reindex(periods).values @ beta
                     + jensen
                     + annual_alpha / 4.0
                     + rng.normal(0.0, noise, len(periods)))
        true_returns = np.expm1(log_gross)
        # Appraisal smoothing: r_reported_t = (1-theta) r_true_t + theta r_reported_{t-1}
        reported = np.zeros(len(periods))
        for t in range(len(periods)):
            previous = reported[t - 1] if t > 0 else 0.0
            reported[t] = (1.0 - theta) * true_returns[t] + theta * previous

        nav = 0.0
        n_calls, n_dists = 6, 10
        for t, date in enumerate(periods):
            call = commitment / n_calls if t < n_calls else 0.0
            dist = 0.0
            if t >= len(periods) - n_dists:
                dist = nav * (1.0 / (len(periods) - t))
            nav = nav * (1.0 + reported[t]) + call - dist
            if call > 0:
                cf_rows.append({'fund': label, 'vintage_label': label, 'date': date,
                                'amount': -call, 'kind': 'contribution'})
            if dist > 0:
                cf_rows.append({'fund': label, 'vintage_label': label, 'date': date,
                                'amount': dist, 'kind': 'distribution'})
            nav_rows.append({'fund': label, 'vintage_label': label, 'date': date,
                             'nav': float(max(nav, 1e-6))})

    cf = pd.DataFrame(cf_rows).sort_values(['fund', 'date']).reset_index(drop=True)
    navs = pd.DataFrame(nav_rows).sort_values(['fund', 'date']).reset_index(drop=True)
    return cf, navs, levels, rf


def make_collinear_factor_panel(rho: float = 0.90,  # correlation between Equity and Credit
                                beta: np.ndarray = None,  # true loadings, in FACTOR_NAMES order
                                n_quarters: int = 60,  # fifteen years, a realistic fund life
                                residual_vol: float = 0.03,  # idiosyncratic return volatility
                                seed: int = SEED,
                                ) -> Tuple[pd.Series, pd.DataFrame, np.ndarray]:
    """an asset return driven by known loadings on a deliberately collinear factor panel.

    The other generators here draw factors independently, so they produce a
    near-orthogonal panel (maximum absolute correlation about 0.19) and cannot
    exercise the regime the shrinkage estimator exists for. This one sets the
    equity-credit correlation directly, which is the pair that matters in a
    private-credit universe.

    Args:
        rho: correlation between the Equity and Credit factors, in [-1, 1].
        beta: true loadings in ``FACTOR_NAMES`` order. None uses a
            credit-and-equity profile.
        n_quarters: length of the panel.
        residual_vol: standard deviation of the return not explained by factors.
        seed: seed for the draw.

    Returns:
        Tuple of (asset returns, factor returns, true beta). Both series are
        quarterly and share a quarter-end index.

    Raises:
        ValueError: if ``rho`` is outside [-1, 1], or if ``beta`` does not have
            one entry per factor.
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [-1, 1], got {rho!r}")
    if beta is None:
        beta = np.array([0.70, 0.00, 0.50, 0.00])
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (len(FACTOR_NAMES),):
        raise ValueError(f"beta must have one entry per factor {FACTOR_NAMES}, got {beta.shape}")

    rng = np.random.default_rng(seed)
    equity = rng.normal(0.0, 0.08, n_quarters)
    independent = rng.normal(0.0, 0.08, n_quarters)
    credit = rho * equity + np.sqrt(1.0 - rho ** 2) * independent
    index = pd.date_range('2010-03-31', periods=n_quarters, freq='QE')
    factor_returns = pd.DataFrame({'Equity': equity,
                                   'Rates': rng.normal(0.0, 0.02, n_quarters),
                                   'Credit': credit,
                                   'Commodities': rng.normal(0.0, 0.10, n_quarters)},
                                  index=index)[FACTOR_NAMES]
    asset_returns = pd.Series(factor_returns.values @ beta
                              + rng.normal(0.0, residual_vol, n_quarters),
                              index=index, name='asset')
    return asset_returns, factor_returns, beta
