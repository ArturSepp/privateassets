"""
privateassets.matf._benchmarks — the single-factor deflators MATF displaces.

Two incumbents, kept here so the multi-factor measure can be compared against
them on the same cash flows rather than described as better in prose.

The KN24 deflator prices a fund against a levered position in one market index:
hold beta in the market and the rest in cash. The KN16 stochastic discount
factor prices it against a power-utility investor's marginal rate of
substitution over that same index. Both charge the fund for exactly one
exposure, which is what the MATF deflator generalises.

Both take the equity factor as an **excess** log return, matching the rest of
the package, and add the risk-free leg back where the economics needs a total
return. Getting that wrong understates the benchmark by roughly the risk-free
rate over the horizon, which overstates the fund's alpha against it.
"""

# packages
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple
# qis / project
from privateassets.matf._deflator import horizon_indices

QUARTERS_PER_YEAR = 4
# Below this annualised variance the series is not a market: gamma = mu / sigma^2
# explodes, and a constant series returns floating-point noise rather than zero.
MIN_ANNUAL_VARIANCE = 1e-12


def kn24_benchmark_deflator(cf_dates: List[pd.Timestamp],
                            t0: pd.Timestamp,  # horizon origin, normally the first capital call
                            cum_log_equity_excess: np.ndarray,  # (T,) cumulative log EXCESS equity return
                            cum_log_rf: np.ndarray,  # (T,) cumulative log(1 + rf) at quarter ends
                            quarter_ends: pd.DatetimeIndex,
                            beta: float,  # market exposure of the benchmark portfolio
                            sigma2_annual: float,  # annualised variance of the market log return
                            ) -> np.ndarray:
    """single-factor benchmark-portfolio deflator (Korteweg-Nagel style).

    Prices the fund against a portfolio holding ``beta`` in the market and
    ``1 - beta`` in cash, rebalanced continuously:

        R^b_h = exp{ rf_h + beta r^ex_h - (1/2) beta (beta - 1) sigma^2 h }

    where ``r^ex`` is the market log return in excess of the risk-free rate. The
    risk-free rate enters once, through ``rf_h``. Subtracting it a second time
    inside the bracket, which is easy to do when the factor panel is already
    excess, understates the benchmark by ``(1 - beta) rf_h``.

    At ``beta = 0`` the deflator is the risk-free accrual; at ``beta = 1`` it is
    the market's total return. Both are pinned by tests.

    Args:
        cf_dates: dates to evaluate the deflator at.
        t0: horizon origin.
        cum_log_equity_excess: cumulative log excess equity return at
            ``quarter_ends``.
        cum_log_rf: cumulative log(1 + rf) at ``quarter_ends``.
        quarter_ends: quarter-end dates the cumulative arrays are indexed by.
        beta: market exposure of the benchmark portfolio.
        sigma2_annual: annualised variance of the market log return.

    Returns:
        Deflator value at each date, NaN where the date or the origin falls
        before the panel.

    Raises:
        ValueError: if the cumulative arrays do not align with ``quarter_ends``,
            or if ``sigma2_annual`` is negative.
    """
    n_quarters = len(quarter_ends)
    if len(cum_log_equity_excess) != n_quarters:
        raise ValueError(f"cum_log_equity_excess must have length {n_quarters}, "
                         f"got {len(cum_log_equity_excess)}")
    if len(cum_log_rf) != n_quarters:
        raise ValueError(f"cum_log_rf must have length {n_quarters}, got {len(cum_log_rf)}")
    if sigma2_annual < 0:
        raise ValueError(f"sigma2_annual must be non-negative, got {sigma2_annual!r}")

    idx_t, idx_t0, horizon_years, in_panel = horizon_indices(cf_dates, t0, quarter_ends)
    out = np.full(len(cf_dates), np.nan, dtype=float)
    if idx_t0 < 0:
        return out

    safe = np.where(in_panel, idx_t, 0)
    r_excess = cum_log_equity_excess[safe] - cum_log_equity_excess[idx_t0]
    rf_h = cum_log_rf[safe] - cum_log_rf[idx_t0]

    log_rb = rf_h + beta * r_excess - 0.5 * beta * (beta - 1.0) * sigma2_annual * horizon_years
    out[in_panel] = np.exp(log_rb[in_panel])
    return out


def kn16_sdf_params(equity_excess_log_returns: pd.Series,
                    rf_simple: pd.Series,  # simple yield per period, same index
                    periods_per_year: int = QUARTERS_PER_YEAR,
                    ) -> Tuple[float, float, float]:
    """calibrate the power-utility SDF to the market's own moments.

    The stochastic discount factor is ``M_h = exp(delta h - gamma r^m_h)``, with
    ``r^m`` the market's **total** log return. Requiring it to price the
    risk-free asset and the market itself pins both parameters:

        gamma = mu / sigma^2
        delta = -rf - (1/2) gamma^2 sigma^2 + gamma (rf + mu - (1/2) sigma^2)

    where ``mu`` is the arithmetic equity premium. Because the input series is
    already in excess terms, ``mu = log E[exp(r^ex)] = mean(r^ex) + sigma^2 / 2``
    directly. Subtracting the risk-free rate again at this step understates the
    premium by exactly ``rf`` and leaves the SDF mispricing the market.

    Args:
        equity_excess_log_returns: market log returns in excess of the
            risk-free rate, one observation per period.
        rf_simple: simple risk-free yield per period, on the same index.
        periods_per_year: periods per year, used to annualise.

    Returns:
        Tuple of ``(delta, gamma, sigma2_annual)``.

    Raises:
        ValueError: if fewer than two observations survive, or if the realised
            variance does not exceed ``MIN_ANNUAL_VARIANCE``.
    """
    returns = pd.Series(equity_excess_log_returns).dropna()
    if len(returns) < 2:
        raise ValueError(f"need at least 2 observations, got {len(returns)}")

    sigma2_annual = float(returns.var(ddof=1)) * periods_per_year
    if not sigma2_annual > MIN_ANNUAL_VARIANCE:
        raise ValueError(f"realised variance must exceed {MIN_ANNUAL_VARIANCE}, got "
                         f"{sigma2_annual!r}; gamma = mu / sigma^2 is not identified below it")

    rf_annual = float(pd.Series(rf_simple).dropna().mean()) * periods_per_year
    # The series is excess, so its lognormal mean IS the premium. No second subtraction.
    mu = float(returns.mean()) * periods_per_year + 0.5 * sigma2_annual

    gamma = mu / sigma2_annual
    delta = (-rf_annual
             - 0.5 * gamma ** 2 * sigma2_annual
             + gamma * (rf_annual + mu - 0.5 * sigma2_annual))
    return delta, gamma, sigma2_annual


def kn16_gpme_deflator(cf_dates: List[pd.Timestamp],
                       t0: pd.Timestamp,  # horizon origin
                       cum_log_equity_excess: np.ndarray,  # (T,) cumulative log EXCESS equity return
                       cum_log_rf: np.ndarray,  # (T,) cumulative log(1 + rf) at quarter ends
                       quarter_ends: pd.DatetimeIndex,
                       delta: float,  # subjective discount rate from kn16_sdf_params
                       gamma: float,  # risk aversion from kn16_sdf_params
                       ) -> np.ndarray:
    """generalised PME deflator: the reciprocal of the power-utility SDF.

    Returns ``1 / M_h`` so it substitutes for any other deflator in
    :func:`~privateassets.matf.vintage_direct_alpha`.

    The SDF is defined on the market's **total** log return, so the risk-free
    leg is added back to the excess factor here. Evaluating it on the excess
    return instead leaves the kernel inconsistent with the ``delta`` that was
    calibrated against a total return, and the market no longer prices to one.

    Args:
        cf_dates: dates to evaluate the deflator at.
        t0: horizon origin.
        cum_log_equity_excess: cumulative log excess equity return.
        cum_log_rf: cumulative log(1 + rf) at ``quarter_ends``.
        quarter_ends: quarter-end dates the cumulative arrays are indexed by.
        delta: subjective discount rate, from :func:`kn16_sdf_params`.
        gamma: relative risk aversion, from :func:`kn16_sdf_params`.

    Returns:
        Deflator value at each date, NaN where the date or the origin falls
        before the panel.

    Raises:
        ValueError: if the cumulative arrays do not align with ``quarter_ends``.
    """
    n_quarters = len(quarter_ends)
    if len(cum_log_equity_excess) != n_quarters:
        raise ValueError(f"cum_log_equity_excess must have length {n_quarters}, "
                         f"got {len(cum_log_equity_excess)}")
    if len(cum_log_rf) != n_quarters:
        raise ValueError(f"cum_log_rf must have length {n_quarters}, got {len(cum_log_rf)}")

    idx_t, idx_t0, horizon_years, in_panel = horizon_indices(cf_dates, t0, quarter_ends)
    out = np.full(len(cf_dates), np.nan, dtype=float)
    if idx_t0 < 0:
        return out

    safe = np.where(in_panel, idx_t, 0)
    r_excess = cum_log_equity_excess[safe] - cum_log_equity_excess[idx_t0]
    rf_h = cum_log_rf[safe] - cum_log_rf[idx_t0]
    r_total = r_excess + rf_h  # the SDF is defined on the total market return

    log_m = delta * horizon_years - gamma * r_total
    out[in_panel] = np.exp(-log_m[in_panel])
    return out
