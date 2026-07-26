"""
privateassets.matf._panel_mle — panel MLE for a common AR(1) smoothing coefficient.

Appraisal NAVs are reported with a lag and are anchored to the previous mark, so
the reported return series is a moving average of the true one. Getmansky-Lo-
Makarov invert that with an AR coefficient theta. Estimating theta on a single
fund is noisy, because a fund contributes only a few dozen quarterly
observations.

This estimates one theta across all funds while letting each keep its own
residual variance:

    r_{i,t} = theta r_{i,t-1} + e_{i,t},    e_{i,t} ~ N(0, sigma_i^2)

Profiling out sigma_i^2 at its MLE, sigma_i^2(theta) = mean_t (r_{i,t} - theta
r_{i,t-1})^2, reduces the problem to one dimension:

    L_p(theta) = -(1/2) Sum_i n_i log sigma_i^2(theta) + const

which is then maximised over theta in (-1, 1). Pooling the funds is what buys
the precision; letting sigma_i^2 differ is what stops a single volatile fund
from setting theta for everyone.

The estimated theta feeds the unsmoothing step, which lives in ``qis``:
``qis.unsmooth_returns_glm(returns, ar_order=1, theta=theta_hat)``.
"""

# packages
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Dict, Optional, Sequence, Tuple, Union

MIN_OBS_FOR_THETA = 3  # a series shorter than this carries no usable lag pair
MIN_OBS_PER_VINTAGE_MLE = 5  # a per-series estimate below this is too noisy to report

SeriesList = Sequence[Union[np.ndarray, Tuple[str, np.ndarray]]]


def _as_label_array(item: Union[np.ndarray, Tuple[str, np.ndarray]],
                    position: int,
                    ) -> Tuple[str, np.ndarray]:
    """normalise one entry of a series list to a (label, array) pair."""
    if isinstance(item, tuple):
        label, values = item
        return str(label), np.asarray(values, dtype=float)
    return f'series_{position}', np.asarray(item, dtype=float)


def panel_ar1_neg_log_likelihood(theta: float,
                                 series_list: SeriesList,  # demeaned return series, one per fund
                                 ) -> float:
    """profiled negative log-likelihood of the common-theta panel AR(1).

    Args:
        theta: AR(1) coefficient.
        series_list: demeaned return series, one per fund, either as arrays or as
            ``(label, array)`` pairs. Series shorter than ``MIN_OBS_FOR_THETA``
            are skipped.

    Returns:
        ``-L_p(theta)``, or infinity where a profiled variance is not positive.
    """
    nll = 0.0
    for position, item in enumerate(series_list):
        _, r = _as_label_array(item, position)
        if len(r) < MIN_OBS_FOR_THETA:
            continue
        residuals = r[1:] - theta * r[:-1]
        sigma2_hat = float(np.mean(residuals ** 2))
        if sigma2_hat <= 0:
            return np.inf
        nll += 0.5 * len(residuals) * np.log(sigma2_hat)
    return nll


def fisher_info_panel_ar1(theta: float,
                          series_list: SeriesList,  # demeaned return series, one per fund
                          ) -> float:
    """Fisher information for theta, summed across the panel.

    With sigma_i^2 known the per-observation information is
    ``Var(r_{i,t-1}) / sigma_i^2``. Profiling sigma_i^2 at its MLE gives
    ``n_i Var(r_{i,t-1}) / Sum_t (r_{i,t} - theta r_{i,t-1})^2``.

    Args:
        theta: AR(1) coefficient to evaluate at.
        series_list: demeaned return series, one per fund.

    Returns:
        Total Fisher information. Zero when no series qualifies.
    """
    fisher = 0.0
    for position, item in enumerate(series_list):
        _, r = _as_label_array(item, position)
        if len(r) < MIN_OBS_FOR_THETA:
            continue
        residuals = r[1:] - theta * r[:-1]
        sigma2_hat = float(np.mean(residuals ** 2))
        if sigma2_hat <= 0:
            continue
        fisher += len(residuals) * float(np.var(r[:-1])) / sigma2_hat
    return fisher


def fit_panel_ar1(series_list: SeriesList,  # demeaned return series, one per fund
                  theta_bounds: Tuple[float, float] = (-0.95, 0.95),  # search interval for theta
                  ) -> Dict[str, Union[float, int, Dict[str, Optional[float]]]]:
    """estimate a common AR(1) smoothing coefficient across a panel of funds.

    Args:
        series_list: demeaned return series, one per fund, either as arrays or as
            ``(label, array)`` pairs.
        theta_bounds: search interval for theta, inside (-1, 1).

    Returns:
        Dict carrying ``theta_hat``, its asymptotic standard error ``se``, the
        profiled ``neg_log_lik``, ``fisher_info``, the per-series OLS estimates
        in ``per_series_theta`` (None where the series is too short), the total
        lag-pair count ``n_total``, and the qualifying series count ``n_series``.

    Raises:
        ValueError: if ``series_list`` is empty or ``theta_bounds`` is not a
            valid interval inside (-1, 1).
    """
    if len(series_list) == 0:
        raise ValueError("series_list is empty")
    lo, hi = theta_bounds
    if not (-1.0 < lo < hi < 1.0):
        raise ValueError(f"theta_bounds must satisfy -1 < lo < hi < 1, got {theta_bounds!r}")

    result = minimize_scalar(panel_ar1_neg_log_likelihood,
                             args=(series_list,),
                             bounds=theta_bounds,
                             method='bounded',
                             options={'xatol': 1e-6})
    theta_hat = float(result.x)
    fisher = fisher_info_panel_ar1(theta_hat, series_list)
    se = float(1.0 / np.sqrt(fisher)) if fisher > 0 else float('nan')

    per_series: Dict[str, Optional[float]] = {}
    n_total = 0
    n_series = 0
    for position, item in enumerate(series_list):
        label, r = _as_label_array(item, position)
        n_total += max(0, len(r) - 1)
        if len(r) < MIN_OBS_PER_VINTAGE_MLE:
            per_series[label] = None
            continue
        y, x = r[1:], r[:-1]
        if np.var(x) < 1e-10:
            per_series[label] = None
            continue
        per_series[label] = float(np.cov(y, x)[0, 1] / np.var(x))
        n_series += 1

    return {
        'theta_hat': theta_hat,
        'se': se,
        'neg_log_lik': float(result.fun),
        'fisher_info': fisher,
        'per_series_theta': per_series,
        'n_total': n_total,
        'n_series': n_series,
    }
