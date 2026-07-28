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

**The estimate is biased down, and the bias is correctable.** Demeaning a short
AR(1) biases its coefficient by about ``-(1 + 3 theta) / n``. At the panel lengths
private-asset funds provide, 40 to 80 quarters, that is several percentage
points, and it flows into the volatility uplift ``1 / (1 - theta)``, understating
unsmoothed risk in the direction that flatters the asset class. Pass
``bias_correction`` to remove it.

One bias it does **not** remove: reported returns are measured with error, worst
while capital is still being called and the modified Dietz denominator is
dominated by the call itself. Measurement error in a regressor attenuates the
coefficient towards zero, and no small-sample correction recovers it, because
simulating from the fitted model reproduces the fitted persistence rather than
the true one. Correcting the demeaning bias therefore moves the estimate towards
the truth without reaching it.
"""

# packages
from __future__ import annotations
import numpy as np
from enum import Enum
from scipy.optimize import minimize_scalar
from typing import Dict, List, Optional, Sequence, Tuple, Union

MIN_OBS_FOR_THETA = 3  # a series shorter than this carries no usable lag pair
MIN_OBS_PER_VINTAGE_MLE = 5  # a per-series estimate below this is too noisy to report
DEFAULT_BIAS_DRAWS = 500  # simulated panels for the bootstrap correction
SIMULATION_BURNIN = 200  # draws discarded so a simulated series starts stationary


class BiasCorrection(str, Enum):
    """How to correct the small-sample bias in the estimated coefficient."""

    NONE = 'none'  # report the raw profile MLE
    KENDALL = 'kendall'  # analytic first-order, from the demeaning bias
    BOOTSTRAP = 'bootstrap'  # parametric: simulate from the fit, measure the bias, remove it

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


def kendall_corrected_theta(theta_hat: float,
                            mean_series_length: float,  # lag pairs per series
                            ) -> float:
    """invert the first-order demeaning bias of an AR(1) coefficient.

    Demeaning induces ``E[theta_hat] ~ theta - (1 + 3 theta) / n``. Solving that
    for the true coefficient gives ``(theta_hat + 1/n) / (1 - 3/n)``.

    For a panel the estimator is a precision-weighted average of the per-series
    estimates, so the relevant ``n`` is the information-weighted harmonic mean of
    the series lengths, which is the total lag-pair count divided by the number
    of series.

    Args:
        theta_hat: the raw estimate.
        mean_series_length: lag pairs per series.

    Returns:
        The corrected coefficient, clipped into (-1, 1).

    Raises:
        ValueError: if ``mean_series_length`` is not greater than 3, below which
            the correction is not defined.
    """
    if not mean_series_length > 3.0:
        raise ValueError(f"mean_series_length must exceed 3 for the correction to be "
                         f"defined, got {mean_series_length!r}")
    corrected = (theta_hat + 1.0 / mean_series_length) / (1.0 - 3.0 / mean_series_length)
    return float(np.clip(corrected, -0.99, 0.99))


def simulate_panel_ar1(theta: float,
                       sigmas: Sequence[float],  # residual sd per series
                       lengths: Sequence[int],  # observations per series
                       rng: np.random.Generator,
                       demean: bool = True,  # match how the real series are prepared
                       ) -> List[Tuple[str, np.ndarray]]:
    """draw one synthetic panel from a fitted common-theta AR(1).

    Each series starts from its stationary distribution rather than from zero: a
    zero start leaves a transient in which the series is less persistent than its
    own parameter, which would be measured as bias and then wrongly removed.

    Args:
        theta: common AR(1) coefficient.
        sigmas: residual standard deviation per series.
        lengths: observations per series.
        rng: seeded generator.
        demean: subtract each series' sample mean, as production does.

    Returns:
        A list of ``(label, array)`` pairs shaped like the input panel.
    """
    panel: List[Tuple[str, np.ndarray]] = []
    for i, (sigma, length) in enumerate(zip(sigmas, lengths, strict=True)):
        draws = rng.normal(0.0, sigma, length + SIMULATION_BURNIN)
        series = np.zeros(length + SIMULATION_BURNIN)
        for t in range(1, len(series)):
            series[t] = theta * series[t - 1] + draws[t]
        values = series[SIMULATION_BURNIN:]
        panel.append((f'sim_{i}', values - values.mean() if demean else values))
    return panel


def bootstrap_corrected_theta(theta_hat: float,
                              sigmas: Sequence[float],  # residual sd per series
                              lengths: Sequence[int],  # observations per series
                              num_draws: int = DEFAULT_BIAS_DRAWS,
                              seed: int = 1,
                              ) -> Tuple[float, float]:
    """remove the bias by measuring it on panels simulated from the fit.

    Simulate panels of the same shape from the fitted coefficient, re-estimate on
    each, and take the mean shortfall as the bias. Subtracting it makes no
    assumption about the functional form of the bias, so it handles the
    heterogeneous series lengths a real panel has, which the analytic formula
    only approximates.

    Args:
        theta_hat: the raw estimate.
        sigmas: residual standard deviation per series.
        lengths: observations per series.
        num_draws: simulated panels.
        seed: seed for the simulation.

    Returns:
        Tuple of (corrected coefficient, measured bias). The bias is negative
        when the raw estimate understates.

    Raises:
        ValueError: if ``num_draws`` is not positive.
    """
    if num_draws <= 0:
        raise ValueError(f"num_draws must be positive, got {num_draws!r}")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(num_draws):
        panel = simulate_panel_ar1(theta_hat, sigmas, lengths, rng)
        result = minimize_scalar(panel_ar1_neg_log_likelihood, args=(panel,),
                                 bounds=(-0.95, 0.95), method='bounded',
                                 options={'xatol': 1e-6})
        estimates.append(float(result.x))
    bias = float(np.mean(estimates)) - theta_hat
    return float(np.clip(theta_hat - bias, -0.99, 0.99)), bias


def fit_panel_ar1(series_list: SeriesList,  # demeaned return series, one per fund
                  theta_bounds: Tuple[float, float] = (-0.95, 0.95),  # search interval for theta
                  bias_correction: BiasCorrection = BiasCorrection.NONE,
                  num_bias_draws: int = DEFAULT_BIAS_DRAWS,  # simulated panels, BOOTSTRAP only
                  seed: int = 1,  # seed for the BOOTSTRAP correction
                  ) -> Dict[str, Union[float, int, Dict[str, Optional[float]]]]:
    """estimate a common AR(1) smoothing coefficient across a panel of funds.

    Args:
        series_list: demeaned return series, one per fund, either as arrays or as
            ``(label, array)`` pairs.
        theta_bounds: search interval for theta, inside (-1, 1).
        bias_correction: how to correct the small-sample bias. Defaults to
            ``NONE`` so the raw estimate is never changed without being asked.
        num_bias_draws: simulated panels, used by ``BOOTSTRAP`` only.
        seed: seed for the ``BOOTSTRAP`` correction.

    Returns:
        Dict carrying ``theta_hat`` (corrected when a correction is requested),
        ``theta_raw`` (always the uncorrected estimate), ``bias_correction``,
        ``bias`` (the amount removed, NaN when none), the asymptotic standard
        error ``se``, the profiled ``neg_log_lik``, ``fisher_info``, the
        per-series OLS estimates in ``per_series_theta``, the total lag-pair
        count ``n_total``, and the qualifying series count ``n_series``.

        ``se`` is the asymptotic error of the raw estimate. A correction moves
        the point estimate without narrowing that interval.

    Raises:
        ValueError: if ``series_list`` is empty, if ``theta_bounds`` is not a
            valid interval inside (-1, 1), or if a correction is requested on a
            panel too short to support it.
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

    theta_raw = theta_hat
    bias = float('nan')
    if bias_correction is not BiasCorrection.NONE:
        usable = [(len(r) - 1, float(np.std(r[1:] - theta_raw * r[:-1])))
                  for _, r in (_as_label_array(item, i)
                               for i, item in enumerate(series_list))
                  if len(r) >= MIN_OBS_FOR_THETA]
        if not usable:
            raise ValueError("no series is long enough to support a bias correction")
        lengths = [n for n, _ in usable]
        sigmas = [sd for _, sd in usable]
        if bias_correction is BiasCorrection.KENDALL:
            theta_hat = kendall_corrected_theta(theta_raw, sum(lengths) / len(lengths))
            bias = theta_raw - theta_hat
        else:
            theta_hat, bias = bootstrap_corrected_theta(theta_raw, sigmas, lengths,
                                                        num_draws=num_bias_draws, seed=seed)

    return {
        'theta_hat': theta_hat,
        'theta_raw': theta_raw,
        'bias_correction': bias_correction.value,
        'bias': bias,
        'se': se,
        'neg_log_lik': float(result.fun),
        'fisher_info': fisher,
        'per_series_theta': per_series,
        'n_total': n_total,
        'n_series': n_series,
    }
