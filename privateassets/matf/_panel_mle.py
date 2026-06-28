"""
Panel MLE for common AR(1) coefficient across vintages.

Model: each vintage i has quarterly NAV-implied returns r_{i,t} that
follow:
   r_{i,t} = θ · r_{i,t-1} + ε_{i,t},  ε_{i,t} ~ N(0, σ_i²)

with COMMON AR coefficient θ shared across vintages, but idiosyncratic
residual variances σ_i².

Profile-out variances and reduce to a 1-D optimization in θ. With
σ_i² profiled at MLE σ̂_i²(θ) = mean_t (r_{i,t} - θ r_{i,t-1})², the
profile log-likelihood for θ is:

   L_p(θ) = -(1/2) Σ_i n_i log σ̂_i²(θ)   + const

Maximize over θ ∈ (-1, 1).

Compare to:
  - Production pooled AR(1): θ_pooled = 0.142
  - Per-vintage average:     avg θ_i  = 0.136

Author: A. Sepp / the desk
"""
from __future__ import annotations
from typing import Dict
import numpy as np
from scipy.optimize import minimize_scalar


def panel_ar1_neg_log_likelihood(
    theta: float,
    series_list: list,   # list of (label, np.ndarray) per vintage, demeaned
) -> float:
    """
    Profile-out negative log-likelihood for panel AR(1) with common θ
    and per-vintage idiosyncratic σ_i². Series are pre-demeaned.

    Returns: -L_p(θ) where L_p is the profile log-likelihood.
    """
    nll = 0.0
    for item in series_list:
        r = item[1] if isinstance(item, tuple) else item
        if len(r) < 3:
            continue
        eps = r[1:] - theta * r[:-1]
        n_i = len(eps)
        sigma2_hat = np.mean(eps ** 2)
        if sigma2_hat <= 0:
            return np.inf
        nll += 0.5 * n_i * np.log(sigma2_hat)
    return nll


def panel_ar1_grad(theta: float, series_list: list) -> float:
    """Numerical derivative of NLL wrt theta (used for diagnostics)."""
    eps = 1e-6
    return (panel_ar1_neg_log_likelihood(theta + eps, series_list)
            - panel_ar1_neg_log_likelihood(theta - eps, series_list)) / (2 * eps)


def fisher_info_panel_ar1(theta_hat: float, series_list: list) -> float:
    """
    Approximate Fisher information for theta from the panel.
    For AR(1) with σ_i² known, the per-obs Fisher info is
    Var(r_{i,t-1}) / σ_i². With σ_i² profiled at MLE this becomes
    n_i · Var(r_{i,t-1}) / sum((r_{i,t} - θ r_{i,t-1})²).
    """
    fisher = 0.0
    for item in series_list:
        # series_list is list of (label, array)
        r = item[1] if isinstance(item, tuple) else item
        if len(r) < 3:
            continue
        eps = r[1:] - theta_hat * r[:-1]
        sigma2_hat = np.mean(eps ** 2)
        if sigma2_hat <= 0:
            continue
        var_lag = np.var(r[:-1])
        n_i = len(eps)
        fisher += n_i * var_lag / sigma2_hat
    return fisher


def fit_panel_ar1(series_list: list) -> Dict:
    """
    Fit common-θ panel AR(1) by profile MLE, return θ_hat and
    asymptotic standard error.
    """
    res = minimize_scalar(
        panel_ar1_neg_log_likelihood,
        args=(series_list,),
        bounds=(-0.95, 0.95),
        method='bounded',
        options={'xatol': 1e-6},
    )
    theta_hat = float(res.x)
    fisher = fisher_info_panel_ar1(theta_hat, series_list)
    se = float(1.0 / np.sqrt(fisher)) if fisher > 0 else float('nan')

    # Also compute per-vintage MLE for comparison
    per_vintage = {}
    for v, r in series_list:
        if len(r) < 5:
            per_vintage[v] = None
            continue
        # Per-vintage AR(1) MLE = OLS of r_t on r_{t-1}
        y = r[1:]; x = r[:-1]
        if np.var(x) < 1e-10:
            per_vintage[v] = None
            continue
        theta_i = float(np.cov(y, x)[0, 1] / np.var(x))
        per_vintage[v] = theta_i

    # n_total
    n_total = sum(max(0, len(r) - 1) for _, r in series_list)

    return {
        'theta_hat': theta_hat,
        'se': se,
        'neg_log_lik': float(res.fun),
        'fisher_info': fisher,
        'per_vintage_theta': per_vintage,
        'n_total': n_total,
        'n_vintages': len([1 for _, r in series_list if len(r) >= 5]),
    }
