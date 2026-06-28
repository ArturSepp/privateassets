"""
Fixed-θ AR(1) unsmoothing (Getmansky-Lo-Makarov inversion).

Production AR(1) coefficient is fixed at the panel MLE estimate
θ̂_panel = 0.176 (unweighted, cap-filtered, demeaned), rather than
re-estimated from the pooled series each run. This decouples the
unsmoothing parameter from the specific aggregation choice.

Formula:    r_t* = (r_t - θ · r_{t-1}) / (1 - θ)

For the first observation (no lag), we keep r_0* = r_0.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Production AR(1) smoothing coefficient
# = panel MLE, unweighted, cap-filtered, demeaned (see panel_ar1_mle.py)
PRODUCTION_THETA = 0.176
PRODUCTION_VOL_INFLATION = 1.0 / (1.0 - PRODUCTION_THETA)  # 1.214


def apply_fixed_theta_unsmooth(
    returns: pd.Series,
    theta: float = PRODUCTION_THETA,
) -> pd.Series:
    """
    AR(1) Getmansky-Lo-Makarov inversion with fixed θ.

    Args:
        returns: observed return series (any frequency)
        theta:   AR(1) coefficient (default = production panel MLE)

    Returns:
        Unsmoothed series with same index/length as input. First obs
        is preserved; subsequent obs use lag-correction.
    """
    if abs(theta) >= 0.99:
        raise ValueError(f"theta={theta} too close to ±1 — unsmoothing diverges")
    vals = returns.values.astype(float)
    out = vals.copy()
    denom = 1.0 - theta
    for t in range(1, len(vals)):
        if np.isnan(vals[t]) or np.isnan(vals[t - 1]):
            continue
        out[t] = (vals[t] - theta * vals[t - 1]) / denom
    return pd.Series(out, index=returns.index, name=returns.name)
