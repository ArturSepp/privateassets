"""
Production rolling EWMA factor covariance for MATF-α deflator.

Uses qis.estimate_rolling_ewma_covar (the canonical production
implementation). Configuration:

  - returns_freq    'ME'  (end-of-month log returns)
  - span            60    (5-year EWMA half-life equivalent)
  - demean          True  (rolling EWMA mean subtracted)
  - rebalancing     'QE'  (one Σ per quarter end)

This matches the ROSAA SAA factor-covariance convention.

Author: A. Sepp / the desk
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
import qis


def build_rolling_sigma_h(
    levels_daily: pd.DataFrame,
    span_months: int = 60,
    returns_freq: str = 'ME',
    demean: bool = True,
) -> Dict[pd.Timestamp, np.ndarray]:
    """
    Build a dict of quarterly factor covariance matrices, one per
    quarter-end, using the qis canonical EWMA-rolling implementation.

    Returns
    -------
    dict mapping quarter-end Timestamp -> (M, M) Σ_h matrix in
    QUARTERLY units (i.e., the matrix that multiplies horizon h
    measured in quarters in the Eq 3 deflator).
    """
    covars_an = qis.estimate_rolling_ewma_covar(
        prices=levels_daily,
        returns_freq=returns_freq,
        rebalancing_freq='QE',
        span=span_months,
        demean=demean,
        apply_an_factor=True,
    )
    # qis returns ANNUALIZED covariance.
    # For Eq 3, we need quarterly Σ_h, so divide by 4.
    out = {}
    for date, sigma_an in covars_an.items():
        key = date.tz_localize(None) if getattr(date, 'tz', None) else date
        out[key] = sigma_an.values / 4.0
    return out


def closest_or_default_sigma(
    sigma_dict: Dict[pd.Timestamp, np.ndarray],
    quarter_end: pd.Timestamp,
    default: np.ndarray,
) -> np.ndarray:
    """Look up Σ for a quarter end, falling back to default if absent."""
    qe = quarter_end.tz_localize(None) if getattr(quarter_end, 'tz', None) else quarter_end
    if qe in sigma_dict:
        return sigma_dict[qe]
    # Find the closest preceding date
    available = sorted([d for d in sigma_dict.keys() if d <= qe])
    if available:
        return sigma_dict[available[-1]]
    return default
