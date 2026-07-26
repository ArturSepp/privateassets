"""
privateassets.matf._inference — resampling inference for factor loadings.

A private-asset panel is short and serially dependent, so the textbook standard
error on a factor loading is not credible. Resampling in blocks preserves the
dependence within a block while breaking it across blocks, which is what makes
the resulting interval mean something.

Resampling is delegated to ``qis.generate_bootstrapped_indices``. The indices
are drawn once and applied to the asset series and the factor panel together, so
each draw keeps the contemporaneous alignment between a fund's return and the
factor returns of the same quarter. Resampling the two independently would
destroy the very covariance the loading measures.

Draws are seeded and the seed is an argument, so an interval is reproducible.
Note that ``qis.BootstrapType.STATIONARY`` wraps circularly from ``qis 5.1.0``,
so an interval computed under an earlier version does not reproduce. State the
``qis`` version alongside any published interval.
"""

# packages
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
# qis / project
import qis
from privateassets.matf._betas import FactorBetas, fit_factor_betas

DEFAULT_NUM_SAMPLES = 1000  # resample draws
DEFAULT_BLOCK_SIZE = 12  # mean block length in periods; 12 quarters spans a cycle
DEFAULT_SEED = 1
# CLARABEL approaches a sign constraint from the interior and stops short of it,
# so "at the boundary" is a tolerance, not an equality. Loadings of this size are
# economically zero.
DEFAULT_ZERO_TOLERANCE = 1e-4


@dataclass(frozen=True)
class BetaBootstrap:
    """Resampled distribution of a factor-loading vector.

    Attributes:
        point: the loading fitted on the observed sample, not a draw mean.
        draws: one row per successful draw, one column per factor.
        lower: lower percentile of each loading.
        upper: upper percentile of each loading.
        share_at_zero: fraction of draws within tolerance of zero, per factor.
            Under a binding sign constraint the optimiser piles mass exactly on
            the boundary, so this is a description of how often the constraint
            binds. It is **not** a p-value and must not be reported as one.
        num_failed: draws whose fit did not converge and were discarded.
        ci: the percentile pair used.
        seed: seed passed to the index generator.
        qis_version: version of ``qis`` that produced the indices.
    """

    point: pd.Series
    draws: pd.DataFrame
    lower: pd.Series
    upper: pd.Series
    share_at_zero: pd.Series
    num_failed: int
    ci: Tuple[float, float]
    seed: int
    qis_version: str


def bootstrap_factor_betas(asset_returns: pd.Series,
                           factor_returns: pd.DataFrame,
                           num_samples: int = DEFAULT_NUM_SAMPLES,  # resample draws
                           block_size: int = DEFAULT_BLOCK_SIZE,  # mean block length in periods
                           bootstrap_type: qis.BootstrapType = qis.BootstrapType.STATIONARY,
                           seed: int = DEFAULT_SEED,
                           ci: Tuple[float, float] = (5.0, 95.0),  # percentile pair
                           zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,  # counts as at the boundary
                           fit_kwargs: Optional[Dict[str, Any]] = None,  # passed to fit_factor_betas
                           ) -> BetaBootstrap:
    """block-resample a factor panel and refit the loadings on every draw.

    The asset series and the factor panel are resampled with one shared index
    array, so a draw is a resample of *rows* of the joint panel. Blocks preserve
    the serial dependence that makes a private-asset return series
    autocorrelated.

    Draws whose fit fails to converge are discarded and counted rather than
    silently replaced with NaN, so a degenerate resample cannot quietly shrink
    the effective sample.

    Args:
        asset_returns: return series of the private asset.
        factor_returns: factor returns on the same index.
        num_samples: number of resample draws.
        block_size: mean block length in periods.
        bootstrap_type: resampling scheme from ``qis``.
        seed: seed for the index generator.
        ci: percentile pair for the interval.
        zero_tolerance: a loading within this of zero counts as at the boundary.
            The default reflects an interior-point solver stopping short of the
            constraint rather than landing on it.
        fit_kwargs: forwarded to :func:`~privateassets.matf.fit_factor_betas`.
            ``span`` defaults to None here, because a resampled panel has no
            meaningful time ordering for an EWMA weight.

    Returns:
        A :class:`BetaBootstrap`.

    Raises:
        ValueError: if the inputs do not overlap, if ``num_samples`` or
            ``block_size`` is not positive, or if ``ci`` is not an increasing
            pair inside [0, 100].
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples!r}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size!r}")
    low, high = ci
    if not (0.0 <= low < high <= 100.0):
        raise ValueError(f"ci must be an increasing pair inside [0, 100], got {ci!r}")

    name = asset_returns.name or 'asset'
    aligned = pd.concat([asset_returns.rename(name), factor_returns],
                        axis=1, join='inner').dropna()
    if len(aligned) <= block_size:
        raise ValueError(f"need more than block_size={block_size} overlapping observations, "
                         f"got {len(aligned)}")

    fit_kwargs = dict(fit_kwargs or {})
    fit_kwargs.setdefault('span', None)  # a resampled panel has no time ordering to weight

    point_fit: FactorBetas = fit_factor_betas(aligned.iloc[:, 0], aligned.iloc[:, 1:],
                                              **fit_kwargs)

    n_obs = len(aligned)
    indices = qis.generate_bootstrapped_indices(num_data_index=n_obs,
                                                bootstrap_type=bootstrap_type,
                                                num_samples=num_samples,
                                                index_length=n_obs,
                                                block_size=block_size,
                                                seed=seed)
    indices = np.asarray(indices)
    if indices.shape[0] != num_samples:  # qis returns (num_samples, index_length) or its transpose
        indices = indices.T

    rows = []
    num_failed = 0
    for draw in range(num_samples):
        take = np.asarray(indices[draw], dtype=int)
        # One index array applied to both, so the asset and its factors stay paired.
        resampled = aligned.iloc[take].copy()
        resampled.index = aligned.index  # restore a monotone index for frequency inference
        try:
            fit = fit_factor_betas(resampled.iloc[:, 0], resampled.iloc[:, 1:], **fit_kwargs)
        except (ValueError, np.linalg.LinAlgError):
            num_failed += 1
            continue
        rows.append(fit.beta)

    if not rows:
        raise ValueError(f"every one of the {num_samples} draws failed to fit")

    draws = pd.DataFrame(rows).reset_index(drop=True)
    return BetaBootstrap(point=point_fit.beta,
                         draws=draws,
                         lower=draws.quantile(low / 100.0),
                         upper=draws.quantile(high / 100.0),
                         share_at_zero=(draws.abs() <= zero_tolerance).mean(),
                         num_failed=num_failed,
                         ci=ci,
                         seed=seed,
                         qis_version=_qis_version())


def _qis_version() -> str:
    """installed qis version, recorded so a resampled result carries its provenance."""
    try:
        from importlib.metadata import version
        return version('qis')
    except Exception:  # pragma: no cover - only when metadata is unavailable
        return 'unknown'
