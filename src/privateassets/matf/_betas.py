"""
privateassets.matf._betas — factor loadings for the MATF deflator.

The deflator needs a loading vector beta. Ordinary least squares produces one,
and it is usable: over a fifteen-year quarterly history it degrades with
collinearity rather than breaking, and a wrong-signed loading is rare. Shrinkage
is worth its dependency because it is more accurate, not because the alternative
fails.

Measured over 40 seeds on that history, mean absolute loading error against a
known beta, with equity and credit correlated at ``rho``:

======  ==========  ================  ================
rho     OLS         this estimator    OLS sign flips
======  ==========  ================  ================
0.00    0.0797      0.0483            0/40
0.50    0.0831      0.0458            0/40
0.90    0.1095      0.0713            0/40
0.95    0.1296      0.0915            0/40
0.98    0.1696      0.1290            1/40
======  ==========  ================  ================

Shrinkage cuts the error by roughly a third throughout, by a margin that widens
only slowly with collinearity. The sign constraint binds once in 200 fits, so it
is insurance against a loading the economics forbids rather than a routine
correction. Reproduced by
``test_shrinkage_beats_least_squares_under_collinearity``.

This wraps the hierarchical cluster group lasso from ``factorlasso``. Correlated
factors are clustered and penalised as a group, so collinear factors shrink
together rather than trading off against each other arbitrarily, and sign
constraints keep loadings on the side of zero the economics requires.

The estimate is **in-sample by construction**: one beta over the whole panel,
applied to every cash flow. That is the identification design, not an oversight,
and it must be stated wherever a resulting alpha is reported.
"""

# packages
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

DEFAULT_REG_LAMBDA = 1e-5  # group-penalty strength
DEFAULT_SPAN = 36  # EWMA span for the loss weights, in periods of the input frequency
DEFAULT_WARMUP_PERIOD = 12  # periods excluded from the weighted loss while the EWMA settles
DEFAULT_SPAN_FREQ = {'ME': 36, 'QE': 18}  # per-frequency span override passed to factorlasso


class SignConstraint(str, Enum):
    """Admissible sign of a factor loading."""

    POS = 'pos'  # loading constrained to be non-negative
    NEG = 'neg'  # loading constrained to be non-positive
    ZERO = 'zero'  # loading pinned to zero, excluding the factor
    FREE = 'free'  # unconstrained


_SIGN_TO_INT = {SignConstraint.POS: 1, SignConstraint.NEG: -1,
                SignConstraint.ZERO: 0, SignConstraint.FREE: np.nan}


@dataclass(frozen=True)
class FactorBetas:
    """Result of a factor-loading fit.

    Attributes:
        beta: loading per factor, indexed by factor name.
        alpha_const: economic intercept, the weighted mean of the residual.
        r2: in-sample coefficient of determination, NaN when factorlasso does
            not report one.
        n_obs: observations entering the weighted loss, after the warm-up.
        sigma_quarterly_insample: full-sample factor covariance in the units of
            the input returns. **Descriptive only.** It uses the whole panel, so
            passing it into a deflator applied at a date makes that deflator
            look ahead. Use ``rolling_ewma_quarterly_covar`` there instead.
        factor_means: full-sample mean of each factor over the same window.
        derived_signs: signs factorlasso inferred from the per-factor marginal
            regressions, when ``auto_sign_constraints`` is on. None otherwise.
    """

    beta: pd.Series
    alpha_const: float
    r2: float
    n_obs: int
    sigma_quarterly_insample: pd.DataFrame
    factor_means: pd.Series
    derived_signs: Optional[pd.DataFrame] = None


def fit_factor_betas(asset_returns: pd.Series,
                     factor_returns: pd.DataFrame,
                     sign_constraints: Optional[Dict[str, SignConstraint]] = None,  # None: all free
                     prior_beta: Optional[Dict[str, float]] = None,  # None: shrink towards zero
                     reg_lambda: float = DEFAULT_REG_LAMBDA,  # group-penalty strength
                     span: Optional[int] = DEFAULT_SPAN,  # loss-weight EWMA span; None: equal weights
                     warmup_period: int = DEFAULT_WARMUP_PERIOD,  # periods excluded while the EWMA settles
                     span_freq_dict: Optional[Dict[str, int]] = None,  # None: DEFAULT_SPAN_FREQ
                     auto_sign_constraints: bool = True,  # let factorlasso derive signs and combine them
                     cluster_cutoff_fraction: Optional[float] = None,  # None: factorlasso's own default
                     l1_weight: float = 0.0,  # elementwise L1 on top of the group L2
                     solver: str = 'CLARABEL',
                     ) -> FactorBetas:
    """estimate shrunk, sign-coherent factor loadings for one return series.

    Fits the hierarchical cluster group lasso: factors are clustered by
    correlation and penalised as groups, so a collinear pair shrinks together
    instead of one absorbing the other's loading. Sign constraints are applied
    inside the optimisation rather than by truncating afterwards, which keeps
    the solution on the constraint set rather than at a projected point off it.

    Shrinkage is towards ``prior_beta``. Leaving it None shrinks towards zero,
    which is the neutral choice. A non-zero prior is an economic view and
    belongs to the caller, not to this function.

    Args:
        asset_returns: return series of the private asset, at any frequency.
        factor_returns: factor returns on the same index. Excess returns, to
            match the deflator's convention.
        sign_constraints: admissible sign per factor. Factors absent from the
            mapping are unconstrained.
        prior_beta: shrinkage target per factor. Factors absent default to zero.
        reg_lambda: group-penalty strength.
        span: EWMA span for the loss weights, in periods of the input frequency.
            Pass None for equal weights, which is the right choice for in-sample
            identification on a short panel.
        warmup_period: periods excluded from the weighted loss while the EWMA
            settles.
        span_freq_dict: per-frequency span override passed through to
            factorlasso. None uses ``DEFAULT_SPAN_FREQ``.
        auto_sign_constraints: when True, factorlasso derives signs from the
            per-factor marginal regressions and combines them with
            ``sign_constraints``, which reduces sensitivity to a mis-specified
            mask.
        cluster_cutoff_fraction: correlation cutoff for cluster formation. None
            leaves factorlasso's own default in force.
        l1_weight: elementwise L1 mixing weight on top of the group L2 penalty.
        solver: CVXPY solver name.

    Returns:
        A :class:`FactorBetas`.

    Raises:
        ImportError: if ``factorlasso`` is not installed. Install the extra with
            ``pip install "privateassets[factors]"``.
        ValueError: if the inputs do not align, if fewer than
            ``warmup_period + 1`` complete observations survive, or if a sign
            constraint names a factor absent from ``factor_returns``.
    """
    try:
        from factorlasso import LassoModel, LassoModelType
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("fit_factor_betas needs factorlasso; install it with "
                          "pip install \"privateassets[factors]\"") from exc

    if not isinstance(asset_returns, pd.Series):
        raise ValueError(f"asset_returns must be a Series, got {type(asset_returns)!r}")
    if not isinstance(factor_returns, pd.DataFrame):
        raise ValueError(f"factor_returns must be a DataFrame, got {type(factor_returns)!r}")

    factors = list(factor_returns.columns)
    sign_constraints = sign_constraints or {}
    unknown = [f for f in sign_constraints if f not in factors]
    if unknown:
        raise ValueError(f"sign_constraints names factors absent from factor_returns: {unknown}")

    name = asset_returns.name or 'asset'
    aligned = pd.concat([asset_returns.rename(name), factor_returns],
                        axis=1, join='inner').dropna()
    if len(aligned) <= warmup_period:
        raise ValueError(f"need more than warmup_period={warmup_period} complete observations, "
                         f"got {len(aligned)}")

    y_aligned = aligned.iloc[:, 0]
    x_aligned = aligned.iloc[:, 1:]

    signs_df = pd.DataFrame(
        {f: [_SIGN_TO_INT[SignConstraint(sign_constraints.get(f, SignConstraint.FREE))]]
         for f in factors},
        index=[name])
    prior_df = pd.DataFrame({f: [(prior_beta or {}).get(f, 0.0)] for f in factors}, index=[name])

    model_kwargs = dict(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                        group_data=None,
                        demean=True,
                        reg_lambda=reg_lambda,
                        span=span,
                        span_freq_dict=span_freq_dict if span_freq_dict is not None else DEFAULT_SPAN_FREQ,
                        solver=solver,
                        warmup_period=warmup_period,
                        factors_beta_loading_signs=signs_df,
                        factors_beta_prior=prior_df,
                        l1_weight=l1_weight,
                        auto_sign_constraints=auto_sign_constraints)
    if cluster_cutoff_fraction is not None:
        model_kwargs['cutoff_fraction'] = cluster_cutoff_fraction

    model = LassoModel(**model_kwargs)
    model.fit(x=x_aligned, y=y_aligned.to_frame(name), verbose=False)

    beta = model.coef_.iloc[0]
    alpha_const = float(model.alpha_const_.iloc[0])

    # factorlasso's r2 is computed under the same span weighting as beta. A
    # hand-rolled unweighted r2 would not be the same statistic, so it is left
    # as NaN rather than reported as if it were.
    r2 = float('nan')
    result = getattr(model, 'estimation_result_', None)
    if result is not None and hasattr(result, 'r2'):
        r2 = float(np.asarray(result.r2).flatten()[0])

    post_warmup = x_aligned.iloc[warmup_period:]
    sigma = pd.DataFrame(np.cov(post_warmup.values, rowvar=False),
                         index=factors, columns=factors)

    return FactorBetas(beta=beta,
                       alpha_const=alpha_const,
                       r2=r2,
                       n_obs=len(aligned) - warmup_period,
                       sigma_quarterly_insample=sigma,
                       factor_means=post_warmup.mean(),
                       derived_signs=getattr(model, 'derived_signs_', None))
