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

The loading fit also supports the production expert-prior rule through :class:`ExpertPrior`.

Shrinkage cuts the error by roughly a third throughout, by a margin that widens
only slowly with collinearity. The sign constraint binds once in 200 fits, so it
is insurance against a loading the economics forbids rather than a routine
correction. Reproduced by
``test_shrinkage_beats_least_squares_under_collinearity``.

This wraps the hierarchical cluster group lasso from ``factorlasso``, which
groups response assets. With one response asset it reduces to LASSO. Sign
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
from inspect import signature
from numbers import Integral, Real
from typing import Dict, Optional
import warnings

DEFAULT_REG_LAMBDA = 1e-5  # group-penalty strength
DEFAULT_SPAN = 36  # EWMA span for the loss weights, in periods of the input frequency
DEFAULT_WARMUP_PERIOD = 12  # minimum history for factorlasso; offset in legacy result summaries
DEFAULT_SPAN_FREQ = {'ME': 36, 'QE': 18}  # per-frequency span override passed to factorlasso


class SignConstraint(str, Enum):
    """Admissible sign of a factor loading."""

    POS = 'pos'  # loading constrained to be non-negative
    NEG = 'neg'  # loading constrained to be non-positive
    ZERO = 'zero'  # loading pinned to zero, excluding the factor
    FREE = 'free'  # unconstrained


# factorlasso's codes: +1 non-negative, -1 non-positive, 0 fixed at zero, NaN free.
# SignConstraint.FREE maps to NaN, never to 0; automatic signs may still constrain it.
_SIGN_TO_INT = {SignConstraint.POS: 1, SignConstraint.NEG: -1,
                SignConstraint.ZERO: 0, SignConstraint.FREE: np.nan}


@dataclass(frozen=True)
class ExpertPrior:
    """the selected factor's OLS penalty centre and sign-oriented loading floor.

    factorlasso fits a univariate regression with an intercept and the fit's loss
    weights. Its slope centres the selected factor's penalty, with zero centres
    elsewhere. The hard magnitude floor is ``max(0, |b_ols| - bound_n_std * se_hac)``;
    its direction follows the OLS slope. At the default ``reg_lambda=1e-5`` the
    centre has little weight. The production convention uses one standard error,
    one Bartlett lag on quarterly inputs and three on monthly inputs. This is a
    univariate uncertainty statement, not identification of the multifactor loading.
    Manual selected-factor priors stay soft; conflicting hard signs take precedence.

    Attributes:
        factor: non-empty name of a factor_returns column.
        bound_n_std: nonnegative number of standard errors subtracted from ``|b_ols|``.
            Zero places the floor at ``|b_ols|`` itself.
        hac_lags: nonnegative Bartlett bandwidth in input periods, not a boolean.
    """

    factor: str
    bound_n_std: float = 1.0
    hac_lags: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.factor, str) or not self.factor.strip():
            raise ValueError(f"factor must be a non-empty str, got {self.factor!r}")
        if (isinstance(self.bound_n_std, (bool, np.bool_)) or not isinstance(self.bound_n_std, Real)
                or not np.isfinite(self.bound_n_std) or self.bound_n_std < 0):
            raise ValueError(f"bound_n_std must be finite and non-negative, got {self.bound_n_std!r}")
        if (isinstance(self.hac_lags, (bool, np.bool_)) or not isinstance(self.hac_lags, Integral)
                or self.hac_lags < 0):
            raise ValueError(f"hac_lags must be an int >= 0 and not a bool, got {self.hac_lags!r}")


@dataclass(frozen=True)
class PriorDiagnostics:
    """the expert regression and bound reported by factorlasso for one fit.

    Attributes:
        factor: the selected expert factor.
        centre: raw weighted univariate OLS slope; the penalty centre unless an
            explicit prior or conflicting hard sign overrides it. NaN if unavailable.
        se_hac: Bartlett HAC standard error, NaN when upstream did not compute it.
        floor: nonnegative magnitude floor; zero when no floor was imposed.
        imposed: whether the floor entered the solve as a hard bound.
        univariate_r2: centred weighted R2 of the univariate regression, or NaN.
        n_obs: observations in that regression, None when upstream did not report it.
            This can differ from FactorBetas.n_obs, which retains legacy accounting.
    """

    factor: str
    centre: float
    se_hac: float
    floor: float
    imposed: bool
    univariate_r2: float
    n_obs: Optional[int]


@dataclass(frozen=True)
class FactorBetas:
    """Result of a factor-loading fit.

    Attributes:
        beta: loading per factor, indexed by factor name.
        alpha_const: economic intercept, the weighted mean of the residual.
        r2: in-sample coefficient of determination, NaN when factorlasso does
            not report one.
        n_obs: legacy count of aligned rows minus warmup_period; not factorlasso's
            actual loss-observation count. Retained for backward compatibility.
        sigma_quarterly_insample: full-sample factor covariance in the units of
            the input returns. **Descriptive only.** It uses the whole panel, so
            passing it into a deflator applied at a date makes that deflator
            look ahead. Use ``rolling_factor_covar`` there instead. These summaries
            exclude the first warmup_period rows under the legacy convention.
        factor_means: descriptive mean of each factor over that same window.
        derived_signs: signs factorlasso inferred from the per-factor marginal
            regressions, when ``auto_sign_constraints`` is on. None otherwise.
        prior: expert regression and floor diagnostics, None without an expert prior.
    """

    beta: pd.Series
    alpha_const: float
    r2: float
    n_obs: int
    sigma_quarterly_insample: pd.DataFrame
    factor_means: pd.Series
    derived_signs: Optional[pd.DataFrame] = None
    prior: Optional[PriorDiagnostics] = None


def fit_factor_betas(asset_returns: pd.Series,
                     factor_returns: pd.DataFrame,
                     sign_constraints: Optional[Dict[str, SignConstraint]] = None,  # None: all free
                     prior_beta: Optional[Dict[str, float]] = None,  # None: shrink towards zero
                     reg_lambda: float = DEFAULT_REG_LAMBDA,  # group-penalty strength
                     span: Optional[int] = DEFAULT_SPAN,  # loss-weight EWMA span; None: equal weights
                     warmup_period: int = DEFAULT_WARMUP_PERIOD,  # minimum history; legacy summary offset
                     span_freq_dict: Optional[Dict[str, int]] = None,  # None: DEFAULT_SPAN_FREQ
                     auto_sign_constraints: bool = True,  # let factorlasso derive signs and combine them
                     cluster_cutoff_fraction: Optional[float] = None,  # None: factorlasso's own default
                     l1_weight: float = 0.0,  # elementwise L1 on top of the group L2
                     solver: str = 'CLARABEL',
                     *,
                     expert_prior: Optional[ExpertPrior] = None,  # None: no OLS centre or floor
                     ) -> FactorBetas:
    """estimate shrunk, sign-coherent factor loadings for one return series.

    Uses factorlasso's hierarchical cluster group lasso interface, which reduces
    to LASSO for this single-response fit. Sign constraints are applied
    inside the optimisation rather than by truncating afterwards, which keeps
    the solution on the constraint set rather than at a projected point off it.

    Shrinkage is towards ``prior_beta``. Leaving it None shrinks towards zero,
    which is the neutral choice. A non-zero prior is an economic view and
    belongs to the caller, not to this function.

    An ``expert_prior`` applies the production capital-market-assumptions prior
    convention: one weighted univariate OLS centre and a sign-oriented HAC floor,
    with zero centres elsewhere. Finite ``prior_beta`` entries override computed
    centres, including an explicit zero. An override on the expert factor disables
    its hard floor; a conflicting hard sign also takes precedence. Leaving
    ``prior_beta`` None is the production setting.

    Args:
        asset_returns: return series of the private asset, at any frequency.
        factor_returns: factor returns on the same index. Excess returns, to
            match the deflator's convention.
        sign_constraints: explicit admissible sign per factor. FREE and absent
            entries supply no hard sign, but auto_sign_constraints can derive one.
        prior_beta: shrinkage target per factor. Factors absent default to zero
            without an expert prior; finite entries override its computed centres.
        expert_prior: selected-factor OLS centre and floor; None preserves the
            ordinary fit. Set hac_lags in units of the input frequency.
        reg_lambda: group-penalty strength.
        span: EWMA span for the loss weights, in periods of the input frequency.
            Pass None for equal weights, which is the right choice for in-sample
            identification on a short panel.
        warmup_period: minimum usable history passed to factorlasso. The legacy
            result count and descriptive summaries exclude this many rows, but
            the equal-weight factorlasso loss uses all aligned rows.
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
        RuntimeError: if the installed factorlasso lacks expert-prior support.
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
    if expert_prior is not None:
        if not isinstance(expert_prior, ExpertPrior):
            raise ValueError(f"expert_prior must be an ExpertPrior, got {expert_prior!r}")
        if expert_prior.factor not in factors:
            raise ValueError(f"expert_prior factor {expert_prior.factor!r} absent from columns {factors!r}")
        required = {'factor_for_prior', 'expert_prior_bound_n_std', 'expert_prior_hac_lags'}
        if not required.issubset(signature(LassoModel).parameters):
            import factorlasso
            raise RuntimeError("expert_prior requires factorlasso>=1.0.0; "
                               f"installed {factorlasso.__version__}")
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
    if expert_prior is not None:
        prior_df = pd.DataFrame(np.nan, index=[name], columns=factors)
        for factor, value in (prior_beta or {}).items():
            if factor in factors and np.isfinite(value):
                prior_df.loc[name, factor] = value

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
    if expert_prior is not None:
        model_kwargs.update(apply_ols_prior=True, prior_selection_type='highest_r2',
                            factor_for_prior=pd.Series({name: expert_prior.factor}),
                            expert_prior_bound_n_std=expert_prior.bound_n_std,
                            expert_prior_hac_lags=expert_prior.hac_lags)

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
                       derived_signs=getattr(model, 'derived_signs_', None),
                       prior=_prior_diagnostics(model, name, expert_prior) if expert_prior else None)


def _prior_diagnostics(model: object, name: str, prior: ExpertPrior) -> PriorDiagnostics:
    """read upstream diagnostics without inventing statistics for a suppressed bound."""
    bounds = getattr(model, 'prior_bound_diagnostics_', None)
    rows = None if bounds is None or bounds.empty else bounds[
        (bounds['asset'] == name) & (bounds['factor'] == prior.factor)]
    r2 = getattr(model, 'ols_r2_', None)
    univariate_r2 = (float(r2.loc[name, prior.factor])
                     if r2 is not None and name in r2.index and prior.factor in r2.columns else float('nan'))
    if rows is None or rows.empty:
        warnings.warn(f"expert prior diagnostics unavailable for {name!r}, factor {prior.factor!r}",
                      UserWarning, stacklevel=3)
        return PriorDiagnostics(prior.factor, float('nan'), float('nan'), 0.0, False,
                                univariate_r2, None)
    row = rows.iloc[0]
    observations = row.get('observations', float('nan'))
    imposed = bool(row['imposed'])
    return PriorDiagnostics(factor=prior.factor,
                            centre=float(row['reference_beta']),
                            se_hac=float(row['se_hac']),
                            floor=float(row['floor']) if imposed and np.isfinite(row['floor']) else 0.0,
                            imposed=imposed,
                            univariate_r2=univariate_r2,
                            n_obs=int(observations) if pd.notna(observations) else None)
