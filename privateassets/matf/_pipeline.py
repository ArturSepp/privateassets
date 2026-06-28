"""
MATF-alpha pipeline — implementation of the design doc framework for
the strategy, leveraging:

  * _pme.py for cash flow / NAV plumbing
  * futures_risk_factors.csv for MATF factor levels (1999-2026)
  * qis.unsmooth_returns_glm for AR(q) Getmansky-Lo-Makarov unsmoothing
  * cvxpy for sign-constrained ridge regression (HCGL stand-in for v0.1)

Layers run side-by-side:
  L1a  KS-PME              (single-factor, beta=1, no Jensen)
  L1b  KN16-GPME           (CAPM SDF discount)
  L1c  KN24-alpha CAPM     (single-factor benchmark portfolio, GMM beta)
  L2   MATF-alpha          (multi-factor benchmark, sign-constrained beta)

Analysis window starts 2000-01-01 (factor history begins 1999-04, but we
clip vintages with first call before 2000 to keep the cash-flow universe
fully covered by the factor panel).

Author: A. Sepp / the desk
"""
from __future__ import annotations
from privateassets.matf import OUTPUT_DIR

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import brentq, minimize_scalar
import qis
import cvxpy as cp
from factorlasso import LassoModel, LassoModelType, compute_ewm_covar

from privateassets.matf._pme import load_cash_flows, load_navs, xirr, _short_label  # noqa


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Cfg:
    DATA: Path = None  # filled by __post_init__
    FACTORS: Path = None  # filled by __post_init__
    RF_CSV: Path = None  # filled by __post_init__
    OUT: Path = OUTPUT_DIR
    ASOF: str = '2025-12-31'
    ANALYSIS_START: str = '2000-01-01'

    # MATF factors (column order in CSV matches MatfRiskFactors enum)
    FACTORS_ALL: Tuple[str, ...] = (
        'Equity', 'Rates', 'Credit', 'Carry', 'Inflation',
        'Commodities', 'Private Equity', 'Rates Vol', 'Fx',
    )

    # Sign mask for the strategy (drawdown distressed credit).
    # Production rule:
    #   non-negative on Equity, Rates, Credit, Carry, PE
    #   non-positive on Long Rates Vol
    #   unconstrained on Inflation, Commodities, FX
    SIGN_MASK: Dict[str, str] = None  # filled in __post_init__

    # Per-factor prior MEANS for ridge regularization.
    # Loss is  lam * ||beta - prior_mean||^2 (component-wise).
    # PE prior of 0.5 reflects the institutional belief that distressed
    # credit drawdown vehicles carry meaningful PE-factor exposure on
    # average, even when sample-period variance attribution is noisy.
    PRIOR_MEAN: Dict[str, float] = None  # filled in __post_init__

    # ------------------------------------------------------------------
    # HCGL specification — extends rosaa.core.covar_estimator_spec
    # .get_prod_covar_estimator() to the in-sample identification setting.
    #
    # Production span_freq_dict={'ME':36,'QE':18} is calibrated for
    # ROLLING covariance estimation, where recent regime should dominate.
    # For IN-SAMPLE beta identification on the full LP-data panel we want
    # flat weighting — span=200 makes alpha ≈ 0.01 so the EWMA decays
    # negligibly across our 86-quarter window. Other parameters
    # (reg_lambda, demean, warmup, solver) match production exactly.
    # ------------------------------------------------------------------
    RIDGE_LAMBDA: float = 1e-5          # rosaa prod: reg_lambda=1e-5
    BETAS_SPAN_Q: int = 200             # in-sample: effectively flat weights
                                        # (prod QE span=18 is for rolling estim)
    FACTOR_COVAR_SPAN_Q: int = 200      # match betas span for consistency
    WARMUP_PERIOD_Q: int = 12           # rosaa prod: warmup_period=12

    # AR order for unsmoothing (Getmansky-Lo-Makarov)
    AR_ORDER: int = 1

    # Bootstrap
    BOOT_B: int = 2_000          # legacy alpha-level bootstrap (kept for reference)
    BOOT_B_BETA: int = 500       # beta-level bootstrap (coefficient inference)
    BOOT_B_ALPHA: int = 500      # per-vintage alpha bootstrap (whole-panel)
    MEAN_BLOCK_Q: int = 12       # stationary block mean length (quarters);
                                 # 12 = 3y, captures business-cycle persistence

    # Route A (multi-vintage panel) configuration
    LAM_GRID_A: Tuple[float, ...] = (1e-5, 1e-3, 1e-2, 1e-1, 1.0)
    LAM_HEADLINE_A: float = 1e-2  # at this λ, A.2 ≈ B with mild per-vintage slack

    # Risk-free curve: use US 3M as ref
    RF_BLOOMBERG: str = 'USD'  # placeholder; we synthesize below

    DPI_MATURITY: float = 0.8

    # Factor covariance specification for the deflator (Eq 3).
    # 'rolling_60m' is production: rolling EWMA on monthly returns with
    # span=60 months and demean=True, via qis.estimate_rolling_ewma_covar.
    # 'constant' uses a single full-sample EWMA covariance.
    SIGMA_SPEC: str = 'rolling_60m'
    ROLLING_SPAN_MONTHS: int = 60
    ROLLING_RETURNS_FREQ: str = 'ME'
    ROLLING_DEMEAN: bool = True

    # Unsmoothing AR(1) coefficient. Production setting fixes θ at the
    # unweighted, cap-filtered, demeaned panel MLE estimate computed in
    # panel_ar1_mle.py. The vol inflation factor is 1/(1-θ) ≈ 1.214.
    # Set to None to defer to qis.unsmooth_returns_glm's pooled estimate.
    UNSMOOTH_THETA: float = 0.176

    def __post_init__(self) -> None:
        # Fill in data and output paths from package config if not provided
        from privateassets.matf import DATA_XLSX, FACTORS_CSV, RF_CSV, OUTPUT_DIR
        if self.DATA is None:
            self.DATA = DATA_XLSX
        if self.FACTORS is None:
            self.FACTORS = FACTORS_CSV
        if self.RF_CSV is None:
            self.RF_CSV = RF_CSV
        if self.OUT is None:
            self.OUT = OUTPUT_DIR

        # Sign-mask convention: 'pos', 'neg', 'zero' (excluded), 'free'
        # Carry pinned to zero: in the multi-factor decomposition the
        # Carry factor is highly collinear with Credit (both are
        # spread-tightening trades during normal regimes). After Credit
        # absorbs the dominant systematic exposure, the residual Carry
        # loading is statistically marginal (block-bootstrap p ≈ 0.11).
        # Forcing β_Carry = 0 yields a sparser, more interpretable
        # decomposition and avoids the Carry/Credit identification
        # ambiguity. β_Credit will absorb what was previously split.
        self.SIGN_MASK = {
            'Equity':         'pos',
            'Rates':          'pos',
            'Credit':         'pos',
            'Carry':          'zero',  # excluded — collinear with Credit
            'Inflation':      'free',
            'Commodities':    'free',
            'Private Equity': 'pos',
            'Rates Vol':      'neg',
            'Fx':             'free',
        }
        self.PRIOR_MEAN = {
            'Equity':         0.0,
            'Rates':          0.0,
            'Credit':         0.0,
            'Carry':          0.0,
            'Inflation':      0.0,
            'Commodities':    0.0,
            'Private Equity': 0.5,   # institutional prior on PE factor exposure
            'Rates Vol':      0.0,
            'Fx':             0.0,
        }


# =============================================================================
# Factor data loading
# =============================================================================

def load_factor_levels(cfg: Cfg) -> pd.DataFrame:
    """Load MATF factor LEVELS from CSV. Each column is a vol-targeted
    total-return index starting at 100."""
    df = pd.read_csv(cfg.FACTORS, parse_dates=['date'], index_col='date')
    df = df[list(cfg.FACTORS_ALL)].sort_index()
    return df


def factor_quarterly_log_returns(levels: pd.DataFrame) -> pd.DataFrame:
    """Resample factor levels to quarter-end and return log returns."""
    qend = levels.resample('QE').last().dropna(how='all')
    log_ret = np.log(qend / qend.shift(1)).dropna(how='all')
    return log_ret


def load_rf_quarterly(rf_path: Path,
                      start: pd.Timestamp,
                      end: pd.Timestamp) -> pd.Series:
    """Load USD 3M rate from CSV, resample to quarter-end mean, convert to
    quarterly simple yield (i.e. annualized rate / 4).

    Expected CSV format: columns ['date', '3m_rate']; rate is annualized.
    """
    df = pd.read_csv(rf_path, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    # Quarter-end mean of the daily annualized rate, divide by 4 → quarterly simple
    rf_q = df['3m_rate'].resample('QE').mean() / 4.0
    return rf_q.loc[start:end]


# =============================================================================
# NAV-implied quarterly fund returns (modified Dietz) + pooling
# =============================================================================

def per_vintage_excess_unsmoothed(
    ret_panel: pd.DataFrame,
    rf_quarterly: pd.Series,
    ar_order: int,
    min_obs: int = 6,
) -> pd.DataFrame:
    """
    For each vintage column, subtract rf, AR(q)-unsmooth, and return log
    excess returns. Vintages with fewer than ``min_obs`` quarterly
    observations get pass-through (no unsmoothing) — too few obs for
    AR-coefficient identification.

    Returns a DataFrame with the same shape as ``ret_panel`` and NaN
    where vintages were not active.
    """
    rf_aligned = rf_quarterly.reindex(ret_panel.index).ffill().fillna(0.0)
    excess = ret_panel.subtract(rf_aligned, axis=0)

    out = pd.DataFrame(index=ret_panel.index, columns=ret_panel.columns, dtype=float)
    for v in ret_panel.columns:
        s = excess[v].dropna()
        if len(s) < max(min_obs, ar_order + 4):
            out[v] = np.log1p(s.clip(lower=-0.99)).reindex(ret_panel.index)
            continue
        qgrid_v = pd.date_range(
            s.index.min().to_period('Q').to_timestamp(how='end').normalize(),
            s.index.max(), freq='QE')
        s_aligned = s.reindex(qgrid_v).ffill().fillna(0.0)
        try:
            unsmoothed, _ = qis.unsmooth_returns_glm(
                s_aligned, ar_order=ar_order, return_diagnostics=True)
            out[v] = np.log1p(unsmoothed.clip(lower=-0.99)).reindex(ret_panel.index)
        except (ValueError, np.linalg.LinAlgError):
            out[v] = np.log1p(s.clip(lower=-0.99)).reindex(ret_panel.index)
    return out


def fit_route_A_multi_vintage(
    y_panel: pd.DataFrame,        # T × K, NaN where vintage inactive
    X: pd.DataFrame,              # T × M factor returns
    sign_mask: Dict[str, str],
    prior_mean_per_vintage: pd.DataFrame,   # K × M — typically replicated rows
    lam: float,
    span: int,
    warmup_period: int,
    group_label: str = 'all',
) -> Tuple[pd.DataFrame, LassoModel]:
    """
    Multi-vintage panel HCGL with all vintages in a single group.

    The penalty becomes Σ_k λ · ‖β[:, k] - μ[:, k]‖₂ — a group L2 norm
    over vintages for each factor. This couples vintages so that a
    factor is either active across all vintages (with possibly varying
    magnitudes) or off for all of them. The 'shared sparsity' is the
    asset-class assumption: all vintages of the strategy are
    distressed credit and should have the same factor exposure pattern.

    Returns
    -------
    coef : pd.DataFrame, shape (K, M)
        Per-vintage factor loadings.
    model : factorlasso.LassoModel
        Fitted model object (for downstream R², residuals, etc.).
    """
    # NaN columns dropped — vintages with no usable data
    valid_cols = y_panel.dropna(how='all', axis=1).columns
    y_use = y_panel[valid_cols]
    K = len(valid_cols)
    cols = list(X.columns)

    # Sign mask DataFrame: K × M, replicated across vintages
    sign_to_int = {'pos': 1, 'neg': -1, 'zero': 0, 'free': np.nan}
    signs_df = pd.DataFrame(
        np.tile([sign_to_int[sign_mask.get(c, 'free')] for c in cols], (K, 1)),
        index=valid_cols, columns=cols, dtype=float,
    )

    # Prior DataFrame: K × M, taken from prior_mean_per_vintage (which is K × M)
    prior_df = prior_mean_per_vintage.reindex(index=valid_cols, columns=cols).fillna(0.0)

    # Single group across all vintages
    group_data = pd.Series(group_label, index=valid_cols)

    model = LassoModel(
        model_type=LassoModelType.GROUP_LASSO,
        group_data=group_data,
        reg_lambda=lam,
        span=span,
        l1_weight=0.0,
        demean=True,
        solver='CLARABEL',
        warmup_period=warmup_period,
        factors_beta_loading_signs=signs_df,
        factors_beta_prior=prior_df,
    )
    model.fit(x=X, y=y_use, verbose=False)
    return model.coef_, model


def fit_route_A_simple(
    y_panel: pd.DataFrame,
    X: pd.DataFrame,
    sign_mask: Dict[str, str],
    prior_mean_dict: Dict[str, float],
    lam: float,
    span: int,
    warmup_period: int,
) -> Tuple[pd.DataFrame, LassoModel]:
    """Convenience wrapper: build per-vintage prior frame from a single dict
    (replicated across vintages), then call fit_route_A_multi_vintage."""
    prior_df = pd.DataFrame(
        np.tile([prior_mean_dict.get(c, 0.0) for c in X.columns],
                (y_panel.shape[1], 1)),
        index=y_panel.columns, columns=X.columns, dtype=float,
    )
    return fit_route_A_multi_vintage(
        y_panel=y_panel, X=X, sign_mask=sign_mask,
        prior_mean_per_vintage=prior_df,
        lam=lam, span=span, warmup_period=warmup_period,
    )


def per_vintage_quarterly_returns(
    cf: pd.DataFrame,
    navs: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each fund (vintage), construct quarterly NAV-implied returns
    using the modified Dietz approximation:

        r_q = (NAV_end + Distrib_q - Contrib_q - NAV_start)
              / (NAV_start + 0.5 * Contrib_q - 0.5 * Distrib_q)

    Returns
    -------
    returns_panel : DataFrame  index=quarter_end, cols=vintage_label
    capital_panel : DataFrame  same shape, the denominator (active capital)
                               used for contribution-weighted pooling.
    """
    cf = cf.copy()
    navs = navs.copy()
    cf['quarter'] = cf['date'].dt.to_period('Q').dt.to_timestamp(how='end').dt.normalize()
    navs['quarter'] = navs['date'].dt.to_period('Q').dt.to_timestamp(how='end').dt.normalize()

    funds = sorted(cf['vintage_label'].unique())
    quarters = sorted(set(cf['quarter']) | set(navs['quarter']))

    qcf = cf.groupby(['vintage_label', 'quarter']).apply(
        lambda x: pd.Series({
            'contrib': -x.loc[x['amount'] < 0, 'amount'].sum(),
            'distrib': x.loc[x['amount'] > 0, 'amount'].sum(),
        }), include_groups=False
    ).reset_index()
    qcf_w = qcf.pivot(index='quarter', columns='vintage_label',
                      values=['contrib', 'distrib']).fillna(0.0)

    qnav = navs.sort_values('date').groupby(['vintage_label', 'quarter']).last().reset_index()
    nav_w = qnav.pivot(index='quarter', columns='vintage_label', values='nav')
    nav_w = nav_w.reindex(sorted(set(qcf_w.index) | set(nav_w.index))).sort_index()
    nav_w = nav_w.ffill()  # carry NAVs forward between observation dates

    returns = pd.DataFrame(index=nav_w.index, columns=funds, dtype=float)
    capital = pd.DataFrame(index=nav_w.index, columns=funds, dtype=float)

    for f in funds:
        nav_s = nav_w[f] if f in nav_w.columns else pd.Series(index=nav_w.index, dtype=float)
        c_s = qcf_w['contrib'][f] if f in qcf_w['contrib'].columns else \
              pd.Series(0.0, index=nav_w.index)
        d_s = qcf_w['distrib'][f] if f in qcf_w['distrib'].columns else \
              pd.Series(0.0, index=nav_w.index)
        c_s = c_s.reindex(nav_w.index).fillna(0.0)
        d_s = d_s.reindex(nav_w.index).fillna(0.0)

        nav_prev = nav_s.shift(1).fillna(0.0)
        denom = nav_prev + 0.5 * c_s - 0.5 * d_s
        numer = nav_s.fillna(0.0) + d_s - c_s - nav_prev
        r = numer / denom.where(denom > 0)
        returns[f] = r
        capital[f] = denom.clip(lower=0)

    return returns, capital


def pool_manager_returns(
    returns: pd.DataFrame,
    capital: pd.DataFrame,
) -> pd.Series:
    """
    Contribution-weighted aggregate of vintage-level quarterly returns.
    Quarter q's pooled return is sum_i (w_q^i * r_q^i) / sum_i w_q^i,
    where w_q^i is vintage i's denominator (active capital) in quarter q.
    """
    valid = returns.notna() & capital.gt(0)
    R = returns.where(valid, 0.0)
    W = capital.where(valid, 0.0)
    num = (R * W).sum(axis=1)
    den = W.sum(axis=1)
    pooled = num / den.where(den > 0)
    return pooled.dropna().rename('pooled')


# =============================================================================
# Sign-constrained ridge — HCGL stand-in for v0.1
# =============================================================================

@dataclass
class BetaFit:
    beta: pd.Series
    alpha_const: float
    r2: float
    n_obs: int
    Sigma_h: pd.DataFrame  # quarterly factor covariance matrix
    factor_means_q: pd.Series


def fit_signed_ridge(
    y: pd.Series,
    X: pd.DataFrame,
    sign_mask: Dict[str, str],
    prior_mean: Dict[str, float] = None,
    lam: float = 1e-5,
    span: int = 36,
    warmup_period: int = 12,
    auto_sign_constraints: bool = True,
    cluster_cutoff_fraction: Optional[float] = None,
    l1_weight: float = 0.0,
) -> BetaFit:
    """
    Production HCGL factor regression via ``factorlasso.LassoModel``
    (v0.3.6+ spec).

    Despite the historical name (``fit_signed_ridge``), this function now
    wraps the production estimator from
    ``rosaa.core.covar_estimator_spec.get_prod_covar_estimator()``:

        LassoModel(
            model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
            reg_lambda=lam,
            span=span,
            span_freq_dict={'ME': 36, 'QE': 18},
            l1_weight=l1_weight,
            demean=True,
            solver='CLARABEL',
            warmup_period=warmup_period,
            factors_beta_loading_signs=signs_df,
            factors_beta_prior=prior_df,
            auto_sign_constraints=True,
        )

    Parameters
    ----------
    sign_mask
        User-provided sign constraints. Used as the explicit constraint
        when ``auto_sign_constraints=False``. When ``auto_sign_constraints=True``
        the mask is still passed to factorlasso, which combines it with
        signs auto-derived from the marginal regression of y on each
        factor; the auto-derived signs are accessible afterwards via
        ``model.derived_signs_``.
    span
        EWMA span for factor covariance and the loss-function weights.
        Production default ``span=36`` (~9 years quarterly / ~3 years
        monthly). For in-sample identification on thin panels pass
        ``span=None`` to use sample-mean demeaning.
    auto_sign_constraints
        v0.3.6 feature. When True, factorlasso derives data-driven sign
        constraints from per-factor marginal regressions and combines
        them with any user-supplied ``factors_beta_loading_signs``. This
        is the recommended production setting since it reduces the
        sensitivity of β to a possibly mis-specified sign mask while
        still respecting any economic priors the user does want to
        enforce.
    cluster_cutoff_fraction
        Pass-through to factorlasso. When ``None``, factorlasso's
        documented default (0.5) is used. Only set explicitly to override.
    l1_weight
        Elementwise L1 mixing weight on top of the group L2 penalty.
        Default 0.0 (pure group L2 = ridge with sign constraints).

    The ``BetaFit.alpha_const`` field carries the **economic intercept**
    ``α = ȳ_weighted - x̄_weighted · β`` (consistent with β under the same
    span weighting). With factorlasso 0.3.6+ this is exposed directly as
    ``model.alpha_const_``; we read it without post-hoc reconstruction.

    Sign mask conversion: 'pos'→+1, 'neg'→-1, 'zero'→0, 'free'→NaN.
    """
    aligned = pd.concat([y, X], axis=1).dropna()
    if len(aligned) <= warmup_period:
        raise ValueError(f"need > {warmup_period} obs, got {len(aligned)}")

    y_aligned = aligned.iloc[:, 0]
    X_aligned = aligned.iloc[:, 1:]
    cols = list(X_aligned.columns)

    if prior_mean is None:
        prior_mean = {c: 0.0 for c in cols}

    # Build factorlasso DataFrame inputs (single asset → 1 × M)
    sign_to_int = {'pos': 1, 'neg': -1, 'zero': 0, 'free': np.nan}
    signs_df = pd.DataFrame(
        {c: [sign_to_int[sign_mask.get(c, 'free')]] for c in cols},
        index=[y_aligned.name or 'asset'],
    )
    prior_df = pd.DataFrame(
        {c: [prior_mean.get(c, 0.0)] for c in cols},
        index=[y_aligned.name or 'asset'],
    )
    y_df = y_aligned.to_frame(y_aligned.name or 'asset')

    # Build LassoModel with v0.3.6 production spec.
    # We pass cutoff_fraction only when explicitly overridden so the
    # factorlasso default (0.5) remains the single source of truth.
    lasso_model_kwargs = dict(
        model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
        group_data=None,
        demean=True,
        reg_lambda=lam,
        span=span,
        span_freq_dict={'ME': 36, 'QE': 18},
        solver='CLARABEL',
        warmup_period=warmup_period,
        factors_beta_loading_signs=signs_df,
        factors_beta_prior=prior_df,
        l1_weight=l1_weight,
        auto_sign_constraints=auto_sign_constraints,
    )
    if cluster_cutoff_fraction is not None:
        lasso_model_kwargs['cutoff_fraction'] = cluster_cutoff_fraction

    model = LassoModel(**lasso_model_kwargs)
    model.fit(x=X_aligned, y=y_df, verbose=False)

    beta = model.coef_.iloc[0]
    # v0.3.6: alpha_const_ holds the ECONOMIC intercept α (weighted-
    # consistent with β under the same span weighting). The legacy
    # intercept_ attribute holds the raw solver residual-mean diagnostic
    # — not what we want here.
    alpha_const = float(model.alpha_const_.iloc[0])

    # Diagnostics: R² from estimation_result_ if present, else compute
    if model.estimation_result_ is not None and hasattr(model.estimation_result_, 'r2'):
        r2_arr = np.asarray(model.estimation_result_.r2)
        r2_val = float(r2_arr.flatten()[0])
    else:
        yhat = alpha_const + X_aligned.values @ beta.values
        ss_res = float(((y_aligned.values - yhat) ** 2).sum())
        ss_tot = float(((y_aligned.values - y_aligned.values.mean()) ** 2).sum())
        r2_val = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    n_obs = len(aligned) - warmup_period

    # EWMA factor covariance via factorlasso utility (consistent
    # with the production estimator's factor_covar_span semantics).
    Xv = X_aligned.iloc[warmup_period:].values
    Sigma_h_np = compute_ewm_covar(a=Xv, span=span)
    Sigma_h = pd.DataFrame(Sigma_h_np, index=cols, columns=cols)
    means = pd.Series(Xv.mean(axis=0), index=cols)

    return BetaFit(
        beta=beta, alpha_const=alpha_const, r2=r2_val, n_obs=n_obs,
        Sigma_h=Sigma_h, factor_means_q=means,
    )


# =============================================================================
# Deflators — return gross deflator R_h^b for each cash flow date
# =============================================================================

def stationary_block_indices(T: int,
                             mean_block: float,
                             rng: np.random.Generator,
                             min_block: int = 1) -> np.ndarray:
    """Politis-Romano (1994) stationary block bootstrap index sampler.

    Returns ``T`` resampled indices into ``[0, T)``. Block lengths are
    drawn from ``Geom(1/mean_block)`` and floored at ``min_block``.
    Blocks wrap circularly so the rightmost observations are not
    under-sampled.

    For our QE quarterly setting, ``mean_block = 12`` corresponds to
    ~3-year blocks — long enough to capture business-cycle persistence
    (credit cycles ~5-7y), short enough to deliver meaningful
    resample variation across the 86-quarter regression panel
    (~7 blocks per resample on average).

    Parameters
    ----------
    T : int
        Length of the time series being resampled.
    mean_block : float
        Mean block length. Block lengths follow ``Geom(1/mean_block)``
        with mean ``mean_block``.
    rng : np.random.Generator
        Numpy random generator. Pass the same generator across calls
        to get different blocks each iteration.
    min_block : int, default 1
        Minimum block length. ``min_block=1`` is correct for single-
        frequency panels (our case). For the mixed-frequency 15-asset
        universe of the MATF-CMA paper, set ``min_block=3`` so every
        block contains at least one full quarter and quarterly assets
        receive at least one non-NaN observation per block.

    Returns
    -------
    idx : np.ndarray of shape (T,) and dtype int64
        Indices into the original series; can be applied to multiple
        paired panels (asset returns, factor returns, residuals) for
        joint resampling that preserves cross-section correlation.
    """
    p = 1.0 / float(mean_block)
    idx = np.empty(T, dtype=np.int64)
    filled = 0
    while filled < T:
        start = int(rng.integers(0, T))
        L = max(min_block, int(rng.geometric(p)))
        end = min(filled + L, T)
        span = end - filled
        idx[filled:end] = (start + np.arange(span)) % T   # circular
        filled = end
    return idx


def bootstrap_beta(
    y: pd.Series,
    X: pd.DataFrame,
    sign_mask: Dict[str, str],
    prior_mean: Dict[str, float],
    lam: float,
    span: int,
    warmup_period: int,
    B: int = 500,
    mean_block: int = 12,
    rng_seed: int = 17,
) -> pd.DataFrame:
    """
    Stationary block bootstrap (Politis-Romano 1994) for sign-constrained
    ridge coefficient inference.

    For B resamples, draw a block-stationary index sequence into the
    aligned (y, X) panel, refit the signed-ridge regression, and record
    the beta vector. Returns a (B × p) DataFrame.

    Why stationary blocks (not i.i.d.). Quarterly returns are
    autocorrelated even after AR(1) unsmoothing — credit cycles persist
    well beyond one quarter, regime states span multiple years, and the
    (y, X) joint distribution is not stationary across 2008/2020
    breakpoints. Stationary blocks preserve this short-run dependence
    by resampling contiguous segments rather than individual obs.

    NaN handling. The regression-input panel (y, X) is already aligned
    and pooled — per-vintage NaNs (vintages alive only during their
    lifespan) were absorbed at the contribution-weighted pooling step
    upstream. Pooled y_t at quarter t reflects whichever vintages are
    active that quarter, weighted by their active capital. Vintages
    not yet launched or fully wound down get zero weight automatically.
    Bootstrap operates on a clean (n × p+1) matrix.

    Why we do NOT bootstrap per-vintage paths independently. Vintages
    alive at the same quarter share factor exposure (e.g. 2008Q4 hits
    the fund IV/V/VI/VII/VIIb simultaneously). Independent resampling
    breaks this cross-section correlation, the very structure the
    regression identifies β from. The correct vintage-level bootstrap
    is to resample the time index jointly across all vintages AND the
    factor panel — but for the pooling-first pipeline this collapses
    to the bootstrap of the pooled (y, X) panel implemented here.

    References:
        Politis & Romano (1994, JASA) — stationary bootstrap.
        Chatterjee & Lahiri (2011, Ann. Stat.) — LASSO bootstrap consistency.
        Korteweg-Nagel (2024, Appendix C) — fund-level alpha bootstrap analog.
    """
    aligned = pd.concat([y, X], axis=1).dropna()
    if len(aligned) <= warmup_period:
        raise ValueError("insufficient observations for bootstrap")
    aligned = aligned.iloc[warmup_period:].copy()
    n = len(aligned)
    cols = list(aligned.columns[1:])
    rng = np.random.default_rng(rng_seed)

    samples = np.zeros((B, len(cols)))
    for b in range(B):
        idx = stationary_block_indices(n, mean_block=mean_block, rng=rng)
        sub = aligned.iloc[idx]
        y_b = sub.iloc[:, 0]
        X_b = sub.iloc[:, 1:]
        try:
            fit_b = fit_signed_ridge(
                y_b, X_b, sign_mask=sign_mask, prior_mean=prior_mean,
                lam=lam, span=span, warmup_period=0,  # warmup already applied
            )
            samples[b, :] = fit_b.beta.values
        except (RuntimeError, ValueError):
            samples[b, :] = np.nan
    return pd.DataFrame(samples, columns=cols)


def beta_significance_table(
    central: pd.Series,
    boot: pd.DataFrame,
    sign_mask: Dict[str, str],
    prior_mean: Dict[str, float],
) -> pd.DataFrame:
    """
    Build a coefficient-significance summary from the bootstrap distribution.

    Columns:
        central       : point estimate (the in-sample β)
        boot_mean     : bootstrap mean
        boot_se       : bootstrap standard error
        boot_lo / hi  : 5%/95% quantile bounds
        frac_at_bound : fraction of resamples where the sign-mask
                        constraint is binding (β within 1e-6 of zero
                        for masked factors). Closer to 1 ⇒ less significant.
        p_two_sided   : 2 × min(P(β ≤ 0), P(β ≥ 0)) — analog of a two-sided
                        p-value vs the null β = 0. Conservative under
                        sign constraints (treats boundary mass as
                        "evidence for the null"). For UNCONSTRAINED
                        factors this is the standard bootstrap p-value.
    """
    rows = []
    for c in central.index:
        b = boot[c].dropna().values
        if len(b) == 0:
            rows.append({'factor': c, 'central': central[c]})
            continue
        sign = sign_mask.get(c, 'free')
        at_bound = float(np.mean(np.abs(b) < 1e-6))
        # Two-sided "p-value" — fraction of resamples on the wrong side of zero.
        # For 'zero' (excluded by analyst), the coefficient is not estimated;
        # we report NaN to flag this rather than a meaningless number.
        if sign == 'zero':
            p_val = float('nan')
        elif sign == 'pos':
            p_val = 2.0 * np.mean(b <= 1e-6)  # mass at or below zero
        elif sign == 'neg':
            p_val = 2.0 * np.mean(b >= -1e-6)
        else:
            p_val = 2.0 * min(np.mean(b <= 0), np.mean(b >= 0))
        rows.append({
            'factor': c,
            'sign_mask': sign,
            'prior_mean': prior_mean.get(c, 0.0),
            'central': central[c],
            'boot_mean': float(np.mean(b)),
            'boot_se': float(np.std(b, ddof=1)),
            'boot_lo': float(np.quantile(b, 0.05)),
            'boot_hi': float(np.quantile(b, 0.95)),
            'frac_at_bound': at_bound,
            'p_two_sided': float(min(p_val, 1.0)) if not pd.isna(p_val) else float('nan'),
        })
    return pd.DataFrame(rows).set_index('factor')


def factor_log_levels_panel(log_ret: pd.DataFrame) -> pd.DataFrame:
    """Cumulative log levels of factors (starting at 0 at series start)."""
    return log_ret.cumsum()



    """Cumulative log levels of factors (starting at 0 at series start)."""
    return log_ret.cumsum()


def horizon_log_factor(t0: pd.Timestamp, t: pd.Timestamp,
                       log_levels: pd.DataFrame) -> np.ndarray:
    """Return r_h vector for horizon (t0, t]: cumulative log return of each factor."""
    L = log_levels.sort_index()
    L0 = L.asof(t0)
    Lt = L.asof(t)
    if isinstance(L0, pd.Series) and isinstance(Lt, pd.Series):
        return (Lt - L0).values
    return np.full(L.shape[1], np.nan)


def horizon_log_market(t0: pd.Timestamp, t: pd.Timestamp,
                       log_levels_eq: pd.Series) -> float:
    """Single-factor (Equity) log market return over horizon."""
    L = log_levels_eq.sort_index()
    return float(L.asof(t) - L.asof(t0))


def horizon_log_rf(t0: pd.Timestamp, t: pd.Timestamp,
                   rf_q: pd.Series) -> float:
    """
    Cumulative log risk-free return between t0 and t. rf_q is the
    quarterly simple yield series; sum of log(1+rf_q) over completed quarters.
    """
    qends = rf_q.index
    mask = (qends > t0) & (qends <= t)
    chunk = rf_q.loc[mask]
    if chunk.empty:
        return 0.0
    return float(np.log1p(chunk).sum())


def kn24_single_factor_deflator(
    cf_dates: List[pd.Timestamp],
    t0: pd.Timestamp,
    log_levels_eq: pd.Series,
    rf_q: pd.Series,
    beta: float,
    sigma2_h_per_year: float,
) -> List[float]:
    """
    R_h^b = exp{ rf_h + beta*(r_m_h - rf_h) - 0.5*beta*(beta-1)*sigma_h^2 }
    sigma_h^2 scales linearly with horizon h years.
    """
    out = []
    for t in cf_dates:
        h_years = max((t - t0).days, 1) / 365.25
        rfh = horizon_log_rf(t0, t, rf_q)
        rmh = horizon_log_market(t0, t, log_levels_eq)
        sig2 = sigma2_h_per_year * h_years
        rb = rfh + beta * (rmh - rfh) - 0.5 * beta * (beta - 1) * sig2
        out.append(np.exp(rb))
    return out


def matf_multi_factor_deflator(
    cf_dates: List[pd.Timestamp],
    t0: pd.Timestamp,
    log_levels: pd.DataFrame,      # cumulative log levels of factors (excess)
    rf_q: pd.Series,
    beta_vec: np.ndarray,
    Sigma_h_q: np.ndarray,         # quarterly factor covariance
) -> List[float]:
    """
    R_h^{b,MATF} = exp{ rf_h + beta'r_h
                        + 0.5 * beta'diag(Sigma_h)
                        - 0.5 * beta' Sigma_h beta }
    All factor inputs are EXCESS log returns. Sigma_h_q scales by h_quarters.
    """
    diag = np.diag(Sigma_h_q)
    out = []
    for t in cf_dates:
        h_years = max((t - t0).days, 1) / 365.25
        h_q = h_years * 4.0
        rfh = horizon_log_rf(t0, t, rf_q)
        rh = horizon_log_factor(t0, t, log_levels)
        if np.any(np.isnan(rh)):
            out.append(np.nan)
            continue
        Sig_h = Sigma_h_q * h_q
        diag_h = diag * h_q
        rb = (rfh + beta_vec @ rh
              + 0.5 * beta_vec @ diag_h
              - 0.5 * beta_vec @ Sig_h @ beta_vec)
        out.append(float(np.exp(rb)))
    return out


def kn16_gpme_sdf(
    cf_dates: List[pd.Timestamp],
    t0: pd.Timestamp,
    log_levels_eq: pd.Series,
    rf_q: pd.Series,
    delta: float,
    gamma: float,
) -> List[float]:
    """
    KN16 SDF: M_h = exp(delta*h - gamma*r_h^m).
    Returns 1/M_h so it can be used as a deflator interchangeably.
    """
    out = []
    for t in cf_dates:
        h_years = max((t - t0).days, 1) / 365.25
        rmh = horizon_log_market(t0, t, log_levels_eq)
        Mh = np.exp(delta * h_years - gamma * rmh)
        out.append(1.0 / Mh)
    return out


# =============================================================================
# Per-vintage alpha computation
# =============================================================================

def vintage_alpha_and_da(
    cf_g: pd.DataFrame,
    rvpi_nav: float,
    rvpi_date: pd.Timestamp,
    deflators: List[float],
    cf_dates_with_terminal: List[pd.Timestamp],
) -> Tuple[float, float]:
    """
    Compute fund-level alpha (NPV under the deflator) and Direct Alpha
    (annualized excess return).

    deflators length must equal len(cf_dates_with_terminal).
    cf flows are negative for contributions, positive for distributions.
    The NAV is appended as a synthetic terminal distribution.
    """
    amts = list(cf_g['amount'].values) + [rvpi_nav]
    if len(amts) != len(deflators):
        raise ValueError("CF / deflator length mismatch")

    # alpha = NPV of deflated flows (negative contrib -> deflated outflow).
    deflated = [a / d if (d == d and d > 0) else np.nan
                for a, d in zip(amts, deflators)]
    alpha = float(np.nansum(deflated))

    # Direct Alpha: find delta s.t. NPV(deflated * exp(-delta * h)) = 0.
    t0 = cf_dates_with_terminal[0]
    hs = np.array([(t - t0).days / 365.25 for t in cf_dates_with_terminal])

    def npv_da(da_log: float) -> float:
        return float(np.nansum(np.array(deflated) * np.exp(-da_log * hs)))

    try:
        if npv_da(-1.0) * npv_da(1.0) < 0:
            da_log = brentq(npv_da, -1.0, 1.0, xtol=1e-8, maxiter=200)
            da = np.exp(da_log) - 1.0
        else:
            da = float('nan')
    except (ValueError, RuntimeError):
        da = float('nan')

    return alpha, da


def cf_with_terminal_for_vintage(
    cf_g: pd.DataFrame,
    nav_g: pd.DataFrame,
    asof: pd.Timestamp,
) -> Tuple[pd.DataFrame, float, pd.Timestamp, List[pd.Timestamp]]:
    """Standardize CF + NAV input for one vintage."""
    cf_g = cf_g.sort_values('date').copy()
    if len(nav_g):
        last = nav_g.sort_values('date').iloc[-1]
        rvpi_nav = float(last['nav'])
        rvpi_date = pd.Timestamp(last['date'])
    else:
        rvpi_nav = 0.0
        rvpi_date = cf_g['date'].max()
    T = max(rvpi_date, cf_g['date'].max())
    cf_dates = list(pd.to_datetime(cf_g['date']).values) + [T]
    cf_dates = [pd.Timestamp(d) for d in cf_dates]
    return cf_g, rvpi_nav, rvpi_date, cf_dates


# =============================================================================
# KN24 single-factor beta GMM identification
# =============================================================================

def kn24_capm_beta(
    vintages: List[Tuple[pd.DataFrame, pd.DataFrame]],
    asof: pd.Timestamp,
    log_levels_eq: pd.Series,
    rf_q: pd.Series,
    sigma2_eq_per_year: float,
    delta: float,
    gamma: float,
) -> Tuple[float, float]:
    """
    KN24 eq.(17): find beta that minimizes cross-sectional dispersion of
    fund-level alpha subject to mean alpha = mean GPME (alpha_bar from SDF).

    Implementation: scan beta in a grid; the constraint is the average alpha
    target, so we shift each beta's alphas by the constant (alpha_bar - mean(alpha_at_beta))
    before computing dispersion. The beta that minimizes cross-sectional variance
    is the right root (vs the gamma-root that the GPME also satisfies).
    """
    # First compute alpha_bar (mean GPME)
    gpmes = []
    for cf_g, nav_g in vintages:
        cf_v, rvpi_nav, _, dates = cf_with_terminal_for_vintage(cf_g, nav_g, asof)
        defl = kn16_gpme_sdf(dates, dates[0], log_levels_eq, rf_q, delta, gamma)
        a, _ = vintage_alpha_and_da(cf_v, rvpi_nav, dates[-1], defl, dates)
        gpmes.append(a)
    gpmes = np.array(gpmes, dtype=float)
    alpha_bar = float(np.nanmean(gpmes))

    # Search beta in [0.2, 4.5] — smaller-than-gamma root expected for buyout/credit
    beta_grid = np.linspace(0.2, 4.5, 87)
    best_beta = float('nan')
    best_var = np.inf
    for beta in beta_grid:
        alphas = []
        for cf_g, nav_g in vintages:
            cf_v, rvpi_nav, _, dates = cf_with_terminal_for_vintage(cf_g, nav_g, asof)
            defl = kn24_single_factor_deflator(
                dates, dates[0], log_levels_eq, rf_q, beta, sigma2_eq_per_year)
            a, _ = vintage_alpha_and_da(cf_v, rvpi_nav, dates[-1], defl, dates)
            alphas.append(a)
        alphas = np.array(alphas, dtype=float)
        # Centre at alpha_bar, then take dispersion
        if np.all(np.isnan(alphas)):
            continue
        v = float(np.nanvar(alphas, ddof=1))
        if v < best_var:
            best_var = v
            best_beta = beta
    return best_beta, alpha_bar


def kn16_sdf_params(
    log_levels_eq: pd.Series,
    rf_q: pd.Series,
) -> Tuple[float, float, float]:
    """
    Pin down (delta, gamma) by KN16 eq.(6):
        gamma = mu / sigma^2,
        delta = -rf - 0.5 gamma^2 sigma^2 + gamma (rf + mu - 0.5 sigma^2)
    where mu = log E[R_1^m] - rf, sigma = sqrt var(r_1^m), all annualized.
    Returns (delta, gamma, sigma2_annual).
    """
    qret = log_levels_eq.diff().dropna()
    sigma2_q = float(qret.var(ddof=1))
    sigma2_y = sigma2_q * 4.0
    rf_mean_q = float(rf_q.mean())     # mean quarterly simple yield
    rf_y = rf_mean_q * 4.0             # annualized
    # E[R_1^m] = exp(mean(r) + 0.5 sigma^2) per year
    mu_log_y = float(qret.mean()) * 4.0
    E_Rm = np.exp(mu_log_y + 0.5 * sigma2_y)
    mu = np.log(E_Rm) - rf_y
    gamma = mu / sigma2_y
    delta = -rf_y - 0.5 * gamma**2 * sigma2_y + gamma * (rf_y + mu - 0.5 * sigma2_y)
    return delta, gamma, sigma2_y


# =============================================================================
# Bootstrap CI (KN24 Appendix C style, simplified)
# =============================================================================

def matf_deflator_fast(
    cf_dates: List[pd.Timestamp],
    t0: pd.Timestamp,
    cum_log_factor_np: np.ndarray,   # (T, M) cumulative log factor returns
    cum_log_rf_np: np.ndarray,       # (T,) cumulative log(1+rf)
    quarters_ns: np.ndarray,         # (T,) quarter-end timestamps as int64 ns
    beta_vec: np.ndarray,            # (M,)
    Sigma_h_q: np.ndarray,           # (M, M) quarterly factor covariance
) -> np.ndarray:
    """
    Vectorized MATF benchmark-portfolio deflator at multiple cash flow
    dates. Equivalent to ``matf_multi_factor_deflator`` but using
    precomputed cumulative arrays + numpy searchsorted, which is roughly
    100x faster for the bootstrap inner loop.
    """
    diag_h_q = np.diag(Sigma_h_q)
    quad = float(beta_vec @ Sigma_h_q @ beta_vec)
    n_q = len(quarters_ns)

    cf_ns = np.array([pd.Timestamp(t).value for t in cf_dates], dtype=np.int64)
    t0_ns = pd.Timestamp(t0).value

    # 'asof' semantics: largest q <= date  → np.searchsorted side='right' minus 1
    idx_t = np.clip(np.searchsorted(quarters_ns, cf_ns, side='right') - 1,
                    0, n_q - 1)
    idx_t0 = int(np.clip(np.searchsorted(quarters_ns, t0_ns, side='right') - 1,
                         0, n_q - 1))

    rh = cum_log_factor_np[idx_t] - cum_log_factor_np[idx_t0]   # (n_cf, M)
    rfh = cum_log_rf_np[idx_t] - cum_log_rf_np[idx_t0]           # (n_cf,)

    h_years = np.array([max((pd.Timestamp(t) - pd.Timestamp(t0)).days, 1) / 365.25
                        for t in cf_dates])
    h_q = h_years * 4.0

    rb = (rfh
          + rh @ beta_vec
          + 0.5 * h_q * (beta_vec @ diag_h_q)
          - 0.5 * h_q * quad)
    return np.exp(rb)


def vintage_da_fast(
    cf_v: pd.DataFrame,
    rvpi_nav: float,
    cf_dates: List[pd.Timestamp],
    deflators: np.ndarray,
) -> float:
    """Direct Alpha from precomputed deflators — vectorized brentq driver."""
    amts = np.concatenate([cf_v['amount'].values, [rvpi_nav]])
    if len(amts) != len(deflators):
        return float('nan')
    valid = (deflators == deflators) & (deflators > 0)
    if not valid.all():
        return float('nan')
    deflated = amts / deflators

    t0 = cf_dates[0]
    hs = np.array([(pd.Timestamp(t) - pd.Timestamp(t0)).days / 365.25
                   for t in cf_dates])

    def npv_da(da_log: float) -> float:
        return float(np.sum(deflated * np.exp(-da_log * hs)))

    try:
        if npv_da(-1.0) * npv_da(1.0) < 0:
            da_log = brentq(npv_da, -1.0, 1.0, xtol=1e-7, maxiter=100)
            return np.exp(da_log) - 1.0
    except (ValueError, RuntimeError):
        pass
    return float('nan')



def per_vintage_alpha_bootstrap(
    cf: pd.DataFrame,
    navs: pd.DataFrame,
    asof: pd.Timestamp,
    log_ret_q: pd.DataFrame,           # T × M factor log excess returns
    pooled_log_y: pd.Series,           # pooled y_t (excess, log, AR-unsmoothed)
    rf_q: pd.Series,                   # quarterly simple rf yield
    sign_mask: Dict[str, str],
    prior_mean: Dict[str, float],
    lam: float,
    span: int,
    warmup_period: int,
    B: int = 500,
    mean_block: int = 12,
    rng_seed: int = 19,
) -> pd.DataFrame:
    """
    Stationary block bootstrap of per-vintage Direct Alpha under MATF-α.

    Method (whole-panel, joint resampling — the methodologically correct
    approach for this estimator):

        For each resample b = 1, ..., B:
          1. Draw a single stationary-block time index ι^(b) of length T
             over the quarterly panel (Politis-Romano with mean_block=12q).
          2. Apply ι^(b) jointly to:
             - factor return panel X (so factor cross-section correlation
               at any "resampled time" matches the original)
             - pooled y series
             - rf series
          3. Refit β^(b) via factorlasso on the resampled (y^(b), X^(b)).
          4. Build resampled cumulative log factor levels L^(b) = cumsum(X^(b))
             and resampled rf^(b).
          5. For each vintage i with cash flows {(t_j, X_{i,j})}:
                R^(b)_{h(j)} = exp{rf^(b)_h + β^(b)' r^(b)_h
                                + Jensen corrections}
                α^(b)_i      = NPV of cash flows under R^(b)
                DA^(b)_i     = Direct Alpha (annualized) solving NPV = 0

    Returns
    -------
    DataFrame (vintage × B) of bootstrap Direct Alpha samples.

    Why whole-panel, not per-vintage:
        Vintages alive at the same calendar quarter share factor exposure
        (e.g. the fund IV/V/VI/VII/VIIb all hit by 2008-Q4 simultaneously).
        Independent per-vintage resampling generates impossible counter-
        factuals — one vintage gets the equity crash while another at
        the same calendar quarter doesn't — which destroys the cross-
        section correlation structure the regression identifies β from.
        The whole-panel scheme preserves it: vintages co-active in the
        ORIGINAL panel remain co-active in the same resampled blocks.
    """
    # Align all quarterly inputs
    common_idx = (log_ret_q.index
                  .intersection(pooled_log_y.index)
                  .intersection(rf_q.index))
    if len(common_idx) <= warmup_period + 4:
        raise ValueError(f"Insufficient common observations: {len(common_idx)}")
    X = log_ret_q.loc[common_idx]
    y = pooled_log_y.loc[common_idx]
    rf = rf_q.loc[common_idx]
    T = len(common_idx)
    quarters = common_idx

    vintages = sorted(cf['vintage_label'].unique())
    samples = pd.DataFrame(np.nan, index=vintages, columns=range(B), dtype=float)

    # Pre-extract per-vintage CF/NAV (fixed across all bootstrap iterations)
    vintage_inputs = []
    for v in vintages:
        cf_g = cf[cf['vintage_label'] == v]
        nav_g = navs[navs['vintage_label'] == v]
        cf_v, rvpi_nav, _, dates = cf_with_terminal_for_vintage(cf_g, nav_g, asof)
        vintage_inputs.append((v, cf_v, rvpi_nav, dates))

    quarters_ns = np.array([q.value for q in quarters], dtype=np.int64)

    rng = np.random.default_rng(rng_seed)
    n_failed = 0
    for b in range(B):
        iota = stationary_block_indices(T, mean_block=mean_block, rng=rng)

        # Joint resample using SAME time index on every panel
        y_b = pd.Series(y.values[iota], index=quarters, name=y.name)
        X_b = pd.DataFrame(X.values[iota], index=quarters, columns=X.columns)
        rf_b_vals = rf.values[iota]

        # Refit β
        try:
            fit_b = fit_signed_ridge(
                y_b, X_b, sign_mask=sign_mask, prior_mean=prior_mean,
                lam=lam, span=span, warmup_period=warmup_period,
            )
        except (RuntimeError, ValueError):
            n_failed += 1
            continue

        beta_b = fit_b.beta.values
        Sigma_h_b = fit_b.Sigma_h.values

        # Precompute cumulative arrays once per bootstrap iteration
        cum_log_factor_np = X_b.values.cumsum(axis=0)   # (T, M)
        cum_log_rf_np = np.log1p(rf_b_vals).cumsum()    # (T,)

        # Per-vintage Direct Alpha using fast vectorized deflator
        for v, cf_v, rvpi_nav, dates in vintage_inputs:
            try:
                defl = matf_deflator_fast(
                    dates, dates[0], cum_log_factor_np, cum_log_rf_np,
                    quarters_ns, beta_b, Sigma_h_b)
                da = vintage_da_fast(cf_v, rvpi_nav, dates, defl)
                samples.loc[v, b] = da
            except (RuntimeError, ValueError):
                samples.loc[v, b] = np.nan

    if n_failed > 0:
        print(f"  [bootstrap] {n_failed}/{B} resamples failed at β fit")
    return samples



def bootstrap_alpha_ci(
    cf: pd.DataFrame,
    navs: pd.DataFrame,
    asof: pd.Timestamp,
    deflator_fn,
    B: int = 1000,
    rng_seed: int = 17,
) -> Dict[str, Tuple[float, float]]:
    """
    Resample vintages with replacement, recompute the cross-sectional alpha
    distribution, return 5/95% CI bounds for each vintage's central alpha.

    For practical scale: bootstrap the distribution of CROSS-SECTIONAL
    alpha after recentering, return the (5,95) bounds as 'noise band' for
    every vintage. This is the simplification noted in the design doc.
    """
    rng = np.random.default_rng(rng_seed)
    funds = cf['vintage_label'].unique()

    base_alphas = {}
    for f in funds:
        cf_g = cf[cf['vintage_label'] == f]
        nav_g = navs[navs['vintage_label'] == f]
        cf_v, rvpi_nav, _, dates = cf_with_terminal_for_vintage(cf_g, nav_g, asof)
        defl = deflator_fn(dates)
        a, _ = vintage_alpha_and_da(cf_v, rvpi_nav, dates[-1], defl, dates)
        base_alphas[f] = a

    base_arr = np.array(list(base_alphas.values()), dtype=float)
    base_arr = base_arr[~np.isnan(base_arr)]
    centered = base_arr - np.nanmean(base_arr)
    boot_means = []
    boot_disp = []
    for _ in range(B):
        s = rng.choice(centered, size=len(centered), replace=True)
        boot_means.append(s.mean())
        boot_disp.append(s.std(ddof=1))

    sigma_alpha = float(np.std(centered, ddof=1))
    out = {}
    for f, a in base_alphas.items():
        out[f] = (a - 1.645 * sigma_alpha, a + 1.645 * sigma_alpha)
    out['__sigma_alpha__'] = (float(np.percentile(boot_disp, 5)),
                              float(np.percentile(boot_disp, 95)))
    return out


# =============================================================================
# Driver
# =============================================================================

def run(cfg: Cfg) -> Dict:
    cfg.OUT.mkdir(parents=True, exist_ok=True)
    asof = pd.Timestamp(cfg.ASOF)
    print("=" * 78)
    print(" MATF-alpha pipeline, post-2000 universe")
    print("=" * 78)

    # ------------------------------------------------------------- data
    cf = load_cash_flows(cfg.DATA)
    navs = load_navs(cfg.DATA)
    print(f"  Loaded {len(cf):,} CF rows, {len(navs):,} NAV rows")

    # Filter to vintages with first call >= ANALYSIS_START
    start = pd.Timestamp(cfg.ANALYSIS_START)
    first_call = cf.groupby('vintage_label')['date'].min()
    keep = first_call[first_call >= start].index.tolist()
    cf = cf[cf['vintage_label'].isin(keep)].copy()
    navs = navs[navs['vintage_label'].isin(keep)].copy()
    print(f"  {len(keep)} vintages with first call >= {start.date()}: {keep}")

    # ----------------------------------------------------- factor inputs
    levels = load_factor_levels(cfg)
    levels = levels.loc[levels.index >= '1999-04-01']  # actual start of motion
    log_ret_q = factor_quarterly_log_returns(levels)
    log_ret_q = log_ret_q.loc[log_ret_q.index >= start]
    log_levels_q = factor_log_levels_panel(log_ret_q)
    print(f"  MATF factor panel: {log_ret_q.shape[0]} quarters, "
          f"{log_ret_q.index[0].date()} -> {log_ret_q.index[-1].date()}")

    # Risk-free
    rf_q = load_rf_quarterly(cfg.RF_CSV, start, asof + pd.Timedelta(days=400))
    print(f"  USD RF panel: {rf_q.shape[0]} quarters from CSV ({cfg.RF_CSV.name}), "
          f"mean ann. rate {(rf_q.mean()*4)*100:.2f}%")

    # ----------------------------------------------------- pooled returns
    ret_panel, cap_panel = per_vintage_quarterly_returns(cf, navs)
    pooled_smoothed_total = pool_manager_returns(ret_panel, cap_panel)
    pooled_smoothed_total = pooled_smoothed_total.loc[pooled_smoothed_total.index >= start]
    print(f"  Pooled smoothed (total) series: {pooled_smoothed_total.shape[0]} quarters")

    # Convert to EXCESS returns by subtracting quarterly rf — matches MATF
    # CMA pipeline convention (factor returns are already excess; both sides
    # of the regression must be on the same footing).
    rf_aligned = rf_q.reindex(pooled_smoothed_total.index).ffill().fillna(0.0)
    pooled_smoothed = pooled_smoothed_total - rf_aligned
    print(f"  Subtracted rf to get excess pooled series "
          f"(mean rf adj per quarter: {rf_aligned.mean():.4%})")

    # AR(q) Getmansky-Lo-Makarov unsmoothing — forced through a regular
    # quarter-end grid because qis estimates AR coefficients on the
    # observed series.
    qgrid = pd.date_range(pooled_smoothed.index.min().to_period('Q').to_timestamp(how='end').normalize(),
                          pooled_smoothed.index.max(), freq='QE')
    pooled_aligned = pooled_smoothed.reindex(qgrid).ffill().fillna(0.0)
    pooled_unsmoothed, glm_diag = qis.unsmooth_returns_glm(
        pooled_aligned, ar_order=cfg.AR_ORDER, return_diagnostics=True)
    theta_arr = np.atleast_1d(np.asarray(glm_diag.theta).flatten())
    print(f"  AR({cfg.AR_ORDER}) unsmoothing: theta={[round(float(x),3) for x in theta_arr]} "
          f"vol_inflation={glm_diag.vol_inflation_factor:.3f}")

    # Convert pooled simple return to LOG return for regression vs factor LOG returns
    pooled_log_unsmoothed = np.log1p(pooled_unsmoothed.clip(lower=-0.99))

    # ----------------------------------------------------- HCGL beta (signed ridge)
    # X: factor EXCESS log returns (factor returns are already excess by construction
    # in the MATF model — vol-targeted basket of futures financed at rf).
    Xq = log_ret_q.loc[log_ret_q.index >= start]
    common_idx = Xq.index.intersection(pooled_log_unsmoothed.index)
    yq = pooled_log_unsmoothed.loc[common_idx].dropna()
    Xq_aligned = Xq.loc[yq.index]
    fit = fit_signed_ridge(yq, Xq_aligned,
                           sign_mask=cfg.SIGN_MASK,
                           prior_mean=cfg.PRIOR_MEAN,
                           lam=cfg.RIDGE_LAMBDA,
                           span=cfg.BETAS_SPAN_Q,
                           warmup_period=cfg.WARMUP_PERIOD_Q)
    print(f"\n  Production HCGL beta vector (factorlasso GROUP_LASSO_CLUSTERS, "
          f"R^2={fit.r2:.3f}, n={fit.n_obs}, "
          f"lam={cfg.RIDGE_LAMBDA:.0e}, span={cfg.BETAS_SPAN_Q}q, "
          f"warmup={cfg.WARMUP_PERIOD_Q}q):")
    for c, b in fit.beta.items():
        sign = cfg.SIGN_MASK[c]
        marker = ('+' if sign == 'pos' else
                  '-' if sign == 'neg' else
                  '0' if sign == 'zero' else ' ')
        prior = cfg.PRIOR_MEAN[c]
        prior_str = f' (prior={prior:+.2f})' if prior != 0.0 else ''
        print(f"    [{marker}] {c:18s}  {b:+.4f}{prior_str}")

    # ----------------------------------------------------- bootstrap betas
    print(f"\n  Beta stationary block bootstrap "
          f"(B={cfg.BOOT_B_BETA}, mean_block={cfg.MEAN_BLOCK_Q}q)...")
    boot_betas = bootstrap_beta(
        yq, Xq_aligned, sign_mask=cfg.SIGN_MASK, prior_mean=cfg.PRIOR_MEAN,
        lam=cfg.RIDGE_LAMBDA, span=cfg.BETAS_SPAN_Q,
        warmup_period=cfg.WARMUP_PERIOD_Q, B=cfg.BOOT_B_BETA,
        mean_block=cfg.MEAN_BLOCK_Q,
    )
    sig_table = beta_significance_table(
        fit.beta, boot_betas, cfg.SIGN_MASK, cfg.PRIOR_MEAN)
    print("\n  Beta significance (stationary block bootstrap):")
    cols_to_show = ['sign_mask', 'central', 'boot_mean', 'boot_se',
                    'boot_lo', 'boot_hi', 'frac_at_bound', 'p_two_sided']
    print(sig_table[cols_to_show].round(4).to_string())

    # ============================================================
    #  Route A: Multi-vintage panel with single-group GROUP_LASSO
    #  Compare against Route B (pooled, the production default).
    #
    #  Production conclusion: B is the production estimator.
    #  A is a secondary diagnostic — its small-sample issues with
    #  recent vintages mean it doesn't add identification power at
    #  λ_prod = 1e-5; using B as prior pulls A toward B at sensible
    #  λ levels but doesn't generate new information.
    # ============================================================
    print("\n" + "=" * 78)
    print(" Route A vs B comparison")
    print("=" * 78)

    # Build per-vintage excess unsmoothed log returns
    Y_panel = per_vintage_excess_unsmoothed(
        ret_panel, rf_q, ar_order=cfg.AR_ORDER)
    Y_panel = Y_panel.reindex(Xq_aligned.index)
    valid_vintages = Y_panel.dropna(how='all', axis=1).columns
    Y_panel = Y_panel[valid_vintages]
    obs_count = Y_panel.notna().sum().sort_values()
    print(f"  Per-vintage panel: {Y_panel.shape[0]}q × {Y_panel.shape[1]} vintages")
    print(f"  Min/Max obs per vintage: {obs_count.min()} ({obs_count.idxmin()})  "
          f"to {obs_count.max()} ({obs_count.idxmax()})")

    # Cap-weighted aggregator across vintages (committed capital)
    contribs = cf.groupby('vintage_label').apply(
        lambda x: -x.loc[x['amount'] < 0, 'amount'].sum(),
        include_groups=False).reindex(valid_vintages).fillna(0.0)
    w = contribs / contribs.sum()

    # A.1: production λ, zero priors — the small-sample blowup
    coef_A1, _ = fit_route_A_simple(
        Y_panel, Xq_aligned, sign_mask=cfg.SIGN_MASK,
        prior_mean_dict={c: 0.0 for c in cfg.FACTORS_ALL},
        lam=cfg.RIDGE_LAMBDA, span=cfg.BETAS_SPAN_Q,
        warmup_period=cfg.WARMUP_PERIOD_Q,
    )
    beta_A1_cw = (coef_A1.T @ w).rename('A1_cw_lam=1e-5')

    # A.2: B as prior, λ sensitivity
    print("\n  Route A.2 (B as prior) — λ sensitivity:")
    print(f"  {'Factor':18s} {'B':>8s}  "
          + "  ".join(f"{f'λ={lam:g}':>10s}" for lam in cfg.LAM_GRID_A))
    a2_results = {}
    for lam in cfg.LAM_GRID_A:
        coef, _ = fit_route_A_simple(
            Y_panel, Xq_aligned, sign_mask=cfg.SIGN_MASK,
            prior_mean_dict=fit.beta.to_dict(),
            lam=lam, span=cfg.BETAS_SPAN_Q, warmup_period=cfg.WARMUP_PERIOD_Q,
        )
        a2_results[lam] = (coef.T @ w)

    for f in cfg.FACTORS_ALL:
        row = f"  {f:18s} {fit.beta[f]:>+8.3f}  "
        for lam in cfg.LAM_GRID_A:
            row += f"{a2_results[lam][f]:>+10.3f}  "
        print(row)

    # Pick a sensible λ for the headline A.2 comparison
    lam_a_hl = cfg.LAM_HEADLINE_A
    coef_A2, _ = fit_route_A_simple(
        Y_panel, Xq_aligned, sign_mask=cfg.SIGN_MASK,
        prior_mean_dict=fit.beta.to_dict(),
        lam=lam_a_hl, span=cfg.BETAS_SPAN_Q, warmup_period=cfg.WARMUP_PERIOD_Q,
    )
    beta_A2_cw = (coef_A2.T @ w).rename(f'A2_cw_lam={lam_a_hl:g}')

    cmp = pd.concat([
        fit.beta.rename('B_pooled'),
        beta_A1_cw, beta_A2_cw,
    ], axis=1).round(4)
    print("\n  Headline beta comparison (cap-weighted across vintages for A):")
    print(cmp.to_string())

    # ----------------------------------------------------- KN16 / KN24 anchors
    log_levels_eq = log_levels_q['Equity']
    delta, gamma, sigma2_eq_y = kn16_sdf_params(log_levels_eq, rf_q)
    print(f"\n  KN16 SDF parameters:  delta={delta:.4f}, gamma={gamma:.3f}, "
          f"sigma2_eq_y={sigma2_eq_y:.4f}")

    # KN24 single-factor beta via GMM
    vintages_pairs = [(cf[cf['vintage_label'] == f], navs[navs['vintage_label'] == f])
                      for f in keep]
    print("  Solving KN24 GMM for single-factor beta...")
    kn24_beta, alpha_bar = kn24_capm_beta(
        vintages_pairs, asof, log_levels_eq, rf_q, sigma2_eq_y, delta, gamma)
    print(f"  KN24 single-factor beta_eq = {kn24_beta:.3f}  "
          f"(SDF gamma = {gamma:.3f}; alpha_bar = {alpha_bar:.3f})")

    # ----------------------------------------------------- per-vintage alphas
    print("\n" + "=" * 78)
    print(" Per-vintage alpha across four deflators")
    print("=" * 78)
    rows = []
    Sigma_h_q = fit.Sigma_h.values
    beta_vec = fit.beta.values

    for f in keep:
        cf_g = cf[cf['vintage_label'] == f]
        nav_g = navs[navs['vintage_label'] == f]
        cf_v, rvpi_nav, _, dates = cf_with_terminal_for_vintage(cf_g, nav_g, asof)
        contrib = -cf_v.loc[cf_v['amount'] < 0, 'amount'].sum()

        # KS-PME — equity index deflator, beta=1 implicit, no Jensen
        ks_def = [np.exp(horizon_log_market(dates[0], t, log_levels_eq)
                         + horizon_log_rf(dates[0], t, rf_q))
                  for t in dates]
        ks_alpha, ks_da = vintage_alpha_and_da(cf_v, rvpi_nav, dates[-1], ks_def, dates)

        # KN16 GPME — SDF reciprocal as deflator
        kn16_def = kn16_gpme_sdf(dates, dates[0], log_levels_eq, rf_q, delta, gamma)
        kn16_alpha, kn16_da = vintage_alpha_and_da(cf_v, rvpi_nav, dates[-1], kn16_def, dates)

        # KN24 single-factor benchmark portfolio
        kn24_def = kn24_single_factor_deflator(
            dates, dates[0], log_levels_eq, rf_q, kn24_beta, sigma2_eq_y)
        kn24_alpha, kn24_da = vintage_alpha_and_da(cf_v, rvpi_nav, dates[-1], kn24_def, dates)

        # MATF-alpha
        matf_def = matf_multi_factor_deflator(
            dates, dates[0], log_levels_q, rf_q, beta_vec, Sigma_h_q)
        privateassets.matf, matf_da = vintage_alpha_and_da(cf_v, rvpi_nav, dates[-1], matf_def, dates)

        rows.append({
            'vintage_label': f,
            'first_call': cf_g['date'].min(),
            'contributions': contrib,
            'rvpi_nav': rvpi_nav,
            'KS_alpha': ks_alpha, 'KS_DA': ks_da,
            'KN16_alpha': kn16_alpha, 'KN16_DA': kn16_da,
            'KN24_alpha': kn24_alpha, 'KN24_DA': kn24_da,
            'MATF_alpha': privateassets.matf, 'MATF_DA': matf_da,
        })

    res = pd.DataFrame(rows).sort_values('first_call').reset_index(drop=True)
    # Compute DPI per vintage early so it's available for downstream aggregation
    dpi = []
    for f in keep:
        cf_g = cf[cf['vintage_label'] == f]
        contrib = -cf_g.loc[cf_g['amount'] < 0, 'amount'].sum()
        distrib = cf_g.loc[cf_g['amount'] > 0, 'amount'].sum()
        dpi.append({'vintage_label': f, 'DPI': distrib / contrib if contrib > 0 else np.nan})
    res = res.merge(pd.DataFrame(dpi), on='vintage_label', how='left')

    show = res.copy()
    show['first_call'] = show['first_call'].dt.date
    num_cols = [c for c in show.columns if show[c].dtype.kind in 'fi']
    show[num_cols] = show[num_cols].round(4)
    print(show.to_string(index=False))

    # ----------------------------------------------------- bootstrap CI
    print("\n  Per-vintage MATF-α bootstrap (whole-panel block resampling)...")
    boot_da = per_vintage_alpha_bootstrap(
        cf=cf, navs=navs, asof=asof,
        log_ret_q=Xq_aligned, pooled_log_y=yq, rf_q=rf_q,
        sign_mask=cfg.SIGN_MASK, prior_mean=cfg.PRIOR_MEAN,
        lam=cfg.RIDGE_LAMBDA, span=cfg.BETAS_SPAN_Q,
        warmup_period=cfg.WARMUP_PERIOD_Q,
        B=cfg.BOOT_B_ALPHA, mean_block=cfg.MEAN_BLOCK_Q,
    )

    # Build per-vintage α CI table
    da_ci_rows = []
    for v in res['vintage_label']:
        b = boot_da.loc[v].dropna().values
        central = float(res.loc[res['vintage_label'] == v, 'MATF_DA'].iloc[0])
        if len(b) == 0:
            da_ci_rows.append({'vintage_label': v, 'central_DA': central,
                               'boot_mean': np.nan, 'boot_se': np.nan,
                               'lo_5': np.nan, 'hi_95': np.nan,
                               'p_pos_alpha': np.nan})
            continue
        da_ci_rows.append({
            'vintage_label': v,
            'central_DA': central,
            'boot_mean': float(np.mean(b)),
            'boot_se': float(np.std(b, ddof=1)),
            'lo_5': float(np.quantile(b, 0.05)),
            'hi_95': float(np.quantile(b, 0.95)),
            'p_pos_alpha': float(np.mean(b > 0)),  # frac of resamples with α > 0
        })
    da_ci_df = pd.DataFrame(da_ci_rows)

    # Aggregate manager-level Direct Alpha bootstrap (cap-weighted)
    contribs_full = res.set_index('vintage_label')['contributions']
    mature_mask = res.set_index('vintage_label')['DPI'] > cfg.DPI_MATURITY \
        if 'DPI' in res.columns else pd.Series(True, index=res['vintage_label'])
    mature_vintages = mature_mask.index[mature_mask].tolist()
    boot_agg = []
    for b in boot_da.columns:
        col = boot_da[b].reindex(mature_vintages).dropna()
        if len(col) == 0:
            continue
        w_v = contribs_full.reindex(col.index)
        if w_v.sum() <= 0:
            continue
        boot_agg.append(float((col * w_v).sum() / w_v.sum()))
    boot_agg = np.array(boot_agg)

    central_agg = float((da_ci_df.set_index('vintage_label').loc[mature_vintages,
                         'central_DA'] * contribs_full.reindex(mature_vintages)
                        ).sum() / contribs_full.reindex(mature_vintages).sum())

    print("\n  Per-vintage MATF Direct Alpha bootstrap CIs:")
    print(da_ci_df.set_index('vintage_label').round(4).to_string())
    print(f"\n  Cap-weighted aggregate MATF DA (mature, DPI > {cfg.DPI_MATURITY}):")
    print(f"    central          = {central_agg:.4f}")
    print(f"    bootstrap mean   = {np.mean(boot_agg):.4f}")
    print(f"    bootstrap SE     = {np.std(boot_agg, ddof=1):.4f}")
    print(f"    5/95 CI          = [{np.quantile(boot_agg, 0.05):.4f}, "
          f"{np.quantile(boot_agg, 0.95):.4f}]")
    print(f"    P(agg α > 0)     = {np.mean(boot_agg > 0):.4f}")

    # ----------------------------------------------------- cap-weighted aggregates
    # DPI already merged onto res earlier
    mature = res[res['DPI'] > cfg.DPI_MATURITY].copy()
    agg_rows = []
    for label, da_col in [('KS-PME', 'KS_DA'), ('KN16-GPME', 'KN16_DA'),
                          ('KN24-alpha (CAPM)', 'KN24_DA'),
                          ('MATF-alpha', 'MATF_DA')]:
        valid = mature.dropna(subset=[da_col])
        w = valid['contributions']
        cw_da = float((valid[da_col] * w).sum() / w.sum()) if w.sum() > 0 else float('nan')
        eq_da = float(valid[da_col].mean())
        agg_rows.append({
            'metric': label, 'n_mature_vintages': len(valid),
            'cap_wtd_DA': cw_da, 'equal_wtd_DA': eq_da,
        })
    agg = pd.DataFrame(agg_rows)
    print("\n" + "=" * 78)
    print(f" Cap-weighted Direct Alpha (mature vintages, DPI > {cfg.DPI_MATURITY})")
    print("=" * 78)
    print(agg.round(4).to_string(index=False))

    # ----------------------------------------------------- export
    excel_path = cfg.OUT / 'MATF_alpha.xlsx'
    pdf_path = cfg.OUT / 'MATF_alpha_tearsheet.pdf'

    with pd.ExcelWriter(excel_path, engine='openpyxl') as xw:
        res.to_excel(xw, sheet_name='Vintage Alphas', index=False)
        agg.to_excel(xw, sheet_name='Cap-Wtd Aggregates', index=False)
        # Beta sheet
        beta_df = fit.beta.to_frame('beta')
        beta_df['sign_mask'] = pd.Series(cfg.SIGN_MASK)
        beta_df['prior_mean'] = pd.Series(cfg.PRIOR_MEAN)
        beta_df['ann_factor_vol'] = np.sqrt(np.diag(fit.Sigma_h)) * 2.0  # qtly-> ann
        beta_df.to_excel(xw, sheet_name='MATF Beta')
        # Beta significance
        sig_table.to_excel(xw, sheet_name='Beta Significance')
        # Route A vs B comparison
        cmp.to_excel(xw, sheet_name='Route A vs B (headline)')
        coef_A1.to_excel(xw, sheet_name='Route A1 per-vintage')
        coef_A2.to_excel(xw, sheet_name=f'Route A2 per-vintage')
        # Lambda sensitivity for A.2
        a2_lam_df = pd.DataFrame({f'lam={lam:g}': a2_results[lam] for lam in cfg.LAM_GRID_A})
        a2_lam_df.insert(0, 'B_pooled', fit.beta)
        a2_lam_df.to_excel(xw, sheet_name='Route A2 lambda sweep')
        # Diagnostics
        theta_arr = np.atleast_1d(np.asarray(glm_diag.theta).flatten())
        diag = pd.DataFrame({
            'theta': theta_arr,
            'lag_label': [f'theta_{i}' for i in range(len(theta_arr))],
        })
        diag.to_excel(xw, sheet_name='Unsmoothing Diag', index=False)
        # Pooled returns
        pooled_df = pd.DataFrame({
            'pooled_smoothed': pooled_smoothed,
            'pooled_unsmoothed': pooled_unsmoothed.reindex(pooled_smoothed.index),
        })
        pooled_df.to_excel(xw, sheet_name='Pooled Returns')
        # Confidence intervals
        # Per-vintage Direct Alpha bootstrap CIs
        da_ci_df.to_excel(xw, sheet_name='Bootstrap CIs', index=False)
        # Aggregate row
        agg_ci = pd.DataFrame([{
            'metric': 'cap_wtd_aggregate_MATF_DA',
            'central': central_agg,
            'boot_mean': float(np.mean(boot_agg)),
            'boot_se': float(np.std(boot_agg, ddof=1)),
            'lo_5': float(np.quantile(boot_agg, 0.05)),
            'hi_95': float(np.quantile(boot_agg, 0.95)),
            'p_pos': float(np.mean(boot_agg > 0)),
            'B': len(boot_agg),
            'mean_block_q': cfg.MEAN_BLOCK_Q,
        }])
        agg_ci.to_excel(xw, sheet_name='Aggregate CI', index=False)
    print(f"\n  Saved: {excel_path}")

    # ----------------------------------------------------- plots
    with PdfPages(pdf_path) as pdf:
        # Page 1: HCGL beta
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bars = fit.beta.values
        cols = list(fit.beta.index)
        def _color(c):
            s = cfg.SIGN_MASK[c]
            return ('#2ca02c' if s == 'pos' else
                    '#d62728' if s == 'neg' else
                    '#ffd966' if s == 'zero' else
                    '#7f7f7f')
        colors = [_color(c) for c in cols]
        ax.bar(cols, bars, color=colors, edgecolor='k', lw=0.5)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel('Loading')
        ax.set_title(f'MATF-α HCGL beta vector '
                     f'(R²={fit.r2:.2f}, n={fit.n_obs})')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)
        # Color legend
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor='#2ca02c', label='sign-pos mask'),
                           Patch(facecolor='#d62728', label='sign-neg mask'),
                           Patch(facecolor='#ffd966', label='zero (excluded)'),
                           Patch(facecolor='#7f7f7f', label='unconstrained')],
                  loc='best', fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Page 2: Direct Alpha by vintage, four deflators side by side
        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(len(res))
        width = 0.2
        for i, (col, label, color) in enumerate([
            ('KS_DA', 'KS-PME', '#1f77b4'),
            ('KN16_DA', 'KN16 GPME', '#ff7f0e'),
            ('KN24_DA', 'KN24 (CAPM)', '#2ca02c'),
            ('MATF_DA', 'MATF-α', '#d62728'),
        ]):
            ax.bar(x + (i - 1.5) * width, res[col] * 100, width=width,
                   label=label, color=color, edgecolor='k', lw=0.3)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(res['vintage_label'], rotation=30, ha='right')
        ax.set_ylabel('Direct Alpha (%, ann.)')
        ax.set_title('Direct Alpha by Vintage, four deflators')
        ax.legend(loc='best', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Page 3: Pooled smoothed vs unsmoothed
        fig, ax = plt.subplots(figsize=(11, 4.5))
        cum_s = (1.0 + pooled_smoothed).cumprod() - 1
        cum_u = (1.0 + pooled_unsmoothed.reindex(pooled_smoothed.index).fillna(0.0)).cumprod() - 1
        ax.plot(cum_s.index, cum_s.values * 100, label='Smoothed (NAV-implied)',
                color='#1f77b4', lw=1.5)
        ax.plot(cum_u.index, cum_u.values * 100, label=f'Unsmoothed AR({cfg.AR_ORDER})',
                color='#d62728', lw=1.5)
        ax.set_ylabel('Cumulative return (%)')
        ax.set_title('Pooled manager-level quarterly returns: smoothed vs unsmoothed')
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Page 4: cap-weighted aggregate bar
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = agg['metric'].tolist()
        vals = (agg['cap_wtd_DA'].values * 100)
        eqvals = (agg['equal_wtd_DA'].values * 100)
        x = np.arange(len(labels))
        ax.bar(x - 0.2, vals, 0.4, label='Cap-weighted', color='#1f77b4')
        ax.bar(x + 0.2, eqvals, 0.4, label='Equal-weighted', color='#ff7f0e')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel('Direct Alpha (%, ann.)')
        ax.set_title(f'Aggregate Direct Alpha (mature vintages, DPI > {cfg.DPI_MATURITY})')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Page 5: Beta bootstrap distributions — coefficient significance
        fig, ax = plt.subplots(figsize=(11, 5.5))
        cols = list(fit.beta.index)
        positions = np.arange(len(cols))
        data_list = [boot_betas[c].dropna().values for c in cols]
        colors = [_color(c) for c in cols]
        parts = ax.violinplot(data_list, positions=positions, vert=False,
                              showmedians=True, widths=0.7)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.55)
        for key in ('cmedians', 'cbars', 'cmins', 'cmaxes'):
            if key in parts:
                parts[key].set_edgecolor('black')
                parts[key].set_linewidth(0.8)
        ax.scatter(fit.beta.values, positions, marker='D', s=55,
                   color='black', zorder=10, label='Central estimate')
        ax.axvline(0, color='k', lw=0.8, alpha=0.6)
        added_prior_label = False
        for i, c in enumerate(cols):
            mu = cfg.PRIOR_MEAN[c]
            if mu != 0:
                lbl = 'Prior mean' if not added_prior_label else ''
                ax.scatter(mu, i, marker='^', s=70, color='red',
                           zorder=11, label=lbl)
                added_prior_label = True
        ax.set_yticks(positions)
        ax.set_yticklabels(cols)
        ax.set_xlabel('Loading')
        ax.set_title(f'MATF-α beta bootstrap distribution '
                     f'(B = {cfg.BOOT_B_BETA}, stationary block, '
                     f'mean_block = {cfg.MEAN_BLOCK_Q}q)')
        ax.grid(axis='x', alpha=0.3)
        for i, c in enumerate(cols):
            p = sig_table.loc[c, 'p_two_sided']
            if pd.isna(p):
                continue
            note = f'p={p:.3f}'
            ax.text(1.02, i, note, transform=ax.get_yaxis_transform(),
                    ha='left', va='center', fontsize=8,
                    color='#333', bbox=dict(boxstyle='round,pad=0.2',
                                            facecolor='white',
                                            edgecolor='lightgray',
                                            alpha=0.8))
        ax.legend(loc='upper left', fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Page 6: Route A vs B β comparison
        fig, ax = plt.subplots(figsize=(11, 4.8))
        cols = list(fit.beta.index)
        x = np.arange(len(cols))
        width = 0.25
        ax.bar(x - width, fit.beta.values, width=width,
               color='#1f77b4', label='B (pooled, prod)', edgecolor='k', lw=0.4)
        ax.bar(x,         beta_A1_cw.values, width=width,
               color='#d62728', label='A.1 (panel, zero prior, λ=1e-5) — cap-wtd',
               edgecolor='k', lw=0.4)
        ax.bar(x + width, beta_A2_cw.values, width=width,
               color='#2ca02c', label=f'A.2 (panel, B as prior, λ={lam_a_hl:g}) — cap-wtd',
               edgecolor='k', lw=0.4)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(cols, rotation=30, ha='right')
        ax.set_ylabel('Loading')
        ax.set_title('Beta vector: Route B (pooled) vs Route A (multi-vintage panel)')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        # Cap A.1 visualization to keep readable
        if abs(beta_A1_cw).max() > 5:
            ax.set_ylim(-2.5, 4.0)
            ax.text(0.01, 0.97,
                    f'(A.1 visualization clipped — max |β| = {abs(beta_A1_cw).max():.1f}, '
                    'small-sample blowup)',
                    transform=ax.transAxes, fontsize=8, va='top',
                    color='#d62728', style='italic')
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Page 7: Per-vintage MATF-α with bootstrap CIs
        fig, ax = plt.subplots(figsize=(12, 5))
        order = res.sort_values('first_call')['vintage_label'].tolist()
        ci_df_ord = da_ci_df.set_index('vintage_label').loc[order]
        x = np.arange(len(order))
        central = ci_df_ord['central_DA'].values * 100
        lo = np.maximum(0, (ci_df_ord['central_DA'] - ci_df_ord['lo_5']).values) * 100
        hi = np.maximum(0, (ci_df_ord['hi_95'] - ci_df_ord['central_DA']).values) * 100
        # color by sign of central (green = positive, gray = ambiguous)
        colors = ['#2ca02c' if (ci_df_ord['p_pos_alpha'].iloc[i] > 0.95)
                  else '#7f7f7f' if (ci_df_ord['p_pos_alpha'].iloc[i] > 0.50)
                  else '#d62728' for i in range(len(order))]
        ax.bar(x, central, yerr=[lo, hi], capsize=4, color=colors,
               edgecolor='k', lw=0.5, label=None,
               error_kw={'ecolor': 'black', 'lw': 1.0})
        ax.axhline(0, color='k', lw=0.6)
        # Aggregate as horizontal line
        ax.axhline(central_agg * 100, color='#1f77b4', lw=1.5, linestyle='--',
                   alpha=0.7, label=f'Cap-wtd aggregate = {central_agg*100:.1f}% '
                   f'[{np.quantile(boot_agg,0.05)*100:.1f}%, '
                   f'{np.quantile(boot_agg,0.95)*100:.1f}%]')
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=30, ha='right')
        ax.set_ylabel('MATF Direct Alpha (%, ann.)')
        ax.set_title(f'Per-vintage MATF-α with bootstrap 5/95 CIs '
                     f'(B={cfg.BOOT_B_ALPHA}, whole-panel block, '
                     f'mean_block={cfg.MEAN_BLOCK_Q}q)')
        # Legend with color meaning
        from matplotlib.patches import Patch
        legend_elems = [
            Patch(facecolor='#2ca02c', label='P(α>0) > 95%'),
            Patch(facecolor='#7f7f7f', label='50% < P(α>0) < 95%'),
            Patch(facecolor='#d62728', label='P(α>0) < 50%'),
        ]
        # Add the agg line legend entry
        from matplotlib.lines import Line2D
        legend_elems.append(Line2D([0], [0], color='#1f77b4', lw=1.5,
                                   linestyle='--',
                                   label=f'Cap-wtd aggregate'))
        ax.legend(handles=legend_elems, loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    print(f"  Saved: {pdf_path}")
    return {
        'res': res, 'agg': agg, 'beta': fit.beta,
        'glm_diag': glm_diag, 'pooled': pooled_smoothed,
        'kn24_beta': kn24_beta, 'kn16_params': (delta, gamma),
    }


if __name__ == '__main__':
    cfg = Cfg()
    out = run(cfg)
