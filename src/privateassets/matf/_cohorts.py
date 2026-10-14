"""
privateassets.matf._cohorts — pooled-cohort cash-flow extracts.

A commercial cohort dataset pools every fund of one strategy raised in one
vintage year into a single quarterly path of paid-in capital ``c_t``,
distributions ``d_t`` and the residual valuation ``V_t``, each quoted as a
percent of committed capital (or in currency). The unit of observation is a
strategy x vintage cohort, a capital-weighted aggregate of the asset class, not a
fund. The same cohort appears once per percentile band of since-inception IRR,
and the full pool is the band spanning every fund.

This module reads such an extract into :class:`CohortCashFlows` and converts a
set of cohorts into the tidy ``(cf, navs)`` frames the rest of the package
consumes, with each cohort standing where a fund would. From there the chain is
the one every fund panel runs: :func:`~privateassets.matf.nav_implied_returns`,
:func:`~privateassets.matf.ks_pme`, :func:`~privateassets.matf.estimate_matf_alpha`.

Units and conventions. Flows are per-period amounts and the valuation is a level,
all dated at the period end as the extract reports them. Percent columns are
scaled to fractions of committed capital by ``scale``; a currency column is read
with ``scale=1.0``. In the tidy frames contributions are negative and
distributions positive, as everywhere in the package. A missing flow is zero, a
missing valuation is not a mark and is dropped rather than invented.

Entry points: :func:`load_cohort_extract`, :func:`build_cohorts`,
:func:`filter_mature`, :func:`cohorts_to_cash_flows`, :func:`cohort_summary`.

Fund-level readers live in ``_pme`` and nothing here estimates anything. No
input data ships with the package. See DATA_README.md.
"""

# packages
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
# qis / project
from privateassets.matf._pme import CF_COLUMNS, NAV_COLUMNS

# Column names of the tidy cohort frame load_cohort_extract returns.
COHORT_COLUMNS = ['group', 'band', 'vintage', 'num_funds', 'date', 'paid_in', 'dist', 'nav']

# Band label under which a cohort pools every fund rather than an IRR quantile slice.
FULL_POOLED_BAND = 'IRR, Percentile, Minimum: 0, Maximum: 100'

# Residual valuation below this share of paid-in marks a cohort as realised.
DEFAULT_MAX_RESIDUAL_RATIO = 0.10

CohortKey = Tuple[str, int]


@dataclass(frozen=True)
class CohortCashFlows:
    """pooled cash-flow path of one strategy x vintage cohort.

    Attributes:
        group: strategy group label as the extract names it.
        vintage: vintage year.
        num_funds: funds pooled in the cohort at its last period.
        dates: period-end dates, ascending.
        paid_in: per-period paid-in capital ``c_t >= 0``, aligned to ``dates``.
        dist: per-period distributions ``d_t >= 0``, aligned to ``dates``.
        nav: residual valuation level ``V_t >= 0``, aligned to ``dates``. NaN
            where the extract reports no valuation.
    """

    group: str
    vintage: int
    num_funds: int
    dates: pd.DatetimeIndex
    paid_in: np.ndarray
    dist: np.ndarray
    nav: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.dates)
        for name, values in (('paid_in', self.paid_in), ('dist', self.dist), ('nav', self.nav)):
            if len(values) != n:
                raise ValueError(f"{name} has {len(values)} values against {n} dates "
                                 f"for {self.group!r} vintage {self.vintage}")
        if n and not self.dates.is_monotonic_increasing:
            raise ValueError(f"dates must ascend for {self.group!r} vintage {self.vintage}")

    @property
    def label(self) -> str:
        """the cohort as one identifier, used as the fund name in tidy frames."""
        return f"{self.group} | {self.vintage}"

    @property
    def cum_paid_in(self) -> float:
        """total paid-in capital over the path."""
        return float(np.nansum(self.paid_in))

    @property
    def cum_dist(self) -> float:
        """total distributions over the path."""
        return float(np.nansum(self.dist))

    @property
    def final_nav(self) -> float:
        """last reported valuation, or NaN if none was reported."""
        finite = np.flatnonzero(np.isfinite(self.nav))
        return float(self.nav[finite[-1]]) if len(finite) else float('nan')

    @property
    def final_nav_date(self) -> pd.Timestamp:
        """date of the last reported valuation, or the last period end if none was."""
        finite = np.flatnonzero(np.isfinite(self.nav))
        return pd.Timestamp(self.dates[finite[-1]] if len(finite) else self.dates[-1])

    @property
    def dpi(self) -> float:
        """distributions over paid-in."""
        return self.cum_dist / self.cum_paid_in if self.cum_paid_in > 0 else float('nan')

    @property
    def tvpi(self) -> float:
        """distributions plus residual valuation over paid-in."""
        if self.cum_paid_in <= 0:
            return float('nan')
        residual = self.final_nav if np.isfinite(self.final_nav) else 0.0
        return (self.cum_dist + residual) / self.cum_paid_in

    @property
    def residual_ratio(self) -> float:
        """residual valuation as a share of paid-in. Near zero means realised."""
        if self.cum_paid_in <= 0 or not np.isfinite(self.final_nav):
            return float('nan')
        return self.final_nav / self.cum_paid_in

    def signed_flows(self) -> pd.Series:
        """net LP cash flow per period, ``d_t - c_t``, without the residual valuation.

        Pass this with :attr:`final_nav` and :attr:`final_nav_date` to the
        single-benchmark measures, which append the residual themselves.
        """
        net = np.nan_to_num(self.dist, nan=0.0) - np.nan_to_num(self.paid_in, nan=0.0)
        return pd.Series(net, index=self.dates, name=self.label)


def load_cohort_extract(path: Union[str, Path],
                        group_col: str = 'Custom Group Name',  # strategy group label
                        band_col: str = 'Percentile Filter',  # IRR percentile band of the row
                        vintage_col: str = 'Vintage',  # vintage year
                        num_funds_col: str = 'NumberOfFunds',  # funds pooled at the period
                        date_col: str = 'PeriodEndDate',  # period-end date of the row
                        paid_in_col: str = 'PaidIn(%)',  # per-period paid-in, percent of committed
                        dist_col: str = 'Dist(%)',  # per-period distributions, percent of committed
                        nav_col: str = 'Valuation(%)',  # residual valuation level, percent of committed
                        scale: float = 0.01,  # multiplies the three amount columns; 1.0 for currency
                        na_values: Tuple[str, ...] = ('NA', 'N/A'),  # tokens the extract uses for missing
                        ) -> pd.DataFrame:
    """read a pooled-cohort cash-flow extract into the tidy cohort frame.

    Args:
        path: delimited file with one row per cohort, band and period.
        group_col: column carrying the strategy group label.
        band_col: column carrying the IRR percentile band of the row.
        vintage_col: column carrying the vintage year.
        num_funds_col: column carrying the number of funds pooled at the period.
        date_col: column carrying the period-end date.
        paid_in_col: column carrying per-period paid-in capital.
        dist_col: column carrying per-period distributions.
        nav_col: column carrying the residual valuation level.
        scale: factor applied to the three amount columns. The default turns
            percent of committed capital into a fraction. Pass 1.0 to read a
            currency column as it is.
        na_values: tokens the extract uses for a missing value.

    Returns:
        DataFrame with columns ``COHORT_COLUMNS``, sorted by group, band,
        vintage and date. A missing flow reads as zero, a missing valuation
        stays NaN.

    Raises:
        ValueError: if a required column is absent, or ``scale`` is not positive.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale!r}")
    source = {'group': group_col, 'band': band_col, 'vintage': vintage_col,
              'num_funds': num_funds_col, 'date': date_col,
              'paid_in': paid_in_col, 'dist': dist_col, 'nav': nav_col}
    df = pd.read_csv(path, na_values=list(na_values), keep_default_na=True)
    missing = [c for c in source.values() if c not in df.columns]
    if missing:
        raise ValueError(f"extract {path} is missing columns {missing}; got {list(df.columns)}")

    out = pd.DataFrame({name: df[col] for name, col in source.items()})
    out['date'] = pd.to_datetime(out['date'], errors='coerce')
    out = out.dropna(subset=['group', 'band', 'vintage', 'date']).copy()
    out['group'] = out['group'].astype(str)
    out['band'] = out['band'].astype(str)
    out['vintage'] = out['vintage'].astype(int)
    out['num_funds'] = pd.to_numeric(out['num_funds'], errors='coerce')
    for name in ('paid_in', 'dist', 'nav'):
        out[name] = pd.to_numeric(out[name], errors='coerce') * scale
    out['paid_in'] = out['paid_in'].fillna(0.0)
    out['dist'] = out['dist'].fillna(0.0)
    return out[COHORT_COLUMNS].sort_values(['group', 'band', 'vintage', 'date']).reset_index(drop=True)


def build_cohorts(extract: pd.DataFrame,
                  groups: Optional[List[str]] = None,  # None keeps every strategy group
                  band: str = FULL_POOLED_BAND,  # the band whose rows form the path
                  ) -> Dict[CohortKey, CohortCashFlows]:
    """assemble one :class:`CohortCashFlows` per strategy x vintage from one band.

    Args:
        extract: tidy cohort frame from :func:`load_cohort_extract`.
        groups: strategy group labels to keep. None keeps all of them.
        band: percentile band whose rows form each cohort's path. The default is
            the full pool. A quantile band is a slice of funds sorted on their
            realised outcome, which is a different object.

    Returns:
        Dict keyed by ``(group, vintage)``, each path sorted by date.

    Raises:
        ValueError: if a required column is absent, the band is not in the
            extract, or ``groups`` selects no rows.
    """
    missing = [c for c in COHORT_COLUMNS if c not in extract.columns]
    if missing:
        raise ValueError(f"extract is missing columns {missing}; got {list(extract.columns)}")
    present = set(extract['band'].unique())
    if band not in present:
        raise ValueError(f"band {band!r} is not in the extract; got {sorted(present)}")
    use = extract[extract['band'] == band]
    if groups is not None:
        use = use[use['group'].isin(groups)]
        if use.empty:
            raise ValueError(f"no rows for groups {groups!r} in band {band!r}")

    cohorts: Dict[CohortKey, CohortCashFlows] = {}
    for (group, vintage), g in use.groupby(['group', 'vintage'], sort=True):
        g = g.sort_values('date')
        num_funds = g['num_funds'].dropna()
        cohorts[(str(group), int(vintage))] = CohortCashFlows(
            group=str(group),
            vintage=int(vintage),
            num_funds=int(num_funds.iloc[-1]) if len(num_funds) else 0,
            dates=pd.DatetimeIndex(g['date'].values),
            paid_in=g['paid_in'].to_numpy(dtype=float),
            dist=g['dist'].to_numpy(dtype=float),
            nav=g['nav'].to_numpy(dtype=float),
        )
    return cohorts


def filter_mature(cohorts: Dict[CohortKey, CohortCashFlows],
                  max_residual_ratio: float = DEFAULT_MAX_RESIDUAL_RATIO,  # residual NAV over paid-in
                  ) -> Dict[CohortKey, CohortCashFlows]:
    """keep cohorts whose residual valuation is below a share of paid-in.

    A realised cohort's return is set by distributions that were paid, so it is
    free of appraisal smoothing. A young cohort is mostly residual valuation,
    which is appraised, and reintroduces the smoothing the cash-flow route is
    there to avoid.

    Args:
        cohorts: output of :func:`build_cohorts`.
        max_residual_ratio: a cohort is kept when its residual valuation over
            paid-in is finite and strictly below this.

    Returns:
        The kept cohorts, in the input order.

    Raises:
        ValueError: if ``max_residual_ratio`` is negative.
    """
    if max_residual_ratio < 0:
        raise ValueError(f"max_residual_ratio must be non-negative, got {max_residual_ratio!r}")
    return {key: c for key, c in cohorts.items()
            if np.isfinite(c.residual_ratio) and c.residual_ratio < max_residual_ratio}


def cohorts_to_cash_flows(cohorts: Dict[CohortKey, CohortCashFlows],
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """convert cohorts into the tidy ``(cf, navs)`` frames the estimator consumes.

    Each cohort takes the place of a fund: ``fund`` and ``vintage_label`` both
    carry :attr:`CohortCashFlows.label`. A period with paid-in capital yields one
    contribution row with a negative amount, a period with distributions one
    distribution row with a positive amount, both dated at the period end. A
    period with a reported valuation yields one NAV row.

    Args:
        cohorts: output of :func:`build_cohorts` or :func:`filter_mature`.

    Returns:
        Tuple of (cf, navs) with columns ``CF_COLUMNS`` and ``NAV_COLUMNS``,
        sorted by fund then date.

    Raises:
        ValueError: if ``cohorts`` is empty.
    """
    if not cohorts:
        raise ValueError("cohorts is empty")
    cf_rows: List[Dict] = []
    nav_rows: List[Dict] = []
    for cohort in cohorts.values():
        label = cohort.label
        for date, paid_in, dist, nav in zip(cohort.dates, cohort.paid_in, cohort.dist, cohort.nav,
                                            strict=True):
            if paid_in > 0:
                cf_rows.append({'fund': label, 'vintage_label': label, 'date': date,
                                'amount': -float(paid_in), 'kind': 'contribution'})
            if dist > 0:
                cf_rows.append({'fund': label, 'vintage_label': label, 'date': date,
                                'amount': float(dist), 'kind': 'distribution'})
            if np.isfinite(nav):
                nav_rows.append({'fund': label, 'vintage_label': label, 'date': date,
                                 'nav': float(nav)})
    cf = pd.DataFrame(cf_rows, columns=CF_COLUMNS)
    navs = pd.DataFrame(nav_rows, columns=NAV_COLUMNS)
    cf['date'] = pd.to_datetime(cf['date'])
    navs['date'] = pd.to_datetime(navs['date'])
    return (cf.sort_values(['fund', 'date']).reset_index(drop=True),
            navs.sort_values(['fund', 'date']).reset_index(drop=True))


def cohort_summary(cohorts: Dict[CohortKey, CohortCashFlows]) -> pd.DataFrame:
    """one row per cohort with its size, span and realised multiples.

    Args:
        cohorts: output of :func:`build_cohorts` or :func:`filter_mature`.

    Returns:
        DataFrame indexed by ``(group, vintage)`` with ``num_funds``,
        ``n_periods``, ``first_date``, ``last_date``, ``paid_in``, ``dist``,
        ``final_nav``, ``DPI``, ``TVPI`` and ``residual_ratio``. Empty when
        ``cohorts`` is empty.
    """
    rows = []
    for (group, vintage), c in cohorts.items():
        rows.append({'group': group, 'vintage': vintage, 'num_funds': c.num_funds,
                     'n_periods': len(c.dates),
                     'first_date': pd.Timestamp(c.dates[0]) if len(c.dates) else pd.NaT,
                     'last_date': pd.Timestamp(c.dates[-1]) if len(c.dates) else pd.NaT,
                     'paid_in': c.cum_paid_in, 'dist': c.cum_dist, 'final_nav': c.final_nav,
                     'DPI': c.dpi, 'TVPI': c.tvpi, 'residual_ratio': c.residual_ratio})
    columns = ['group', 'vintage', 'num_funds', 'n_periods', 'first_date', 'last_date',
               'paid_in', 'dist', 'final_nav', 'DPI', 'TVPI', 'residual_ratio']
    return pd.DataFrame(rows, columns=columns).set_index(['group', 'vintage'])
