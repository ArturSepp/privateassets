"""
privateassets.matf._returns — reported returns from cash flows and marks.

A fund does not report a return series. It reports a NAV at each period end and
a stream of dated capital calls and distributions. The return is reconstructed
by the modified Dietz approximation:

    r = (NAV_end + D - C - NAV_start) / (NAV_start + 0.5 C - 0.5 D)

The denominator is the capital actually at work over the period, treating flows
as arriving mid-period. That same denominator weights a vintage when several are
pooled into one manager series, because a fund with 5m at work should not move
the aggregate as much as one with 500m.

**One panel, one reporting frequency.** Every return in the output spans exactly
one period of the stated frequency. Nothing here interpolates, forward-fills or
otherwise invents a mark that was not reported:

- Forward-filling a mark produces a period of exactly zero return, which is
  indistinguishable from a genuinely flat period. It manufactures the smoothness
  the unsmoothing step then tries to remove, inflating the estimated AR
  coefficient and with it the volatility uplift 1/(1-theta).
- Computing a return across a gap and filing it at the panel frequency mixes
  one-period and two-period returns under one label, which corrupts every
  annualisation and every autocorrelation computed from the series.

A panel whose vintages report at different frequencies is not one panel. Use
:func:`infer_reporting_frequency` to see what each vintage reports at, and
:func:`split_by_reporting_frequency` to partition it into groups that can each
be estimated on their own frequency. Estimating the groups separately is
honest; interpolating them onto a common grid is an assumption about
unobserved path, and this package does not make it for you.
"""

# packages
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Reporting frequencies a private-asset panel plausibly uses, by months between
# marks. The value is the pandas period-end alias to build the grid on.
FREQUENCY_BY_MONTHS = {1: 'ME', 3: 'QE', 6: '2QE', 12: 'YE'}
PERIODS_PER_YEAR_BY_MONTHS = {1: 12, 3: 4, 6: 2, 12: 1}


def _period_ends(dates: pd.Series, freq: str) -> pd.Series:
    """snap dates to the end of the period they fall in."""
    return pd.to_datetime(dates).dt.to_period(_period_alias(freq)).dt.to_timestamp(
        how='end').dt.normalize()


def _period_alias(freq: str) -> str:
    """pandas period alias matching a period-end offset alias."""
    return {'ME': 'M', 'QE': 'Q', '2QE': '2Q', 'YE': 'Y'}.get(freq, freq.rstrip('E'))


def infer_reporting_frequency(navs: pd.DataFrame) -> pd.Series:
    """median months between consecutive marks, per vintage.

    Read this before choosing the frequency to estimate on. A panel where the
    values differ across vintages is not one panel.

    Args:
        navs: tidy NAV frame with ``vintage_label``, ``date`` and ``nav``.

    Returns:
        Median months between consecutive marks, indexed by vintage label. A
        vintage with a single mark is NaN.

    Raises:
        ValueError: if a required column is missing.
    """
    missing = [c for c in ('vintage_label', 'date') if c not in navs.columns]
    if missing:
        raise ValueError(f"navs is missing columns {missing}; got {list(navs.columns)}")

    out = {}
    for vintage, group in navs.groupby('vintage_label'):
        dates = pd.to_datetime(group['date']).sort_values().drop_duplicates()
        if len(dates) < 2:
            out[vintage] = float('nan')
            continue
        months = dates.dt.year * 12 + dates.dt.month
        out[vintage] = float(np.median(np.diff(months.values)))
    return pd.Series(out, name='months_between_marks').sort_index()


def split_by_reporting_frequency(cf: pd.DataFrame,
                                 navs: pd.DataFrame,
                                 ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """partition a mixed-frequency panel into groups that share one frequency.

    Each group is estimated on its own frequency and the results compared, which
    is the alternative to interpolating everything onto a common grid.

    Args:
        cf: tidy cash-flow frame.
        navs: tidy NAV frame.

    Returns:
        Dict keyed by months between marks, each value a ``(cf, navs)`` pair
        restricted to the vintages reporting at that frequency. Vintages with a
        single mark are excluded, since their frequency is unknown.
    """
    frequency = infer_reporting_frequency(navs).dropna()
    groups: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for months in sorted(frequency.unique()):
        vintages = set(frequency.index[frequency == months])
        groups[int(months)] = (cf[cf['vintage_label'].isin(vintages)].copy(),
                               navs[navs['vintage_label'].isin(vintages)].copy())
    return groups


def nav_implied_returns(cf: pd.DataFrame,
                        navs: pd.DataFrame,
                        freq: str = 'QE',  # period-end alias the panel reports on
                        require_regular: bool = True,  # reject a panel with gaps
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """reconstruct per-period returns per vintage by modified Dietz.

    Every returned value spans exactly one period of ``freq``. A vintage whose
    marks skip a period is a reporting-frequency mismatch, not a fund with a
    missing quarter, and is rejected by default rather than silently producing
    a shorter series.

    Args:
        cf: tidy cash-flow frame with ``vintage_label``, ``date`` and ``amount``.
            Contributions are negative.
        navs: tidy NAV frame with ``vintage_label``, ``date`` and ``nav``.
        freq: period-end alias the panel reports on, one of
            ``FREQUENCY_BY_MONTHS`` values.
        require_regular: raise when a vintage skips a period between its first
            and last mark. Setting this False drops the affected periods instead,
            which shortens the series without telling you.

    Returns:
        Tuple of (returns, capital), both on a complete ``freq`` index with one
        column per vintage. ``capital`` is the modified Dietz denominator, which
        is the weight to pool with. Periods outside a vintage's reporting life,
        and its first marked period, are NaN in ``returns`` and zero in
        ``capital``.

    Raises:
        ValueError: if a required column is missing, if ``freq`` is not a
            supported alias, or if ``require_regular`` is set and any vintage
            skips a period.
    """
    for frame, name, required in ((cf, 'cf', ('vintage_label', 'date', 'amount')),
                                  (navs, 'navs', ('vintage_label', 'date', 'nav'))):
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing columns {missing}; got {list(frame.columns)}")
    if freq not in FREQUENCY_BY_MONTHS.values():
        raise ValueError(f"freq must be one of {sorted(FREQUENCY_BY_MONTHS.values())}, "
                         f"got {freq!r}")

    cf = cf.copy()
    navs = navs.copy()
    cf['period'] = _period_ends(cf['date'], freq)
    navs['period'] = _period_ends(navs['date'], freq)

    vintages = sorted(set(cf['vintage_label']) | set(navs['vintage_label']))

    flows = (cf.assign(contrib=lambda d: np.where(d['amount'] < 0, -d['amount'], 0.0),
                       distrib=lambda d: np.where(d['amount'] > 0, d['amount'], 0.0))
             .groupby(['vintage_label', 'period'])[['contrib', 'distrib']].sum()
             .reset_index())
    contrib = flows.pivot(index='period', columns='vintage_label', values='contrib')
    distrib = flows.pivot(index='period', columns='vintage_label', values='distrib')

    marks = (navs.sort_values('date').groupby(['vintage_label', 'period'])['nav'].last()
             .reset_index()
             .pivot(index='period', columns='vintage_label', values='nav'))

    observed = sorted(set(contrib.index) | set(marks.index))
    grid = pd.date_range(min(observed), max(observed), freq=freq)
    contrib = contrib.reindex(index=grid, columns=vintages).fillna(0.0)
    distrib = distrib.reindex(index=grid, columns=vintages).fillna(0.0)
    marks = marks.reindex(index=grid, columns=vintages)

    gaps = _reporting_gaps(marks)
    if require_regular and gaps:
        raise ValueError(
            f"these vintages skip a {freq} period between their first and last mark: "
            f"{gaps}. The panel is not reported at one frequency. Inspect it with "
            f"infer_reporting_frequency and partition it with "
            f"split_by_reporting_frequency, or pass require_regular=False to drop "
            f"the affected periods.")

    nav_start = marks.shift(1)
    denominator = nav_start.fillna(0.0) + 0.5 * contrib - 0.5 * distrib
    numerator = marks - nav_start.fillna(0.0) + distrib - contrib

    # Both ends of the period must be marked, so every return spans one period.
    usable = (denominator > 0) & marks.notna() & nav_start.notna()
    returns = (numerator / denominator).where(usable)
    capital = denominator.where(usable, 0.0).clip(lower=0.0)
    return returns, capital


def _reporting_gaps(marks: pd.DataFrame) -> Dict[str, int]:
    """periods each vintage skips between its first and last mark."""
    gaps: Dict[str, int] = {}
    for vintage in marks.columns:
        reported = marks[vintage].notna()
        if reported.sum() < 2:
            continue
        positions = np.flatnonzero(reported.values)
        span = positions[-1] - positions[0] + 1
        skipped = int(span - len(positions))
        if skipped > 0:
            gaps[str(vintage)] = skipped
    return gaps


def pool_vintage_returns(returns: pd.DataFrame,
                         capital: pd.DataFrame,
                         ) -> pd.Series:
    """pool vintage returns into one manager series, weighted by capital at work.

    Each period is a weighted mean across the vintages reporting it, with the
    modified Dietz denominator as the weight. A period in which no vintage
    reports is dropped rather than returned as zero.

    Args:
        returns: per-period returns, one column per vintage.
        capital: capital at work, aligned to ``returns``.

    Returns:
        The pooled return series, named ``pooled``.

    Raises:
        ValueError: if the two frames do not share an index and columns.
    """
    if not returns.index.equals(capital.index) or not returns.columns.equals(capital.columns):
        raise ValueError("returns and capital must share an index and columns")

    valid = returns.notna() & capital.gt(0)
    weighted = returns.where(valid, 0.0) * capital.where(valid, 0.0)
    weights = capital.where(valid, 0.0).sum(axis=1)
    pooled = weighted.sum(axis=1) / weights.where(weights > 0)
    return pooled.dropna().rename('pooled')
