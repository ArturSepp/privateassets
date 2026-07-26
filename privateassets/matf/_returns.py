"""
privateassets.matf._returns — NAV-implied returns from cash flows and marks.

A fund does not report a return series. It reports a NAV at each quarter end and
a stream of dated capital calls and distributions. The return has to be
reconstructed, and the modified Dietz approximation is the standard way:

    r_q = (NAV_end + D_q - C_q - NAV_start) / (NAV_start + 0.5 C_q - 0.5 D_q)

The denominator is the capital actually at work over the quarter, treating flows
as arriving mid-quarter. That same denominator is what weights a vintage when
several are pooled into one manager-level series, because a fund with 5m at work
should not move the aggregate as much as one with 500m.

**A quarter with no NAV report has an unknown return, not a zero one.** Carrying
the previous mark forward produces a quarter of exactly zero return, which is
indistinguishable from a genuinely flat quarter and manufactures the very
smoothness the unsmoothing step then tries to remove. Inflating the AR
coefficient inflates the volatility uplift 1/(1-theta). Unreported quarters are
left missing here and excluded from the pooled series.

**Every reported return spans exactly one quarter.** A return is computed only
when the quarter and the one before it both carry a mark, so a gap produces two
missing quarters rather than one two-quarter return wearing a quarterly label. A
series that is genuinely reported less often than quarterly is not made
quarterly by this function: interpolate it with
``qis.interpolate_infrequent_returns``, which draws the intermediate path from a
Brownian bridge instead of asserting one.
"""

# packages
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple


def nav_implied_quarterly_returns(cf: pd.DataFrame,
                                  navs: pd.DataFrame,
                                  carry_navs_forward: bool = False,  # see the module note before enabling
                                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """reconstruct quarterly returns per vintage by modified Dietz.

    Args:
        cf: tidy cash-flow frame with ``vintage_label``, ``date`` and ``amount``.
            Contributions are negative.
        navs: tidy NAV frame with ``vintage_label``, ``date`` and ``nav``.
        carry_navs_forward: forward-fill a missing quarter's NAV from the last
            reported one. Off by default: it turns an unreported quarter into a
            zero-return quarter and biases the estimated smoothing coefficient
            upward.

    Returns:
        Tuple of (returns, capital), both on a complete quarterly index with one
        column per vintage. ``capital`` is the modified Dietz denominator, which
        is the weight to pool with. A quarter that is unreported, that follows an
        unreported quarter, or that has no positive denominator is NaN in
        ``returns`` and zero in ``capital``.

    Raises:
        ValueError: if a required column is missing from either frame.
    """
    for frame, name, required in ((cf, 'cf', ('vintage_label', 'date', 'amount')),
                                  (navs, 'navs', ('vintage_label', 'date', 'nav'))):
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing columns {missing}; got {list(frame.columns)}")

    cf = cf.copy()
    navs = navs.copy()
    for frame in (cf, navs):
        frame['quarter'] = (pd.to_datetime(frame['date']).dt.to_period('Q')
                            .dt.to_timestamp(how='end').dt.normalize())

    vintages = sorted(set(cf['vintage_label']) | set(navs['vintage_label']))

    flows = (cf.assign(contrib=lambda d: np.where(d['amount'] < 0, -d['amount'], 0.0),
                       distrib=lambda d: np.where(d['amount'] > 0, d['amount'], 0.0))
             .groupby(['vintage_label', 'quarter'])[['contrib', 'distrib']].sum()
             .reset_index())
    contrib = flows.pivot(index='quarter', columns='vintage_label', values='contrib')
    distrib = flows.pivot(index='quarter', columns='vintage_label', values='distrib')

    marks = (navs.sort_values('date').groupby(['vintage_label', 'quarter'])['nav'].last()
             .reset_index()
             .pivot(index='quarter', columns='vintage_label', values='nav'))

    # A complete quarterly range, so an unreported quarter is visibly missing
    # rather than absent from the index altogether.
    observed = sorted(set(contrib.index) | set(marks.index))
    grid = pd.date_range(min(observed), max(observed), freq='QE')
    contrib = contrib.reindex(index=grid, columns=vintages).fillna(0.0)
    distrib = distrib.reindex(index=grid, columns=vintages).fillna(0.0)
    marks = marks.reindex(index=grid, columns=vintages)
    if carry_navs_forward:
        marks = marks.ffill()

    nav_start = marks.shift(1)
    first_mark = marks.notna().cumsum().eq(1) & marks.notna()
    denominator = nav_start.fillna(0.0) + 0.5 * contrib - 0.5 * distrib
    numerator = marks - nav_start.fillna(0.0) + distrib - contrib

    # Both ends of the quarter must be marked, so every return spans one quarter.
    both_ends = marks.notna() & (nav_start.notna() | first_mark)
    usable = (denominator > 0) & both_ends
    returns = (numerator / denominator).where(usable)
    capital = denominator.where(usable, 0.0).clip(lower=0.0)
    return returns, capital


def pool_vintage_returns(returns: pd.DataFrame,
                         capital: pd.DataFrame,
                         ) -> pd.Series:
    """pool vintage returns into one manager series, weighted by capital at work.

    Each quarter is a weighted mean across the vintages reporting that quarter,
    with the modified Dietz denominator as the weight. A quarter in which no
    vintage reports is dropped rather than returned as zero.

    Args:
        returns: quarterly returns, one column per vintage.
        capital: capital at work, aligned to ``returns``.

    Returns:
        The pooled quarterly return series, named ``pooled``.

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
