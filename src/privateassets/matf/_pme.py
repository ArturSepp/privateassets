"""
privateassets.matf._pme — classical PME measures and cash-flow containers.

Single-benchmark Public Market Equivalent measures: KS-PME (Kaplan-Schoar 2005),
Direct Alpha (Gredil-Griffiths-Stucke 2014), and Long-Nickels. Plus XIRR,
per-vintage statistics, capital-weighted aggregation, and the cash-flow and NAV
readers the estimator consumes.

All measures take a benchmark index level series and a tidy cash-flow frame. No
input data ships with the package. See DATA_README.md.
"""

# packages
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import brentq
from typing import Callable, Dict, List, Optional, Tuple, Union

# Column names of the tidy cash-flow and NAV frames every function here consumes.
CF_COLUMNS = ['fund', 'vintage_label', 'date', 'amount', 'kind']
NAV_COLUMNS = ['fund', 'vintage_label', 'date', 'nav']


def _canonical_fund_name(s: str) -> str:
    """collapse whitespace and normalise the legal suffix of a fund name."""
    s = ' '.join(str(s).strip().split())
    return s.replace(', LP', ', L.P.')


def _short_label(fund_name: str) -> str:
    """drop the legal suffix to give a compact vintage label."""
    return fund_name.replace(', L.P.', '').strip()


def load_cash_flows(path: Union[str, Path],
                    sheet: str = 'Fund Net Cash Flows',  # workbook sheet holding dated LP cash flows
                    fund_col: str = 'Fund Name',  # column carrying the fund identifier
                    date_col: str = 'Date',  # column carrying the cash-flow date
                    amount_col: str = 'Cash Flow Amount',  # signed amount, contributions negative
                    canonicalizer: Optional[Callable[[str], str]] = None,  # default: _canonical_fund_name
                    labeler: Optional[Callable[[str], str]] = None,  # default: _short_label
                    ) -> pd.DataFrame:
    """read a cash-flow workbook into the tidy long frame the measures consume.

    Args:
        path: workbook to read.
        sheet: sheet name holding the cash flows.
        fund_col: column carrying the fund identifier.
        date_col: column carrying the cash-flow date.
        amount_col: column carrying the signed amount. Contributions are negative
            (LP outflow) and distributions positive (LP inflow).
        canonicalizer: maps a raw fund name to its canonical form.
        labeler: maps a canonical fund name to a compact vintage label.

    Returns:
        DataFrame with columns ``CF_COLUMNS``, sorted by fund then date.

    Raises:
        ValueError: if a required column is absent from the sheet.
    """
    canon = canonicalizer if canonicalizer is not None else _canonical_fund_name
    label = labeler if labeler is not None else _short_label
    df = pd.read_excel(path, sheet_name=sheet)
    missing = [c for c in (fund_col, date_col, amount_col) if c not in df.columns]
    if missing:
        raise ValueError(f"sheet {sheet!r} of {path} is missing columns {missing}; "
                         f"got {list(df.columns)}")
    df = df.dropna(subset=[fund_col])
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, amount_col]).copy()
    df['fund'] = df[fund_col].apply(canon)
    df['vintage_label'] = df['fund'].apply(label)
    df['date'] = df[date_col]
    df['amount'] = df[amount_col].astype(float)
    df['kind'] = np.where(df['amount'] < 0, 'contribution', 'distribution')
    return df[CF_COLUMNS].sort_values(['fund', 'date']).reset_index(drop=True)


def load_navs(path: Union[str, Path],
              sheet: str = 'Fund Net Asset Values',  # workbook sheet holding reported NAVs
              fund_col: str = 'Fund Name',  # column carrying the fund identifier
              date_col: str = 'Date',  # column carrying the NAV observation date
              nav_col: str = 'Net Asset Value Amount',  # column carrying the reported NAV
              canonicalizer: Optional[Callable[[str], str]] = None,  # default: _canonical_fund_name
              labeler: Optional[Callable[[str], str]] = None,  # default: _short_label
              ) -> pd.DataFrame:
    """read a NAV workbook into the tidy long frame the measures consume.

    Args:
        path: workbook to read.
        sheet: sheet name holding the NAVs.
        fund_col: column carrying the fund identifier.
        date_col: column carrying the NAV observation date.
        nav_col: column carrying the reported NAV.
        canonicalizer: maps a raw fund name to its canonical form.
        labeler: maps a canonical fund name to a compact vintage label.

    Returns:
        DataFrame with columns ``NAV_COLUMNS``, sorted by fund then date.

    Raises:
        ValueError: if a required column is absent from the sheet.
    """
    canon = canonicalizer if canonicalizer is not None else _canonical_fund_name
    label = labeler if labeler is not None else _short_label
    df = pd.read_excel(path, sheet_name=sheet)
    missing = [c for c in (fund_col, date_col, nav_col) if c not in df.columns]
    if missing:
        raise ValueError(f"sheet {sheet!r} of {path} is missing columns {missing}; "
                         f"got {list(df.columns)}")
    df = df.dropna(subset=[fund_col])
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, nav_col]).copy()
    df['fund'] = df[fund_col].apply(canon)
    df['vintage_label'] = df['fund'].apply(label)
    df['date'] = df[date_col]
    df['nav'] = df[nav_col].astype(float)
    return df[NAV_COLUMNS].sort_values(['fund', 'date']).reset_index(drop=True)


def xirr(dates: pd.Series,
         amounts: pd.Series,
         lo: float = -0.5,  # lower bracket for the annualised rate
         hi: float = 5.0,  # upper bracket for the annualised rate
         ) -> float:
    """internal rate of return of dated cash flows, by bracketed root finding.

    Solves ``Σ_t a_t (1 + r)^{-h_t} = 0`` for r, where h_t is the ACT/365.25 year
    fraction from the first date. Brackets on ``[lo, hi]``; if the net present
    value does not change sign there, scans for any sign change before bracketing.

    Args:
        dates: cash-flow dates, ascending.
        amounts: signed cash-flow amounts aligned to ``dates``.
        lo: lower bracket for the annualised rate.
        hi: upper bracket for the annualised rate.

    Returns:
        The annualised rate, or NaN if no root can be bracketed.

    Raises:
        ValueError: if ``dates`` and ``amounts`` differ in length.
    """
    if len(dates) != len(amounts):
        raise ValueError(f"dates and amounts must align, got {len(dates)} and {len(amounts)}")
    if len(dates) == 0:
        return float('nan')
    t0 = pd.Timestamp(pd.Series(dates).iloc[0])
    yfrac = (pd.to_datetime(pd.Series(dates).reset_index(drop=True)) - t0).dt.days.values / 365.25
    amt = np.asarray(amounts, dtype=float)

    def npv(r: float) -> float:
        return float(np.sum(amt / (1.0 + r) ** yfrac))

    try:
        if npv(lo) * npv(hi) < 0:
            return brentq(npv, lo, hi, xtol=1e-8, maxiter=200)
        scan_rs = np.concatenate([
            np.linspace(-0.5, 0, 21)[1:-1],
            np.linspace(0, 1.0, 41),
            np.linspace(1.0, 5.0, 17)[1:],
        ])
        scan_npvs = np.array([npv(r) for r in scan_rs])
        sign_changes = np.where(np.diff(np.sign(scan_npvs)) != 0)[0]
        if len(sign_changes) == 0:
            return float('nan')
        i = sign_changes[0]
        return brentq(npv, float(scan_rs[i]), float(scan_rs[i + 1]), xtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return float('nan')


def ks_pme(cf_dates: pd.Series,
           cf_amounts: pd.Series,
           rvpi_nav: float,  # residual NAV at rvpi_date, 0.0 if fully realised
           rvpi_date: pd.Timestamp,
           bench_idx: pd.Series,  # benchmark total-return index level, indexed by date
           ) -> float:
    """Kaplan-Schoar PME: benchmark-discounted wealth multiple.

    ``PME = (Σ FV(distributions) + residual NAV) / Σ FV(contributions)`` where
    ``FV(a_t) = a_t I(T) / I(t)``. A value above 1.0 means the fund beat the
    benchmark on a wealth-multiple basis.

    Args:
        cf_dates: cash-flow dates.
        cf_amounts: signed cash-flow amounts, contributions negative.
        rvpi_nav: residual NAV carried at ``rvpi_date``.
        rvpi_date: date of the residual NAV.
        bench_idx: benchmark index levels. Looked up with ``asof``, so a level is
            not required on every cash-flow date.

    Returns:
        The PME ratio, or NaN if the benchmark has no level at the terminal date
        or contributions are zero.
    """
    if len(cf_dates) != len(cf_amounts):
        raise ValueError(f"cf_dates and cf_amounts must align, "
                         f"got {len(cf_dates)} and {len(cf_amounts)}")
    bench_idx = bench_idx.sort_index()
    terminal = max(pd.Timestamp(cf_dates.max()), pd.Timestamp(rvpi_date))
    i_terminal = bench_idx.asof(terminal)
    if pd.isna(i_terminal):
        return float('nan')

    fv_contrib = 0.0
    fv_distrib = 0.0
    for d, a in zip(cf_dates, cf_amounts, strict=True):
        i_t = bench_idx.asof(d)
        if pd.isna(i_t) or i_t == 0:
            continue
        factor = i_terminal / i_t
        if a < 0:
            fv_contrib += -a * factor
        else:
            fv_distrib += a * factor

    if rvpi_nav > 0:
        fv_distrib += rvpi_nav  # already at the terminal date, factor is 1

    return fv_distrib / fv_contrib if fv_contrib > 0 else float('nan')


def direct_alpha(cf_dates: pd.Series,
                 cf_amounts: pd.Series,
                 rvpi_nav: float,  # residual NAV at rvpi_date
                 rvpi_date: pd.Timestamp,
                 bench_idx: pd.Series,  # benchmark total-return index level
                 ) -> float:
    """Direct Alpha (Gredil-Griffiths-Stucke 2014): annualised excess return.

    The IRR of cash flows scaled by ``I(T)/I(t)``, with the residual NAV added at
    T. Interprets as the geometric annualised excess over the benchmark,
    ``(1 + r_fund)/(1 + r_bench) - 1``.

    Args:
        cf_dates: cash-flow dates.
        cf_amounts: signed cash-flow amounts, contributions negative.
        rvpi_nav: residual NAV carried at ``rvpi_date``.
        rvpi_date: date of the residual NAV.
        bench_idx: benchmark index levels, looked up with ``asof``.

    Returns:
        The annualised excess return, or NaN if it cannot be bracketed.
    """
    if len(cf_dates) != len(cf_amounts):
        raise ValueError(f"cf_dates and cf_amounts must align, "
                         f"got {len(cf_dates)} and {len(cf_amounts)}")
    bench_idx = bench_idx.sort_index()
    terminal = max(pd.Timestamp(cf_dates.max()), pd.Timestamp(rvpi_date))
    i_terminal = bench_idx.asof(terminal)
    if pd.isna(i_terminal):
        return float('nan')

    deflated_dates: List[pd.Timestamp] = []
    deflated_amts: List[float] = []
    for d, a in zip(cf_dates, cf_amounts, strict=True):
        i_t = bench_idx.asof(d)
        if pd.isna(i_t) or i_t == 0:
            continue
        deflated_dates.append(d)
        deflated_amts.append(a * (i_terminal / i_t))

    if rvpi_nav > 0:
        deflated_dates.append(terminal)
        deflated_amts.append(rvpi_nav)

    return xirr(pd.Series(deflated_dates), pd.Series(deflated_amts))


def long_nickels_pme(cf_dates: pd.Series,
                     cf_amounts: pd.Series,
                     rvpi_nav: float,  # residual NAV at rvpi_date
                     rvpi_date: pd.Timestamp,
                     bench_idx: pd.Series,  # benchmark total-return index level
                     ) -> Dict[str, Union[float, bool]]:
    """Long-Nickels PME: residual NAV against a shadow benchmark position.

    Each contribution buys benchmark units at ``I(t)`` and each distribution
    sells them. The measure compares the fund's residual NAV to the value of the
    shadow position at T.

    The shadow position goes negative when distributions outrun the benchmark,
    which is the known degenerate case of this measure. It is reported rather
    than hidden, through ``shadow_negative``.

    Args:
        cf_dates: cash-flow dates.
        cf_amounts: signed cash-flow amounts, contributions negative.
        rvpi_nav: residual NAV carried at ``rvpi_date``.
        rvpi_date: date of the residual NAV.
        bench_idx: benchmark index levels, looked up with ``asof``.

    Returns:
        Dict with ``ln_pme``, ``shadow_nav`` and ``shadow_negative``. ``ln_pme``
        is NaN when the shadow position is not strictly positive.
    """
    if len(cf_dates) != len(cf_amounts):
        raise ValueError(f"cf_dates and cf_amounts must align, "
                         f"got {len(cf_dates)} and {len(cf_amounts)}")
    bench_idx = bench_idx.sort_index()
    units = 0.0
    shadow_negative = False
    for d, a in zip(cf_dates, cf_amounts, strict=True):
        i_t = bench_idx.asof(d)
        if pd.isna(i_t) or i_t == 0:
            continue
        units += -a / i_t  # contribution adds units, distribution removes them
        if units < 0:
            shadow_negative = True
            break

    terminal = max(pd.Timestamp(cf_dates.max()), pd.Timestamp(rvpi_date))
    i_terminal = bench_idx.asof(terminal)
    shadow_nav = units * i_terminal if not pd.isna(i_terminal) else float('nan')
    ln_pme = rvpi_nav / shadow_nav if (shadow_nav and shadow_nav > 0) else float('nan')
    return {
        'ln_pme': ln_pme,
        'shadow_nav': float(shadow_nav) if not pd.isna(shadow_nav) else float('nan'),
        'shadow_negative': shadow_negative,
    }


def compute_vintage_stats(cf: pd.DataFrame,
                          navs: pd.DataFrame,
                          labeler: Optional[Callable[[str], str]] = None,  # default: _short_label
                          ) -> pd.DataFrame:
    """per-vintage cash-flow statistics, computed without a benchmark.

    Args:
        cf: tidy cash-flow frame with columns ``CF_COLUMNS``.
        navs: tidy NAV frame with columns ``NAV_COLUMNS``.
        labeler: maps a fund name to its vintage label.

    Returns:
        One row per fund carrying DPI, RVPI, TVPI, the net IRR computed with the
        last reported NAV as a synthetic terminal distribution, and the vintage
        year taken from the first capital call.

    Raises:
        ValueError: if ``cf`` is missing a required column.
    """
    missing = [c for c in CF_COLUMNS if c not in cf.columns]
    if missing:
        raise ValueError(f"cf is missing columns {missing}; got {list(cf.columns)}")
    label = labeler if labeler is not None else _short_label
    rows = []
    for fund, g in cf.groupby('fund', sort=False):
        g = g.sort_values('date')
        contrib = -g.loc[g['amount'] < 0, 'amount'].sum()
        distrib = g.loc[g['amount'] > 0, 'amount'].sum()

        nav_g = navs[navs['fund'] == fund].sort_values('date')
        if len(nav_g):
            last = nav_g.iloc[-1]
            rvpi_nav = float(last['nav'])
            rvpi_date = pd.Timestamp(last['date'])
        else:
            rvpi_nav = 0.0
            rvpi_date = g['date'].max()

        irr_dates = pd.concat([g['date'], pd.Series([max(g['date'].max(), rvpi_date)])],
                              ignore_index=True)
        irr_amts = pd.concat([g['amount'], pd.Series([rvpi_nav])], ignore_index=True)
        net_irr = xirr(irr_dates, irr_amts)

        rows.append({
            'fund': fund,
            'vintage_label': label(fund),
            'first_call': g.loc[g['amount'] < 0, 'date'].min(),
            'last_cf': g['date'].max(),
            'last_nav_date': rvpi_date,
            'contributions': contrib,
            'distributions': distrib,
            'rvpi_nav': rvpi_nav,
            'DPI': distrib / contrib if contrib > 0 else np.nan,
            'RVPI': rvpi_nav / contrib if contrib > 0 else np.nan,
            'TVPI': (distrib + rvpi_nav) / contrib if contrib > 0 else np.nan,
            'calc_net_IRR': net_irr,
            'life_years': (g['date'].max() - g['date'].min()).days / 365.25,
        })

    df = pd.DataFrame(rows).sort_values('first_call').reset_index(drop=True)
    df['vintage_year'] = df['first_call'].dt.year
    return df


def cap_weighted_aggregates(stats: pd.DataFrame,
                            bench_labels: List[str],
                            dpi_threshold: float = 0.8,  # a vintage enters the aggregate above this DPI
                            ) -> pd.DataFrame:
    """capital-weighted KS-PME and Direct Alpha over the mature vintages.

    Maturity is a DPI gate, because an unrealised vintage carries a PME that is
    an appraisal rather than a realisation. Weights are contributions.

    Args:
        stats: frame carrying ``contributions``, ``DPI``, and one
            ``KS_PME_<label>`` and ``Direct_Alpha_<label>`` column per benchmark.
        bench_labels: benchmark labels to aggregate.
        dpi_threshold: a vintage enters the aggregate when its DPI exceeds this.

    Returns:
        One row per benchmark with the vintage count, total contributions, and
        both capital-weighted measures.

    Raises:
        ValueError: if a benchmark's columns are absent from ``stats``.
    """
    rows = []
    mature = stats[stats['DPI'] > dpi_threshold].dropna(subset=['contributions']).copy()
    for lbl in bench_labels:
        ks_col, da_col = f'KS_PME_{lbl}', f'Direct_Alpha_{lbl}'
        missing = [c for c in (ks_col, da_col) if c not in stats.columns]
        if missing:
            raise ValueError(f"stats is missing columns {missing} for benchmark {lbl!r}")
        valid = mature.dropna(subset=[ks_col, da_col])
        w = valid['contributions']
        total_w = w.sum()
        rows.append({
            'benchmark': lbl,
            'n_mature_vintages': len(valid),
            'total_contributions': float(total_w),
            'cap_wtd_KS_PME': (valid[ks_col] * w).sum() / total_w if total_w > 0 else float('nan'),
            'cap_wtd_Direct_Alpha': (valid[da_col] * w).sum() / total_w if total_w > 0 else float('nan'),
        })
    return pd.DataFrame(rows)


def cf_with_terminal_for_vintage(cf_g: pd.DataFrame,
                                 nav_g: pd.DataFrame,
                                 ) -> Tuple[pd.DataFrame, float, pd.Timestamp, List[pd.Timestamp]]:
    """append the terminal NAV date to one vintage's cash-flow dates.

    Every deflator-based measure evaluates on the cash-flow dates plus a terminal
    date carrying the residual NAV. This builds that date list once so the
    deflator and the alpha solver see the same grid.

    Args:
        cf_g: one vintage's rows from a tidy cash-flow frame.
        nav_g: the same vintage's rows from a tidy NAV frame.

    Returns:
        Tuple of the date-sorted cash-flow frame, the residual NAV, the residual
        NAV date, and the cash-flow dates with the terminal date appended.
    """
    cf_g = cf_g.sort_values('date').copy()
    if len(nav_g):
        last = nav_g.sort_values('date').iloc[-1]
        rvpi_nav = float(last['nav'])
        rvpi_date = pd.Timestamp(last['date'])
    else:
        rvpi_nav = 0.0
        rvpi_date = cf_g['date'].max()
    terminal = max(rvpi_date, cf_g['date'].max())
    cf_dates = [pd.Timestamp(d) for d in pd.to_datetime(cf_g['date']).values] + [terminal]
    return cf_g, rvpi_nav, rvpi_date, cf_dates


def vintage_direct_alpha(cf_v: pd.DataFrame,
                         rvpi_nav: float,  # residual NAV, appended as the terminal flow
                         cf_dates: List[pd.Timestamp],
                         deflators: np.ndarray,  # deflator value at each date in cf_dates
                         ) -> float:
    """Direct Alpha of one vintage against an arbitrary deflator path.

    Generalises :func:`direct_alpha` from a single benchmark index to any
    deflator, which is what makes the multi-factor measure possible: pass the
    MATF deflator and the same root-finding step returns the multi-factor alpha.

    Solves ``Σ_t (a_t / R_t) exp(-a h_t) = 0`` for the continuously-compounded
    alpha a, then returns ``exp(a) - 1``.

    Args:
        cf_v: one vintage's cash flows, carrying an ``amount`` column.
        rvpi_nav: residual NAV, appended as the terminal flow.
        cf_dates: cash-flow dates with the terminal date appended.
        deflators: deflator value at each date in ``cf_dates``.

    Returns:
        The annualised alpha, or NaN if any deflator is missing or not positive,
        or if the root cannot be bracketed on ``[-1, 1]`` in log terms.

    Raises:
        ValueError: if the amounts and deflators do not align.
    """
    amts = np.concatenate([cf_v['amount'].values, [rvpi_nav]])
    if len(amts) != len(deflators):
        raise ValueError(f"amounts and deflators must align, "
                         f"got {len(amts)} and {len(deflators)}")
    if len(amts) != len(cf_dates):
        raise ValueError(f"amounts and cf_dates must align, "
                         f"got {len(amts)} and {len(cf_dates)}")
    deflators = np.asarray(deflators, dtype=float)
    if not ((deflators == deflators) & (deflators > 0)).all():
        return float('nan')
    deflated = amts / deflators

    t0 = cf_dates[0]
    hs = np.array([(pd.Timestamp(t) - pd.Timestamp(t0)).days / 365.25 for t in cf_dates])

    def npv_da(da_log: float) -> float:
        return float(np.sum(deflated * np.exp(-da_log * hs)))

    try:
        if npv_da(-1.0) * npv_da(1.0) < 0:
            da_log = brentq(npv_da, -1.0, 1.0, xtol=1e-8, maxiter=200)
            return float(np.exp(da_log) - 1.0)
    except (ValueError, RuntimeError):
        pass
    return float('nan')
