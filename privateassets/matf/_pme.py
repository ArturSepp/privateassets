"""
privateassets.matf._pme — classical PME measures and cash-flow I/O.

Single-benchmark Public Market Equivalent measures: KS-PME (Kaplan-Schoar 2005),
Direct Alpha (Gredil-Griffiths-Stucke), and Long-Nickels. Plus XIRR, per-vintage
statistics, cap-weighted aggregation, and the cash-flow and NAV loaders the
estimator pipeline consumes.

Input data is licensed and is never shipped with the package. See DATA_README.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def _canonical_fund_name(s: str) -> str:
    """Generic fund-name normalization: collapse whitespace and normalize the legal suffix."""
    s = ' '.join(str(s).strip().split())
    return s.replace(', LP', ', L.P.')


def _short_label(fund_name: str) -> str:
    """Generic compact vintage label: drop the legal suffix and surrounding whitespace."""
    return fund_name.replace(', L.P.', '').strip()


def load_cash_flows(path: Path,
                    sheet: str = 'Fund Net Cash Flows',  # keep current default so existing files load unchanged
                    canonicalizer: Optional[Callable[[str], str]] = None,  # default: generic _canonical_fund_name
                    labeler: Optional[Callable[[str], str]] = None,  # default: generic _short_label
                    ) -> pd.DataFrame:
    """Tidy long-format CF DF: [fund, vintage_label, date, amount, kind]."""
    canon = canonicalizer if canonicalizer is not None else _canonical_fund_name
    label = labeler if labeler is not None else _short_label
    df = pd.read_excel(path, sheet_name=sheet).dropna(subset=['Fund Name'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Cash Flow Amount']).copy()
    df['fund'] = df['Fund Name'].apply(canon)
    df['vintage_label'] = df['fund'].apply(label)
    df['date'] = df['Date']
    df['amount'] = df['Cash Flow Amount'].astype(float)
    # Sign: contributions negative (LP outflow), distributions positive (LP inflow)
    df['kind'] = np.where(df['amount'] < 0, 'contribution', 'distribution')
    return df[['fund', 'vintage_label', 'date', 'amount', 'kind']].sort_values(['fund', 'date'])


def load_navs(path: Path,
              sheet: str = 'Fund Net Asset Values',  # keep current default so existing files load unchanged
              canonicalizer: Optional[Callable[[str], str]] = None,  # default: generic _canonical_fund_name
              labeler: Optional[Callable[[str], str]] = None,  # default: generic _short_label
              ) -> pd.DataFrame:
    """Tidy NAV DF: [fund, vintage_label, date, nav]."""
    canon = canonicalizer if canonicalizer is not None else _canonical_fund_name
    label = labeler if labeler is not None else _short_label
    df = pd.read_excel(path, sheet_name=sheet).dropna(subset=['Fund Name'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Net Asset Value Amout']).copy()
    df['fund'] = df['Fund Name'].apply(canon)
    df['vintage_label'] = df['fund'].apply(label)
    df = df.rename(columns={'Date': 'date', 'Net Asset Value Amout': 'nav'})
    df['nav'] = df['nav'].astype(float)
    return df[['fund', 'vintage_label', 'date', 'nav']].sort_values(['fund', 'date'])


def xirr(dates: pd.Series, amounts: pd.Series, lo: float = -0.5, hi: float = 5.0) -> float:
    """
    XIRR via bracketed root finding (brentq).
    Default bracket [-50%, +500%] covers all realistic fund IRRs.
    If no sign change in default bracket, falls back to a coarse scan.
    Returns NaN if no IRR can be bracketed.
    """
    t0 = pd.Timestamp(dates.iloc[0])
    yfrac = (pd.to_datetime(pd.Series(dates).reset_index(drop=True)) - t0).dt.days.values / 365.25
    amt = np.asarray(amounts, dtype=float)

    def npv(r: float) -> float:
        return float(np.sum(amt / (1.0 + r) ** yfrac))

    try:
        if npv(lo) * npv(hi) < 0:
            return brentq(npv, lo, hi, xtol=1e-8, maxiter=200)
        # Coarse scan to find any sign change
        scan_rs = np.concatenate([
            np.linspace(-0.5, 0, 21)[1:-1],     # -50% to 0%
            np.linspace(0, 1.0, 41),            # 0% to 100%
            np.linspace(1.0, 5.0, 17)[1:],      # 100% to 500%
        ])
        scan_npvs = np.array([npv(r) for r in scan_rs])
        sign_changes = np.where(np.diff(np.sign(scan_npvs)) != 0)[0]
        if len(sign_changes) == 0:
            return float('nan')
        i = sign_changes[0]
        return brentq(npv, float(scan_rs[i]), float(scan_rs[i + 1]), xtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return float('nan')


def ks_pme(
    cf_dates: pd.Series, cf_amounts: pd.Series,
    rvpi_nav: float, rvpi_date: pd.Timestamp,
    bench_idx: pd.Series,
) -> float:
    """
    Kaplan-Schoar PME = (Σ FV(distributions) + residual NAV at T) / Σ FV(contributions)
    where FV(cf_t) = cf_t * I(T) / I(t).
    >1.0 means LP outperformed the public benchmark on a wealth-multiple basis.
    """
    bench_idx = bench_idx.sort_index()
    T = max(pd.Timestamp(cf_dates.max()), pd.Timestamp(rvpi_date))
    I_T = bench_idx.asof(T)
    if pd.isna(I_T):
        return float('nan')

    fv_contrib = 0.0
    fv_distrib = 0.0
    for d, a in zip(cf_dates, cf_amounts):
        I_t = bench_idx.asof(d)
        if pd.isna(I_t) or I_t == 0:
            continue
        factor = I_T / I_t
        if a < 0:
            fv_contrib += -a * factor
        else:
            fv_distrib += a * factor

    if rvpi_nav > 0:
        fv_distrib += rvpi_nav  # already at T, factor = 1

    return fv_distrib / fv_contrib if fv_contrib > 0 else float('nan')


def direct_alpha(
    cf_dates: pd.Series, cf_amounts: pd.Series,
    rvpi_nav: float, rvpi_date: pd.Timestamp,
    bench_idx: pd.Series,
) -> float:
    """
    Direct Alpha (Gredil-Griffiths-Stucke 2014).
    IRR of cash flows scaled by I(T)/I(t), with residual NAV added at T.
    Returns annualized geometric excess return vs the public benchmark:
        (1 + r_LP) / (1 + r_bench) - 1
    """
    bench_idx = bench_idx.sort_index()
    T = max(pd.Timestamp(cf_dates.max()), pd.Timestamp(rvpi_date))
    I_T = bench_idx.asof(T)
    if pd.isna(I_T):
        return float('nan')

    deflated_dates, deflated_amts = [], []
    for d, a in zip(cf_dates, cf_amounts):
        I_t = bench_idx.asof(d)
        if pd.isna(I_t) or I_t == 0:
            continue
        deflated_dates.append(d)
        deflated_amts.append(a * (I_T / I_t))

    if rvpi_nav > 0:
        deflated_dates.append(T)
        deflated_amts.append(rvpi_nav)

    return xirr(pd.Series(deflated_dates), pd.Series(deflated_amts))


def long_nickels_pme(
    cf_dates: pd.Series, cf_amounts: pd.Series,
    rvpi_nav: float, rvpi_date: pd.Timestamp,
    bench_idx: pd.Series,
) -> dict:
    """
    Long-Nickels PME — for each contribution "buy" benchmark units at I(t),
    for each distribution "sell" units. Compare residual NAV vs shadow PM NAV.

    Known degenerate case: when distributions exceed shadow value mid-life,
    units go negative — flag with `shadow_negative=True` for caller to handle.
    """
    bench_idx = bench_idx.sort_index()
    units = 0.0
    shadow_negative = False
    for d, a in zip(cf_dates, cf_amounts):
        I_t = bench_idx.asof(d)
        if pd.isna(I_t) or I_t == 0:
            continue
        units += -a / I_t  # contribution -> +units; distribution -> -units
        if units < 0:
            shadow_negative = True
            break

    T = max(pd.Timestamp(cf_dates.max()), pd.Timestamp(rvpi_date))
    I_T = bench_idx.asof(T)
    shadow_nav = units * I_T if not pd.isna(I_T) else float('nan')
    ln_pme = rvpi_nav / shadow_nav if (shadow_nav and shadow_nav > 0) else float('nan')
    return {
        'ln_pme': ln_pme,
        'shadow_nav': float(shadow_nav) if not pd.isna(shadow_nav) else float('nan'),
        'shadow_negative': shadow_negative,
    }


def compute_vintage_stats(
    cf: pd.DataFrame, navs: pd.DataFrame,
    asof: pd.Timestamp,
    labeler: Optional[Callable[[str], str]] = None,  # default: generic _short_label
) -> pd.DataFrame:
    """Per-vintage cash-flow statistics (benchmark-free)."""
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

        # XIRR with NAV as synthetic terminal distribution
        irr_dates = pd.concat([g['date'], pd.Series([max(g['date'].max(), rvpi_date)])], ignore_index=True)
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


def cap_weighted_aggregates(
    stats: pd.DataFrame, bench_labels: list[str],
    dpi_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Capital-weighted KS-PME and Direct Alpha across mature vintages (DPI > threshold).
    Weight = contributions.
    """
    mature = stats[stats['DPI'] > dpi_threshold].dropna(subset=['contributions']).copy()
    total_c = mature['contributions'].sum()

    rows = []
    for lbl in bench_labels:
        ks_col = f'KS_PME_{lbl}'
        da_col = f'Direct_Alpha_{lbl}'
        valid = mature.dropna(subset=[ks_col, da_col])
        w = valid['contributions']
        ks_wtd = (valid[ks_col] * w).sum() / w.sum() if w.sum() > 0 else float('nan')
        da_wtd = (valid[da_col] * w).sum() / w.sum() if w.sum() > 0 else float('nan')
        rows.append({
            'benchmark': lbl,
            'n_mature_vintages': len(valid),
            'total_contributions_bn': float(w.sum() / 1e9),
            'cap_wtd_KS_PME': ks_wtd,
            'cap_wtd_Direct_Alpha': da_wtd,
        })
    return pd.DataFrame(rows)
