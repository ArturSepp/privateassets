"""Tests for the classical single-benchmark PME measures."""

# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
from privateassets.matf import (
    cap_weighted_aggregates,
    cf_with_terminal_for_vintage,
    compute_vintage_stats,
    direct_alpha,
    ks_pme,
    long_nickels_pme,
    vintage_direct_alpha,
    xirr,
)
from tests.synthetic_data import make_cash_flows


def test_xirr_recovers_a_known_rate():
    """A doubling over one calendar year is a 100% IRR, on an ACT/365.25 count.

    2020 is a leap year, so the horizon is 366/365.25 years and the exact answer
    is 2**(365.25/366) - 1. Pinning the day count is the point of the test: the
    convention is stated, not implied.
    """
    dates = pd.Series([pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')])
    amounts = pd.Series([-100.0, 200.0])
    expected = 2.0 ** (365.25 / 366.0) - 1.0
    assert xirr(dates, amounts) == pytest.approx(expected, rel=1e-6)
    assert expected == pytest.approx(0.99716, abs=1e-5)


def test_xirr_matches_compounded_growth():
    """A 3x over exactly 4 years is 3**0.25 - 1."""
    dates = pd.Series([pd.Timestamp('2010-01-01'), pd.Timestamp('2014-01-01')])
    amounts = pd.Series([-1.0, 3.0])
    expected = 3.0 ** (365.25 / (4 * 365.25 + 1)) - 1.0
    assert xirr(dates, amounts) == pytest.approx(expected, rel=1e-3)


def test_xirr_returns_nan_when_no_sign_change():
    """All-positive flows have no internal rate of return."""
    dates = pd.Series([pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')])
    assert np.isnan(xirr(dates, pd.Series([100.0, 200.0])))


def test_xirr_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match='must align'):
        xirr(pd.Series([pd.Timestamp('2020-01-01')]), pd.Series([1.0, 2.0]))


def test_ks_pme_is_one_when_the_fund_tracks_the_benchmark():
    """A fund earning exactly the benchmark return has a PME of 1."""
    bench = pd.Series([100.0, 200.0],
                      index=[pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')])
    dates = pd.Series([pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')])
    amounts = pd.Series([-100.0, 200.0])
    pme = ks_pme(dates, amounts, rvpi_nav=0.0,
                 rvpi_date=pd.Timestamp('2021-01-01'), bench_idx=bench)
    assert pme == pytest.approx(1.0)


def test_ks_pme_above_one_when_the_fund_beats_the_benchmark():
    bench = pd.Series([100.0, 200.0],
                      index=[pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')])
    dates = pd.Series([pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')])
    pme = ks_pme(dates, pd.Series([-100.0, 300.0]), rvpi_nav=0.0,
                 rvpi_date=pd.Timestamp('2021-01-01'), bench_idx=bench)
    assert pme == pytest.approx(1.5)


def test_direct_alpha_is_zero_when_the_fund_tracks_the_benchmark():
    """Deflating by the benchmark leaves flows that net to zero at alpha = 0."""
    idx = pd.date_range('2015-12-31', periods=21, freq='QE')
    bench = pd.Series(100.0 * 1.02 ** np.arange(21), index=idx)
    dates = pd.Series([idx[0], idx[-1]])
    amounts = pd.Series([-100.0, 100.0 * float(bench.iloc[-1] / bench.iloc[0])])
    assert direct_alpha(dates, amounts, 0.0, idx[-1], bench) == pytest.approx(0.0, abs=1e-6)


def test_long_nickels_flags_a_negative_shadow_position():
    """Distributions outrunning the benchmark drive the shadow units negative."""
    idx = pd.date_range('2015-12-31', periods=9, freq='QE')
    bench = pd.Series(100.0 * 1.001 ** np.arange(9), index=idx)
    dates = pd.Series([idx[0], idx[2]])
    result = long_nickels_pme(dates, pd.Series([-10.0, 500.0]), 5.0, idx[-1], bench)
    assert result['shadow_negative'] is True
    assert np.isnan(result['ln_pme'])


def test_vintage_stats_multiples_are_internally_consistent():
    """TVPI is DPI plus RVPI, by construction."""
    cf, navs = make_cash_flows()
    stats = compute_vintage_stats(cf, navs)
    assert len(stats) == cf['fund'].nunique()
    assert np.allclose(stats['TVPI'], stats['DPI'] + stats['RVPI'])
    assert (stats['contributions'] > 0).all()


def test_vintage_stats_rejects_a_malformed_frame():
    with pytest.raises(ValueError, match='missing columns'):
        compute_vintage_stats(pd.DataFrame({'fund': ['a']}), pd.DataFrame())


def test_cap_weighted_aggregates_applies_the_dpi_gate():
    """The unrealised fund is excluded, so the mature count is one short."""
    cf, navs = make_cash_flows()
    stats = compute_vintage_stats(cf, navs)
    stats['KS_PME_BENCH'] = 1.2
    stats['Direct_Alpha_BENCH'] = 0.03
    agg = cap_weighted_aggregates(stats, ['BENCH'], dpi_threshold=0.8)
    assert agg.loc[0, 'n_mature_vintages'] == int((stats['DPI'] > 0.8).sum())
    assert agg.loc[0, 'cap_wtd_KS_PME'] == pytest.approx(1.2)


def test_cap_weighted_aggregates_rejects_a_missing_benchmark():
    cf, navs = make_cash_flows()
    stats = compute_vintage_stats(cf, navs)
    with pytest.raises(ValueError, match='missing columns'):
        cap_weighted_aggregates(stats, ['ABSENT'])


def test_terminal_date_is_appended_once():
    cf, navs = make_cash_flows()
    fund = cf['fund'].iloc[0]
    cf_g, rvpi_nav, rvpi_date, dates = cf_with_terminal_for_vintage(
        cf[cf['fund'] == fund], navs[navs['fund'] == fund])
    assert len(dates) == len(cf_g) + 1
    assert dates[-1] >= rvpi_date
    assert rvpi_nav > 0.0


def test_vintage_direct_alpha_matches_direct_alpha_on_a_benchmark_deflator():
    """The generalised solver reduces to Direct Alpha when the deflator is one index.

    This is the reduction that licenses calling the multi-factor measure a
    generalisation rather than a different statistic.
    """
    cf, navs = make_cash_flows()
    fund = cf['fund'].iloc[0]
    cf_g = cf[cf['fund'] == fund]
    nav_g = navs[navs['fund'] == fund]
    cf_v, rvpi_nav, rvpi_date, dates = cf_with_terminal_for_vintage(cf_g, nav_g)

    idx = pd.date_range('2000-12-31', '2026-12-31', freq='QE')
    bench = pd.Series(100.0 * 1.015 ** np.arange(len(idx)), index=idx)

    terminal = dates[-1]
    i_terminal = bench.asof(terminal)
    deflators = np.array([i_terminal / bench.asof(d) for d in dates])
    # The deflator convention is the reciprocal of the Direct Alpha scaling.
    generalised = vintage_direct_alpha(cf_v, rvpi_nav, dates, 1.0 / deflators)

    classical = direct_alpha(cf_v['date'], cf_v['amount'], rvpi_nav, rvpi_date, bench)
    assert generalised == pytest.approx(classical, abs=1e-4)


def test_vintage_direct_alpha_returns_nan_on_a_missing_deflator():
    """A single missing deflator invalidates the vintage rather than dropping a flow.

    Silently deleting a cash flow from the present value and solving on the
    survivors returns a number that looks like an alpha and is not one.
    """
    cf, navs = make_cash_flows()
    fund = cf['fund'].iloc[0]
    cf_v, rvpi_nav, _, dates = cf_with_terminal_for_vintage(
        cf[cf['fund'] == fund], navs[navs['fund'] == fund])
    deflators = np.ones(len(dates))
    deflators[3] = np.nan
    assert np.isnan(vintage_direct_alpha(cf_v, rvpi_nav, dates, deflators))


def test_vintage_direct_alpha_rejects_misaligned_deflators():
    cf, navs = make_cash_flows()
    fund = cf['fund'].iloc[0]
    cf_v, rvpi_nav, _, dates = cf_with_terminal_for_vintage(
        cf[cf['fund'] == fund], navs[navs['fund'] == fund])
    with pytest.raises(ValueError, match='must align'):
        vintage_direct_alpha(cf_v, rvpi_nav, dates, np.ones(len(dates) - 1))


def test_benchmark_measures_reject_misaligned_inputs():
    """Dates and amounts of different lengths raise rather than truncating.

    Silently zipping to the shorter of the two drops cash flows from the measure
    and returns a number that looks valid.
    """
    bench = pd.Series([100.0, 200.0],
                      index=[pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')])
    dates = pd.Series([pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')])
    amounts = pd.Series([-100.0])
    for measure in (ks_pme, direct_alpha, long_nickels_pme):
        with pytest.raises(ValueError, match='must align'):
            measure(dates, amounts, 0.0, pd.Timestamp('2021-01-01'), bench)
