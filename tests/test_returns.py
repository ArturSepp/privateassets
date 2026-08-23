"""
Tests for return reconstruction, reporting-frequency checks, and pooling.

The policy under test: one panel reports at one frequency, every return spans
exactly one period, and a panel that violates that is split rather than
interpolated.
"""

# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
from privateassets.matf import (
    infer_reporting_frequency,
    nav_implied_returns,
    pool_vintage_returns,
    split_by_reporting_frequency,
)
from tests.synthetic_data import make_cash_flows


def _fund(label, nav_by_date, flows=()):
    navs = pd.DataFrame([{'fund': label, 'vintage_label': label, 'date': d, 'nav': v}
                         for d, v in nav_by_date])
    rows = [{'fund': label, 'vintage_label': label, 'date': d, 'amount': a,
             'kind': 'contribution' if a < 0 else 'distribution'} for d, a in flows]
    if not rows:
        rows = [{'fund': label, 'vintage_label': label, 'date': nav_by_date[0][0],
                 'amount': -0.0, 'kind': 'contribution'}]
    return pd.DataFrame(rows), navs


def test_a_doubling_nav_with_no_flows_is_a_hundred_percent_return():
    """With no flows modified Dietz reduces to the plain NAV ratio."""
    q = pd.date_range('2020-03-31', periods=3, freq='QE')
    cf, navs = _fund('F', [(q[0], 100.0), (q[1], 200.0), (q[2], 300.0)])
    returns, capital = nav_implied_returns(cf, navs)
    assert returns.loc[q[1], 'F'] == pytest.approx(1.0)
    assert returns.loc[q[2], 'F'] == pytest.approx(0.5)
    assert capital.loc[q[1], 'F'] == pytest.approx(100.0)


def test_a_flat_nav_with_no_flows_is_a_zero_return():
    q = pd.date_range('2020-03-31', periods=2, freq='QE')
    cf, navs = _fund('F', [(q[0], 100.0), (q[1], 100.0)])
    returns, _ = nav_implied_returns(cf, navs)
    assert returns.loc[q[1], 'F'] == pytest.approx(0.0)


def test_a_distribution_is_not_a_loss():
    """Paying out capital leaves the return unchanged, which is the point of Dietz."""
    q = pd.date_range('2020-03-31', periods=2, freq='QE')
    cf, navs = _fund('F', [(q[0], 100.0), (q[1], 50.0)], flows=[(q[1], 50.0)])
    returns, _ = nav_implied_returns(cf, navs)
    assert returns.loc[q[1], 'F'] == pytest.approx(0.0, abs=1e-12)


def test_a_contribution_is_not_a_gain():
    """NAV doubling because capital was called is a zero return, not a 100% one."""
    q = pd.date_range('2020-03-31', periods=2, freq='QE')
    cf, navs = _fund('F', [(q[0], 100.0), (q[1], 200.0)], flows=[(q[1], -100.0)])
    returns, _ = nav_implied_returns(cf, navs)
    assert returns.loc[q[1], 'F'] == pytest.approx(0.0, abs=1e-12)


def test_mid_period_flows_get_half_weight_in_the_denominator():
    q = pd.date_range('2020-03-31', periods=2, freq='QE')
    cf, navs = _fund('F', [(q[0], 100.0), (q[1], 220.0)], flows=[(q[1], -100.0)])
    _, capital = nav_implied_returns(cf, navs)
    assert capital.loc[q[1], 'F'] == pytest.approx(150.0)  # 100 + 0.5 * 100


def test_a_skipped_period_is_rejected_not_papered_over():
    """A gap means the panel is not reported at the stated frequency.

    The old behaviour computed 130/110 across two quarters and filed it as a
    quarterly return, which corrupts every annualisation built on the series.
    """
    q = pd.date_range('2020-03-31', periods=4, freq='QE')
    cf, navs = _fund('F', [(q[0], 100.0), (q[1], 110.0), (q[3], 130.0)])
    with pytest.raises(ValueError, match='skip a QE period'):
        nav_implied_returns(cf, navs)


def test_the_rejection_names_the_offending_vintages_and_points_at_the_remedy():
    q = pd.date_range('2020-03-31', periods=4, freq='QE')
    cf_a, navs_a = _fund('compliant', [(d, 100.0 + i) for i, d in enumerate(q)])
    cf_b, navs_b = _fund('sparse', [(q[0], 100.0), (q[1], 110.0), (q[3], 130.0)])
    cf = pd.concat([cf_a, cf_b], ignore_index=True)
    navs = pd.concat([navs_a, navs_b], ignore_index=True)
    with pytest.raises(ValueError) as excinfo:
        nav_implied_returns(cf, navs)
    message = str(excinfo.value)
    assert "'sparse': 1" in message      # named, with how many periods it skips
    assert 'compliant' not in message    # and the well-behaved vintage is not blamed
    assert 'split_by_reporting_frequency' in message


def test_dropping_the_affected_periods_is_available_but_opt_in():
    q = pd.date_range('2020-03-31', periods=4, freq='QE')
    cf, navs = _fund('F', [(q[0], 100.0), (q[1], 110.0), (q[3], 130.0)])
    returns, _ = nav_implied_returns(cf, navs, require_regular=False)
    assert returns.loc[q[1], 'F'] == pytest.approx(0.10)
    assert np.isnan(returns.loc[q[2], 'F'])
    assert np.isnan(returns.loc[q[3], 'F'])  # never a two-quarter return


def test_nothing_is_forward_filled():
    """No mark is invented, so no period reads as an artificial zero return."""
    q = pd.date_range('2020-03-31', periods=4, freq='QE')
    cf, navs = _fund('F', [(q[0], 100.0), (q[1], 110.0), (q[3], 130.0)])
    returns, capital = nav_implied_returns(cf, navs, require_regular=False)
    assert not (returns.loc[q[2:], 'F'] == 0.0).any()
    assert capital.loc[q[2], 'F'] == 0.0


def test_a_semi_annual_panel_is_estimated_on_its_own_frequency():
    """Reported twice a year means a semi-annual grid, not an interpolated quarterly one."""
    dates = pd.date_range('2020-06-30', periods=6, freq='2QE')
    cf, navs = _fund('S', [(d, 100.0 * 1.05 ** i) for i, d in enumerate(dates)])
    returns, _ = nav_implied_returns(cf, navs, freq='2QE')
    assert returns['S'].notna().sum() == 5
    assert returns['S'].dropna().iloc[0] == pytest.approx(0.05)


def test_a_semi_annual_panel_is_rejected_on_a_quarterly_grid():
    dates = pd.date_range('2020-06-30', periods=6, freq='2QE')
    cf, navs = _fund('S', [(d, 100.0 * 1.05 ** i) for i, d in enumerate(dates)])
    with pytest.raises(ValueError, match='skip a QE period'):
        nav_implied_returns(cf, navs, freq='QE')


def test_reporting_frequency_is_inferred_per_vintage():
    q = pd.date_range('2020-03-31', periods=8, freq='QE')
    half = pd.date_range('2020-06-30', periods=4, freq='2QE')
    cf_a, navs_a = _fund('quarterly', [(d, 100.0) for d in q])
    cf_b, navs_b = _fund('semiannual', [(d, 100.0) for d in half])
    frequency = infer_reporting_frequency(pd.concat([navs_a, navs_b], ignore_index=True))
    assert frequency['quarterly'] == pytest.approx(3.0)
    assert frequency['semiannual'] == pytest.approx(6.0)


def test_a_single_mark_has_no_inferable_frequency():
    cf, navs = _fund('F', [(pd.Timestamp('2020-03-31'), 100.0)])
    assert np.isnan(infer_reporting_frequency(navs)['F'])


def test_a_mixed_panel_splits_into_homogeneous_groups():
    """The remedy for a mixed panel: estimate each frequency separately."""
    q = pd.date_range('2020-03-31', periods=8, freq='QE')
    half = pd.date_range('2020-06-30', periods=4, freq='2QE')
    cf_a, navs_a = _fund('quarterly', [(d, 100.0 * 1.02 ** i) for i, d in enumerate(q)])
    cf_b, navs_b = _fund('semiannual', [(d, 100.0 * 1.04 ** i) for i, d in enumerate(half)])
    cf = pd.concat([cf_a, cf_b], ignore_index=True)
    navs = pd.concat([navs_a, navs_b], ignore_index=True)

    groups = split_by_reporting_frequency(cf, navs)
    assert sorted(groups) == [3, 6]

    quarterly_cf, quarterly_navs = groups[3]
    returns, _ = nav_implied_returns(quarterly_cf, quarterly_navs, freq='QE')
    assert returns['quarterly'].notna().sum() == 7

    semi_cf, semi_navs = groups[6]
    returns, _ = nav_implied_returns(semi_cf, semi_navs, freq='2QE')
    assert returns['semiannual'].notna().sum() == 3


def test_pooling_weights_by_capital_at_work():
    """A large vintage moves the pooled return more than a small one."""
    q = pd.date_range('2020-03-31', periods=2, freq='QE')
    returns = pd.DataFrame({'big': [np.nan, 0.10], 'small': [np.nan, -0.10]}, index=q)
    capital = pd.DataFrame({'big': [0.0, 900.0], 'small': [0.0, 100.0]}, index=q)
    pooled = pool_vintage_returns(returns, capital)
    assert pooled.loc[q[1]] == pytest.approx((0.10 * 900 - 0.10 * 100) / 1000)


def test_pooling_ignores_vintages_with_no_capital():
    q = pd.date_range('2020-03-31', periods=1, freq='QE')
    returns = pd.DataFrame({'live': [0.05], 'dormant': [99.0]}, index=q)
    capital = pd.DataFrame({'live': [100.0], 'dormant': [0.0]}, index=q)
    assert pool_vintage_returns(returns, capital).loc[q[0]] == pytest.approx(0.05)


def test_pooling_drops_a_period_nobody_reported():
    q = pd.date_range('2020-03-31', periods=2, freq='QE')
    returns = pd.DataFrame({'a': [0.05, np.nan]}, index=q)
    capital = pd.DataFrame({'a': [100.0, 0.0]}, index=q)
    pooled = pool_vintage_returns(returns, capital)
    assert len(pooled) == 1 and q[1] not in pooled.index


def test_pooling_rejects_misaligned_frames():
    q = pd.date_range('2020-03-31', periods=2, freq='QE')
    with pytest.raises(ValueError, match='share an index and columns'):
        pool_vintage_returns(pd.DataFrame({'a': [0.1, 0.2]}, index=q),
                             pd.DataFrame({'b': [1.0, 1.0]}, index=q))


def test_runs_on_the_synthetic_multi_vintage_panel():
    cf, navs = make_cash_flows()
    assert (infer_reporting_frequency(navs) == 3.0).all()
    returns, capital = nav_implied_returns(cf, navs)
    pooled = pool_vintage_returns(returns, capital)
    assert set(returns.columns) == set(cf['vintage_label'].unique())
    assert len(pooled) > 20
    assert pooled.notna().all()
    assert pooled.abs().max() < 5.0  # no exploding denominators


def test_rejects_an_unsupported_frequency():
    cf, navs = make_cash_flows()
    with pytest.raises(ValueError, match='freq must be one of'):
        nav_implied_returns(cf, navs, freq='W')


def test_rejects_a_malformed_frame():
    cf, navs = make_cash_flows()
    with pytest.raises(ValueError, match='missing columns'):
        nav_implied_returns(cf.drop(columns=['amount']), navs)
