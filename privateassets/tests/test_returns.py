"""Tests for NAV-implied return reconstruction and vintage pooling."""

# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
from privateassets.matf import (
    fit_panel_ar1,
    nav_implied_quarterly_returns,
    pool_vintage_returns,
)
from privateassets.tests.synthetic_data import make_cash_flows


def _one_fund(nav_by_quarter, flows=()):
    """A single vintage with hand-specified marks and optional dated flows."""
    navs = pd.DataFrame([{'fund': 'F', 'vintage_label': 'F', 'date': d, 'nav': v}
                         for d, v in nav_by_quarter])
    rows = [{'fund': 'F', 'vintage_label': 'F', 'date': d, 'amount': a,
             'kind': 'contribution' if a < 0 else 'distribution'} for d, a in flows]
    if not rows:  # a vintage always has at least one call
        rows = [{'fund': 'F', 'vintage_label': 'F',
                 'date': nav_by_quarter[0][0], 'amount': -0.0, 'kind': 'contribution'}]
    return pd.DataFrame(rows), navs


def test_a_doubling_nav_with_no_flows_is_a_hundred_percent_return():
    """With no flows modified Dietz reduces to the plain NAV ratio."""
    q = pd.date_range('2020-03-31', periods=3, freq='QE')
    cf, navs = _one_fund([(q[0], 100.0), (q[1], 200.0), (q[2], 300.0)])
    returns, capital = nav_implied_quarterly_returns(cf, navs)
    assert returns.loc[q[1], 'F'] == pytest.approx(1.0)
    assert returns.loc[q[2], 'F'] == pytest.approx(0.5)
    assert capital.loc[q[1], 'F'] == pytest.approx(100.0)


def test_a_flat_nav_with_no_flows_is_a_zero_return():
    q = pd.date_range('2020-03-31', periods=2, freq='QE')
    cf, navs = _one_fund([(q[0], 100.0), (q[1], 100.0)])
    returns, _ = nav_implied_quarterly_returns(cf, navs)
    assert returns.loc[q[1], 'F'] == pytest.approx(0.0)


def test_a_distribution_is_not_a_loss():
    """Paying out capital leaves the return unchanged, which is the point of Dietz.

    NAV halves, but the money left the fund rather than being lost.
    """
    q = pd.date_range('2020-03-31', periods=2, freq='QE')
    cf, navs = _one_fund([(q[0], 100.0), (q[1], 50.0)], flows=[(q[1], 50.0)])
    returns, _ = nav_implied_quarterly_returns(cf, navs)
    assert returns.loc[q[1], 'F'] == pytest.approx(0.0, abs=1e-12)


def test_a_contribution_is_not_a_gain():
    """NAV doubling because capital was called is a zero return, not a 100% one."""
    q = pd.date_range('2020-03-31', periods=2, freq='QE')
    cf, navs = _one_fund([(q[0], 100.0), (q[1], 200.0)], flows=[(q[1], -100.0)])
    returns, _ = nav_implied_quarterly_returns(cf, navs)
    assert returns.loc[q[1], 'F'] == pytest.approx(0.0, abs=1e-12)


def test_mid_quarter_flows_get_half_weight_in_the_denominator():
    q = pd.date_range('2020-03-31', periods=2, freq='QE')
    cf, navs = _one_fund([(q[0], 100.0), (q[1], 220.0)], flows=[(q[1], -100.0)])
    _, capital = nav_implied_quarterly_returns(cf, navs)
    assert capital.loc[q[1], 'F'] == pytest.approx(150.0)  # 100 + 0.5 * 100


def test_an_unreported_quarter_is_missing_not_zero():
    """A quarter with no mark has an unknown return, and must not read as flat."""
    q = pd.date_range('2020-03-31', periods=4, freq='QE')
    cf, navs = _one_fund([(q[0], 100.0), (q[1], 110.0), (q[3], 130.0)])
    returns, _ = nav_implied_quarterly_returns(cf, navs)
    assert list(returns.index) == list(q)  # the gap quarter is present, and empty
    assert np.isnan(returns.loc[q[2], 'F'])
    assert returns.loc[q[1], 'F'] == pytest.approx(0.10)


def test_a_return_spanning_a_gap_is_not_labelled_quarterly():
    """130/110 over two quarters must not be reported as a one-quarter return.

    Both ends of a quarter must be marked, so the quarter after a gap is missing
    too. Otherwise the series mixes one- and two-quarter returns under one label,
    and every annualisation and autocorrelation built on it is wrong.
    """
    q = pd.date_range('2020-03-31', periods=4, freq='QE')
    cf, navs = _one_fund([(q[0], 100.0), (q[1], 110.0), (q[3], 130.0)])
    returns, _ = nav_implied_quarterly_returns(cf, navs)
    assert np.isnan(returns.loc[q[3], 'F'])


def test_carrying_navs_forward_manufactures_a_zero_return():
    """The opt-in flag reproduces the old behaviour, and this shows what it costs."""
    q = pd.date_range('2020-03-31', periods=4, freq='QE')
    cf, navs = _one_fund([(q[0], 100.0), (q[1], 110.0), (q[3], 130.0)])
    filled, _ = nav_implied_quarterly_returns(cf, navs, carry_navs_forward=True)
    assert filled.loc[q[2], 'F'] == pytest.approx(0.0)


def test_forward_filling_biases_the_smoothing_coefficient_upward():
    """Injected zero-return quarters raise the estimated theta.

    This is the mechanism, measured: a higher theta inflates the volatility
    uplift 1/(1-theta) and so overstates unsmoothed risk.
    """
    rng = np.random.default_rng(0)
    q = pd.date_range('2000-03-31', periods=160, freq='QE')
    nav, marks = 100.0, []
    for i, d in enumerate(q):
        nav *= float(np.exp(rng.normal(0.01, 0.05)))
        if i < 2 or rng.random() > 0.25:  # a quarter is occasionally unreported
            marks.append((d, nav))
    cf, navs = _one_fund(marks)

    sparse, _ = nav_implied_quarterly_returns(cf, navs)
    filled, _ = nav_implied_quarterly_returns(cf, navs, carry_navs_forward=True)

    def theta(frame):
        series = frame['F'].dropna().values
        return fit_panel_ar1([('F', series - series.mean())])['theta_hat']

    assert theta(filled) > theta(sparse)


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


def test_pooling_drops_a_quarter_nobody_reported():
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
    returns, capital = nav_implied_quarterly_returns(cf, navs)
    pooled = pool_vintage_returns(returns, capital)
    assert set(returns.columns) == set(cf['vintage_label'].unique())
    assert len(pooled) > 20
    assert pooled.notna().all()
    assert pooled.abs().max() < 5.0  # no exploding denominators


def test_rejects_a_malformed_frame():
    cf, navs = make_cash_flows()
    with pytest.raises(ValueError, match='missing columns'):
        nav_implied_quarterly_returns(cf.drop(columns=['amount']), navs)
