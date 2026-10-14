"""
Pooled-cohort extracts: the reader, the maturity filter and the bridge into the
tidy frames the estimator consumes.

The synthetic extract carries the traps the delivered file carries: five
percentile bands per cohort, a literal ``NA`` for a missing currency amount, and
an unrealised youngest vintage.
"""

# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
from privateassets.matf import (
    CF_COLUMNS,
    COHORT_COLUMNS,
    FULL_POOLED_BAND,
    NAV_COLUMNS,
    CohortCashFlows,
    build_cohorts,
    cohort_summary,
    cohorts_to_cash_flows,
    direct_alpha,
    estimate_matf_alpha,
    filter_mature,
    infer_reporting_frequency,
    ks_pme,
    load_cohort_extract,
    nav_implied_returns,
    pool_vintage_returns,
)
from tests.synthetic_data import make_cohort_extract, make_factor_levels, make_rf_quarterly


@pytest.fixture(scope='module')
def extract_path(tmp_path_factory):
    path = tmp_path_factory.mktemp('extract') / 'cohorts.csv'
    make_cohort_extract().to_csv(path, index=False)
    return path


@pytest.fixture(scope='module')
def extract(extract_path):
    return load_cohort_extract(extract_path)


@pytest.fixture(scope='module')
def cohorts(extract):
    return build_cohorts(extract)


def test_the_reader_normalises_columns_dates_and_units(extract):
    """columns are renamed, dates parsed, percent scaled to fractions, NA tokens read."""
    assert list(extract.columns) == COHORT_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(extract['date'])
    assert extract['vintage'].dtype.kind == 'i'
    # 95% of commitment called over the full pool, as a fraction
    full = extract[extract['band'] == FULL_POOLED_BAND]
    called = full.groupby(['group', 'vintage'])['paid_in'].sum()
    assert np.allclose(called.values, 0.95, atol=1e-6)
    # a missing flow is zero, never NaN
    assert extract['paid_in'].notna().all() and extract['dist'].notna().all()


def test_a_currency_column_reads_unscaled_and_keeps_its_missing_valuation(extract_path):
    """the NA token in the first quarter stays NaN in the valuation and zero in the flows."""
    usd = load_cohort_extract(extract_path, paid_in_col='PaidIn(MM USD)', dist_col='Dist(MM USD)',
                              nav_col='Valuation(MM USD)', scale=1.0)
    first = usd.sort_values('date').groupby(['group', 'band', 'vintage']).head(1)
    assert first['nav'].isna().all()
    assert (first['paid_in'] == 0.0).all()
    assert usd['paid_in'].sum() > 10.0 * 0.95 * 6  # currency amounts, not fractions


def test_a_missing_column_is_named(extract_path):
    with pytest.raises(ValueError, match='Valuation'):
        load_cohort_extract(extract_path, nav_col='Valuation(bps)')


def test_build_cohorts_takes_one_band_and_one_path_per_cohort(extract, cohorts):
    """five bands per cohort in the file, one path per cohort out."""
    assert len(cohorts) == 2 * 3
    assert all(len(c.dates) == 48 for c in cohorts.values())
    assert all(c.num_funds == 12 for c in cohorts.values())  # the full pool, not a quartile
    with pytest.raises(ValueError, match='band'):
        build_cohorts(extract, band='IRR, Percentile, Minimum: 0, Maximum: 50')
    with pytest.raises(ValueError, match='groups'):
        build_cohorts(extract, groups=['9:No Such Group'])
    only = build_cohorts(extract, groups=['2:Beta Funds'])
    assert set(k[0] for k in only) == {'2:Beta Funds'}


def test_cohort_multiples_by_hand():
    """DPI, TVPI and the residual ratio follow their definitions, NaN valuations ignored."""
    dates = pd.DatetimeIndex(['2001-03-31', '2001-06-30', '2001-09-30', '2001-12-31'])
    c = CohortCashFlows(group='g', vintage=2001, num_funds=3, dates=dates,
                        paid_in=np.array([0.5, 0.5, 0.0, 0.0]),
                        dist=np.array([0.0, 0.0, 0.6, 0.6]),
                        nav=np.array([0.5, 1.0, np.nan, 0.1]))
    assert c.cum_paid_in == pytest.approx(1.0)
    assert c.dpi == pytest.approx(1.2)
    assert c.tvpi == pytest.approx(1.3)
    assert c.residual_ratio == pytest.approx(0.1)
    assert c.final_nav_date == pd.Timestamp('2001-12-31')
    assert list(c.signed_flows().values) == pytest.approx([-0.5, -0.5, 0.6, 0.6])
    with pytest.raises(ValueError, match='dist'):
        CohortCashFlows(group='g', vintage=2001, num_funds=3, dates=dates,
                        paid_in=np.zeros(4), dist=np.zeros(3), nav=np.zeros(4))


def test_the_maturity_filter_drops_the_unrealised_vintage(cohorts):
    mature = filter_mature(cohorts, max_residual_ratio=0.10)
    assert set(mature) == {(g, v) for (g, v) in cohorts if v != 2007}
    assert filter_mature(cohorts, max_residual_ratio=2.0) == cohorts
    with pytest.raises(ValueError, match='non-negative'):
        filter_mature(cohorts, max_residual_ratio=-0.1)


def test_the_summary_has_one_row_per_cohort(cohorts):
    summary = cohort_summary(cohorts)
    assert len(summary) == len(cohorts)
    assert summary.loc[('1:Alpha Funds', 2001), 'n_periods'] == 48
    assert summary['residual_ratio'].loc[(slice(None), 2007)].gt(0.10).all()
    assert cohort_summary({}).empty


def test_the_bridge_produces_the_tidy_frames(cohorts):
    """cohorts become funds: schema, signs, totals and marks all line up."""
    cf, navs = cohorts_to_cash_flows(cohorts)
    assert list(cf.columns) == CF_COLUMNS and list(navs.columns) == NAV_COLUMNS
    assert (cf.loc[cf['kind'] == 'contribution', 'amount'] < 0).all()
    assert (cf.loc[cf['kind'] == 'distribution', 'amount'] > 0).all()
    for key, cohort in cohorts.items():
        rows = cf[cf['fund'] == cohort.label]
        assert rows['vintage_label'].eq(cohort.label).all()
        assert -rows.loc[rows['amount'] < 0, 'amount'].sum() == pytest.approx(cohort.cum_paid_in)
        assert rows.loc[rows['amount'] > 0, 'amount'].sum() == pytest.approx(cohort.cum_dist)
        marks = navs[navs['fund'] == cohort.label]
        assert len(marks) == int(np.isfinite(cohort.nav).sum())
    with pytest.raises(ValueError, match='empty'):
        cohorts_to_cash_flows({})


def test_the_frames_run_through_the_return_reconstruction(cohorts):
    """the chain every fund panel runs accepts a cohort panel unchanged."""
    cf, navs = cohorts_to_cash_flows(cohorts)
    months = infer_reporting_frequency(navs)
    assert (months == 3).all()
    returns, capital = nav_implied_returns(cf, navs, freq='QE', require_regular=True)
    pooled = pool_vintage_returns(returns, capital)
    assert len(pooled) > 40
    assert np.isfinite(pooled).all()


def test_single_benchmark_measures_price_a_cohort(cohorts):
    """a cohort's signed flows plus its residual go straight into KS-PME and Direct Alpha."""
    cohort = cohorts[('1:Alpha Funds', 2001)]
    flows = cohort.signed_flows()
    flat = pd.Series(100.0, index=flows.index)  # a flat benchmark leaves the wealth multiple
    pme = ks_pme(cf_dates=pd.Series(flows.index), cf_amounts=flows.reset_index(drop=True),
                 rvpi_nav=cohort.final_nav, rvpi_date=cohort.final_nav_date, bench_idx=flat)
    assert pme == pytest.approx(cohort.tvpi)
    alpha = direct_alpha(cf_dates=pd.Series(flows.index), cf_amounts=flows.reset_index(drop=True),
                         rvpi_nav=cohort.final_nav, rvpi_date=cohort.final_nav_date, bench_idx=flat)
    assert np.isfinite(alpha) and alpha > 0


def test_the_estimator_prices_cohorts_against_supplied_loadings(cohorts):
    """end to end on a core install: supplied beta, no loading fit, every mature cohort priced."""
    cf, navs = cohorts_to_cash_flows(filter_mature(cohorts))
    factor_levels = make_factor_levels(start='1999-12-31', end='2020-12-31')
    quarter_ends = factor_levels.resample('QE').last().index
    rf = make_rf_quarterly(quarter_ends)
    beta = pd.Series({'Equity': 0.8, 'Rates': 0.0, 'Credit': 0.3, 'Commodities': 0.0})
    result = estimate_matf_alpha(cf, navs, factor_levels=factor_levels, rf_rate=rf,
                                 beta=beta, theta=0.0, dpi_threshold=0.8,
                                 sigma_default=np.cov(np.log(factor_levels.resample('QE').last())
                                                      .diff().dropna().values.T))
    assert result.n_mature == 4
    assert np.isfinite(result.cap_weighted_alpha)
    assert set(result.vintage_alpha['vintage_label']) == {c.label for c in filter_mature(cohorts).values()}
