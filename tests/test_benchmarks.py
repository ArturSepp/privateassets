"""
Tests for the single-factor benchmark deflators.

The load-bearing tests are the economic invariants: a benchmark portfolio with
no market exposure must earn the risk-free rate, one with full exposure must
earn the market, and a calibrated SDF must price both to one. Each of these
fails under the risk-free double-subtraction that these functions were written
to correct, so they are not decoration.
"""

# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
from privateassets.matf import (
    kn16_gpme_deflator,
    kn16_sdf_params,
    kn24_benchmark_deflator,
)
from tests.synthetic_data import make_factor_levels, make_rf_quarterly

RF_SIMPLE_QUARTERLY = 0.0125  # 5% a year, large enough that a double subtraction shows


def _panel(rf_quarterly: float = RF_SIMPLE_QUARTERLY):
    """Cumulative log excess equity and log risk-free on a quarter-end grid."""
    levels = make_factor_levels()
    quarter_ends = pd.DatetimeIndex(levels.resample('QE').last().index)
    excess = np.log(levels['Equity'].resample('QE').last()).diff().fillna(0.0)
    rf = make_rf_quarterly(quarter_ends, rate=rf_quarterly * 4)
    return (quarter_ends, excess.cumsum().values, np.log1p(rf).cumsum().values,
            excess, rf)


def test_zero_beta_benchmark_earns_the_risk_free_rate():
    """No market exposure means the benchmark portfolio is cash."""
    quarter_ends, cum_ex, cum_rf, _, _ = _panel()
    out = kn24_benchmark_deflator(cf_dates=list(quarter_ends[[20, 40, 60]]),
                                  t0=quarter_ends[20], cum_log_equity_excess=cum_ex,
                                  cum_log_rf=cum_rf, quarter_ends=quarter_ends,
                                  beta=0.0, sigma2_annual=0.04)
    expected = [float(np.exp(cum_rf[i] - cum_rf[20])) for i in (20, 40, 60)]
    assert out == pytest.approx(expected)


def test_unit_beta_benchmark_earns_the_total_market_return():
    """Full market exposure means the benchmark is the market, risk-free included.

    This is the invariant the double subtraction broke: with the bug the
    deflator returned the *excess* return, understating the benchmark by the
    accrued risk-free rate and so overstating any alpha measured against it.
    """
    quarter_ends, cum_ex, cum_rf, _, _ = _panel()
    i0, i1 = 20, 60
    out = kn24_benchmark_deflator(cf_dates=[quarter_ends[i1]], t0=quarter_ends[i0],
                                  cum_log_equity_excess=cum_ex, cum_log_rf=cum_rf,
                                  quarter_ends=quarter_ends, beta=1.0, sigma2_annual=0.04)
    total = (cum_rf[i1] - cum_rf[i0]) + (cum_ex[i1] - cum_ex[i0])
    assert out[0] == pytest.approx(float(np.exp(total)))
    # and it is strictly above the excess-only value the bug produced
    assert out[0] > float(np.exp(cum_ex[i1] - cum_ex[i0]))


def test_jensen_term_vanishes_at_beta_zero_and_one():
    """The convexity correction beta(beta-1) is zero at both unlevered points."""
    quarter_ends, cum_ex, cum_rf, _, _ = _panel()
    for beta in (0.0, 1.0):
        low = kn24_benchmark_deflator([quarter_ends[60]], quarter_ends[20], cum_ex,
                                      cum_rf, quarter_ends, beta, sigma2_annual=0.0)
        high = kn24_benchmark_deflator([quarter_ends[60]], quarter_ends[20], cum_ex,
                                       cum_rf, quarter_ends, beta, sigma2_annual=0.25)
        assert low[0] == pytest.approx(high[0])


def test_leverage_above_one_is_penalised_by_the_convexity_term():
    """A levered position loses to volatility drag, so beta > 1 falls with variance."""
    quarter_ends, cum_ex, cum_rf, _, _ = _panel()
    low = kn24_benchmark_deflator([quarter_ends[60]], quarter_ends[20], cum_ex,
                                  cum_rf, quarter_ends, 2.0, sigma2_annual=0.0)
    high = kn24_benchmark_deflator([quarter_ends[60]], quarter_ends[20], cum_ex,
                                   cum_rf, quarter_ends, 2.0, sigma2_annual=0.25)
    assert high[0] < low[0]


def test_kn24_shares_the_out_of_panel_convention():
    """A pre-panel date is NaN here exactly as it is in the MATF deflator."""
    quarter_ends, cum_ex, cum_rf, _, _ = _panel()
    early = quarter_ends[0] - pd.Timedelta(days=400)
    out = kn24_benchmark_deflator([early, quarter_ends[30]], quarter_ends[10], cum_ex,
                                  cum_rf, quarter_ends, 1.0, 0.04)
    assert np.isnan(out[0])
    assert np.isfinite(out[1])


def test_kn24_rejects_misaligned_arrays():
    quarter_ends, cum_ex, cum_rf, _, _ = _panel()
    with pytest.raises(ValueError, match='cum_log_rf must have length'):
        kn24_benchmark_deflator([quarter_ends[10]], quarter_ends[5], cum_ex,
                                cum_rf[:-1], quarter_ends, 1.0, 0.04)


def test_sdf_recovers_the_equity_premium_from_an_excess_series():
    """gamma = mu / sigma^2, and mu must be the premium, not the premium minus rf.

    Drawn from a known data-generating process so the target is exact rather
    than whatever the sample happens to give.
    """
    rng = np.random.default_rng(11)
    rf_annual, sigma2_annual, premium = 0.03, 0.15 ** 2, 0.05
    log_mean_total = rf_annual + premium - 0.5 * sigma2_annual
    n = 200_000
    total = rng.normal(log_mean_total / 4, np.sqrt(sigma2_annual / 4), n)
    excess = pd.Series(total - rf_annual / 4)
    rf = pd.Series(np.full(n, rf_annual / 4))

    delta, gamma, sigma2 = kn16_sdf_params(excess, rf)
    assert sigma2 == pytest.approx(sigma2_annual, rel=0.02)
    assert gamma * sigma2 == pytest.approx(premium, abs=0.004)


def test_calibrated_sdf_prices_the_risk_free_asset_and_the_market():
    """E[M R] = 1 for both, which is what pins delta and gamma.

    Under the double subtraction the market prices to about 1.03 instead of 1.
    """
    rng = np.random.default_rng(5)
    rf_annual, sigma2_annual, premium = 0.03, 0.15 ** 2, 0.05
    log_mean_total = rf_annual + premium - 0.5 * sigma2_annual
    n = 400_000
    total_annual = rng.normal(log_mean_total, np.sqrt(sigma2_annual), n)

    excess_q = pd.Series(rng.normal((log_mean_total - rf_annual) / 4,
                                    np.sqrt(sigma2_annual / 4), n))
    delta, gamma, _ = kn16_sdf_params(excess_q, pd.Series(np.full(n, rf_annual / 4)))

    m = np.exp(delta - gamma * total_annual)
    assert float(np.mean(m * np.exp(rf_annual))) == pytest.approx(1.0, abs=0.01)
    assert float(np.mean(m * np.exp(total_annual))) == pytest.approx(1.0, abs=0.02)


def test_gpme_deflator_uses_the_total_market_return():
    """The kernel must see rf + excess, matching the delta it was calibrated with."""
    quarter_ends, cum_ex, cum_rf, excess, rf = _panel()
    delta, gamma, _ = kn16_sdf_params(excess, rf)
    i0, i1 = 20, 60
    out = kn16_gpme_deflator([quarter_ends[i1]], quarter_ends[i0], cum_ex, cum_rf,
                             quarter_ends, delta, gamma)
    r_total = (cum_ex[i1] - cum_ex[i0]) + (cum_rf[i1] - cum_rf[i0])
    horizon = max((quarter_ends[i1] - quarter_ends[i0]).days, 1) / 365.25
    assert out[0] == pytest.approx(float(np.exp(gamma * r_total - delta * horizon)))


def test_gpme_deflator_rises_with_the_market():
    """A higher market path discounts a payoff less, so the deflator increases."""
    quarter_ends, cum_ex, cum_rf, excess, rf = _panel()
    delta, gamma, _ = kn16_sdf_params(excess, rf)
    base = kn16_gpme_deflator([quarter_ends[60]], quarter_ends[20], cum_ex, cum_rf,
                              quarter_ends, delta, gamma)
    # a ramp, not a constant: adding a level to a cumulative series leaves every
    # horizon return unchanged and would perturb nothing
    steeper = cum_ex + np.linspace(0.0, 0.5, len(cum_ex))
    lifted = kn16_gpme_deflator([quarter_ends[60]], quarter_ends[20], steeper,
                                cum_rf, quarter_ends, delta, gamma)
    assert lifted[0] > base[0]


def test_sdf_params_reject_a_degenerate_sample():
    with pytest.raises(ValueError, match='at least 2 observations'):
        kn16_sdf_params(pd.Series([0.01]), pd.Series([0.01]))
    with pytest.raises(ValueError, match='variance must exceed'):
        kn16_sdf_params(pd.Series([0.01] * 10), pd.Series([0.01] * 10))
