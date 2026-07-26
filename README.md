# privateassets

Multi-factor, money-weighted PME for private-asset cash flows.

A single-benchmark PME divides fund cash flows by the return of one index. That
charges the fund for one exposure and credits everything else to skill. The MATF
deflator divides them by the return of a *tradable multi-factor portfolio*, so a
distressed-credit fund is measured against the credit and equity basket it
actually loaded on rather than against equities alone.

The package generalises Direct Alpha, KS-PME and GPME from one benchmark to a
multi-factor deflator, and ships the classical measures alongside so the two can
be compared on the same cash flows.

## Install

```bash
pip install privateassets
```

Sign-constrained shrinkage betas need the `factors` extra:

```bash
pip install "privateassets[factors]"
```

## Use

Classical single-benchmark measures take a tidy cash-flow frame and a benchmark
index level series:

```python
import pandas as pd
from privateassets.matf import ks_pme, direct_alpha, compute_vintage_stats

stats = compute_vintage_stats(cf=cash_flows, navs=navs)
pme = ks_pme(cf_dates=cf['date'], cf_amounts=cf['amount'],
             rvpi_nav=nav, rvpi_date=nav_date, bench_idx=benchmark_levels)
alpha = direct_alpha(cf_dates=cf['date'], cf_amounts=cf['amount'],
                     rvpi_nav=nav, rvpi_date=nav_date, bench_idx=benchmark_levels)
```

The whole estimator is one call. It returns every intermediate it computed, and
writes nothing:

```python
from privateassets.matf import estimate_matf_alpha

result = estimate_matf_alpha(cf=cash_flows, navs=navs,
                             factor_levels=factor_levels,   # excess index levels
                             rf_rate=rf_quarterly,
                             num_bootstrap=1000, seed=1)

print(result.cap_weighted_alpha)      # annualised, capital-weighted
print(result.vintage_alpha)           # per vintage
print(result.betas.beta)              # loadings
print(result.beta_bootstrap.lower)    # resampled interval
print(result.provenance)              # versions, seed, specification
```

The covariance inside the deflator is point-in-time: deleting every observation
after a vintage closes leaves its alpha unchanged to 1e-12, which
`test_no_look_ahead_end_to_end` asserts. The loadings are in-sample by
construction — one beta over the whole panel — and `provenance` says so, because
that caveat has to travel with the number.

Pass `beta=` to price against a loading vector you already have instead of
fitting one.

The stages are also available individually.

A fund reports marks and cash flows, not returns. Reconstruct the return series
first:

```python
from privateassets.matf import (infer_reporting_frequency, nav_implied_returns,
                                pool_vintage_returns, split_by_reporting_frequency)

print(infer_reporting_frequency(navs))          # months between marks, per vintage
returns, capital = nav_implied_returns(cf=cash_flows, navs=navs, freq='QE')
quarterly_returns = pool_vintage_returns(returns, capital)
```

**One panel, one reporting frequency.** Every return spans exactly one period,
and nothing is forward-filled or interpolated. A panel whose vintages report at
different frequencies raises, naming the offenders:

```python
groups = split_by_reporting_frequency(cf, navs)
returns, capital = nav_implied_returns(*groups[6], freq='2QE')   # the semi-annual reporters
```

Estimating the groups separately is honest. Interpolating them onto a common
grid is an assumption about an unobserved path, and this package does not make
it for you.

Factor loadings come from the shrinkage estimator. The panel is short and the
factors are collinear, so an unconstrained least-squares beta is not usable:

```python
from privateassets.matf import SignConstraint, fit_factor_betas

fit = fit_factor_betas(asset_returns=quarterly_returns,
                       factor_returns=quarterly_factor_returns,
                       sign_constraints={'Equity': SignConstraint.POS,
                                         'Credit': SignConstraint.POS},
                       span=None)          # equal weights for in-sample identification
beta = fit.beta.values
```

Needs the `[factors]` extra. The shrinkage target defaults to zero — a non-zero
prior is an economic view, so it is yours to pass, not the library's to assume.

The multi-factor measure then replaces the benchmark index with a deflator path
and solves the same root-finding step:

```python
import numpy as np
from privateassets.matf import (cf_with_terminal_for_vintage, factor_log_levels_panel,
                                matf_deflator, rolling_factor_covar,
                                vintage_direct_alpha)

quarter_ends = pd.DatetimeIndex(factor_levels.resample('QE').last().index)
quarterly = np.log(factor_levels.resample('QE').last()).diff().fillna(0.0)
cum_log_factor = factor_log_levels_panel(quarterly).values
cum_log_rf = np.log1p(rf_quarterly).cumsum().values
sigma_by_quarter = rolling_factor_covar(factor_levels)

cf_v, rvpi_nav, rvpi_date, dates = cf_with_terminal_for_vintage(cf_g, nav_g)
deflators = matf_deflator(cf_dates=dates, t0=dates[0],
                          cum_log_factor=cum_log_factor, cum_log_rf=cum_log_rf,
                          quarter_ends=quarter_ends, beta=beta,
                          sigma_by_quarter=sigma_by_quarter,
                          sigma_default=sigma_burnin)
alpha = vintage_direct_alpha(cf_v, rvpi_nav, dates, deflators)
```

Passing a single benchmark's reciprocal index ratio as the deflator returns the
classical Direct Alpha, to within root-finder tolerance. That reduction is
pinned by `test_vintage_direct_alpha_matches_direct_alpha_on_a_benchmark_deflator`.

## Unsmoothing

Appraisal NAVs are reported with a lag and anchored to the previous mark, so
reported returns are a moving average of true ones. Estimate the AR(1)
coefficient across a panel of funds here, then apply it with `qis`:

```python
import qis
from privateassets.matf import fit_panel_ar1

result = fit_panel_ar1(demeaned_series_by_fund)
unsmoothed = qis.unsmooth_returns_glm(returns, ar_order=1, theta=result['theta_hat'])
```

The inversion itself is `qis.unsmooth_returns_glm`. This package does not carry a
second copy of it.

**The estimate is biased down on short panels.** Demeaning an AR(1) of length n
biases the coefficient by about `-(1 + 3θ)/n`. At 80 quarterly observations and
θ = 0.35 that is roughly 2.5 percentage points, and it flows into the volatility
inflation `1/(1-θ)`, understating unsmoothed risk. No bias correction is applied.
`test_short_panels_understate_theta_by_the_kendall_bias` pins the magnitude.

## Comparing against the single-factor incumbents

`kn24_benchmark_deflator` and `kn16_gpme_deflator` price the same cash flows
against one market index, so the multi-factor result can be reported next to
what it replaces. Both take the equity factor as an excess log return and add
the risk-free leg back where the economics needs a total return.

```python
from privateassets.matf import kn16_gpme_deflator, kn16_sdf_params

delta, gamma, sigma2 = kn16_sdf_params(equity_excess_log_returns, rf_quarterly)
kn_deflators = kn16_gpme_deflator(cf_dates=dates, t0=dates[0],
                                  cum_log_equity_excess=cum_log_equity,
                                  cum_log_rf=cum_log_rf, quarter_ends=quarter_ends,
                                  delta=delta, gamma=gamma)
kn_alpha = vintage_direct_alpha(cf_v, rvpi_nav, dates, kn_deflators)
```

## Inference

Loadings are resampled in blocks through `qis`, with the asset and its factors
resampled together:

```python
from privateassets.matf import bootstrap_factor_betas

boot = bootstrap_factor_betas(asset_returns, factor_returns,
                              num_samples=1000, block_size=12, seed=1)
print(boot.lower, boot.upper, boot.qis_version)
```

`share_at_zero` reports how often a sign constraint binds. It is not a p-value:
under a binding constraint the mass sits on the boundary, so the quantity tracks
the constraint, not the evidence.

## Conventions

- Factor inputs are **excess** log returns. The risk-free rate enters the
  deflator once, through its own term.
- Covariance is **point-in-time**: the matrix at a quarter end uses only returns
  observed by that date. `test_rolling_covariance_is_point_in_time` enforces it.
- Deflator matrices are in **quarterly** units and scale by the horizon in
  quarters.
- Day counts are ACT/365.25 throughout.
- A cash-flow date before the factor panel starts returns NaN. It is not pinned
  to the first quarter end.
- There is one covariance path, `rolling_factor_covar`, which delegates to
  `qis.estimate_rolling_ewma_covar` at a 60-month span. The 36-month
  full-window-mean variant that shipped in 0.1.0 is gone.
- One panel, one reporting frequency. Every return spans exactly one period,
  and nothing is forward-filled or interpolated.

## Dependencies

Built on [`qis`](https://pypi.org/project/qis/) for unsmoothing, covariance
estimation and resampling, and optionally on
[`factorlasso`](https://pypi.org/project/factorlasso/) for sign-constrained
shrinkage betas. It does not depend on `optimalportfolios`, which is a sibling.

**Licence note.** This package is MIT. `factorlasso` is GPL-3, so a redistributed
work combining the two takes on GPL-3 obligations. Installing the `factors` extra
is what creates that combination. The core PME and deflator paths do not import
it.

## Data

No data ships with this repository, and none may be added. Every input is
licensed and read from a path you supply. See `DATA_README.md`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

152 tests, no network and no data files. `privateassets/tests/synthetic_data.py`
draws a seeded panel carrying the defects real panels carry: irregular cash-flow
dates, a J-curve, unrealised residual NAVs, and a factor panel that starts after
the first fund does.

Tests needing the `[factors]` extra skip rather than fail, so a core install
stays green.

Four of the tests are enforcement rather than behaviour — they fail the suite if
the package imports with a filesystem side effect, documents an argument it does
not take, ships a proprietary identifier, or imports a competing analytics stack.

## Status

`0.5.0` runs from fund reporting to alpha in one call, and each stage is usable
on its own. The reporting and factsheet layer is not in this release. See
`CHANGELOG.md`.

Two caveats travel with every number and are recorded in `provenance`: the
loadings are in-sample, and the estimated smoothing coefficient is attenuated by
measurement noise in the J-curve period.

## Citation

See `CITATION.cff`.
