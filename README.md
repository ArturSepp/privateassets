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
                                matf_deflator, rolling_ewma_quarterly_covar,
                                vintage_direct_alpha)

quarter_ends = pd.DatetimeIndex(factor_levels.resample('QE').last().index)
quarterly = np.log(factor_levels.resample('QE').last()).diff().fillna(0.0)
cum_log_factor = factor_log_levels_panel(quarterly).values
cum_log_rf = np.log1p(rf_quarterly).cumsum().values
sigma_by_quarter = rolling_ewma_quarterly_covar(factor_levels, quarter_ends)

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
- `rolling_ewma_quarterly_covar` (36-month span, full-window mean) and
  `build_rolling_sigma` (60-month span, EWMA-rolling mean, through
  `qis.estimate_rolling_ewma_covar`) do **not** agree numerically. State which
  one a result uses.

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

85 tests, no network and no data files. `privateassets/tests/synthetic_data.py`
draws a seeded panel carrying the defects real panels carry: irregular cash-flow
dates, a J-curve, unrealised residual NAVs, and a factor panel that starts after
the first fund does.

Tests needing the `[factors]` extra skip rather than fail, so a core install
stays green.

Four of the tests are enforcement rather than behaviour — they fail the suite if
the package imports with a filesystem side effect, documents an argument it does
not take, ships a proprietary identifier, or imports a competing analytics stack.

## Status

`0.2.0` covers the estimator: classical PME measures, the MATF deflator, the
factor-loading fit, and the panel MLE for the unsmoothing coefficient. The
bootstrap inference layer and the reporting stack are not in this release. See
`CHANGELOG.md`.

## Citation

See `CITATION.cff`.
