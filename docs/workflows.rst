Core workflows
==============

Classical single-benchmark measures take a tidy cash-flow frame and a benchmark
index level series:

.. code-block:: python

   import pandas as pd
   from privateassets.matf import ks_pme, direct_alpha, compute_vintage_stats

   stats = compute_vintage_stats(cf=cash_flows, navs=navs)
   pme = ks_pme(cf_dates=cf['date'], cf_amounts=cf['amount'],
                rvpi_nav=nav, rvpi_date=nav_date, bench_idx=benchmark_levels)
   alpha = direct_alpha(cf_dates=cf['date'], cf_amounts=cf['amount'],
                        rvpi_nav=nav, rvpi_date=nav_date, bench_idx=benchmark_levels)


The whole estimator is one call. It returns every intermediate it computed, and
writes nothing:

.. code-block:: python

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


The covariance inside the deflator is point-in-time: deleting every observation
after a vintage closes leaves its alpha unchanged to 1e-12, which
``test_no_look_ahead_end_to_end`` asserts. The loadings are in-sample by
construction — one beta over the whole panel — and ``provenance`` says so, because
that caveat has to travel with the number.

Pass ``beta=`` to price against a loading vector you already have instead of
fitting one.

The stages are also available individually.

A fund reports marks and cash flows, not returns. Reconstruct the return series
first:

.. code-block:: python

   from privateassets.matf import (infer_reporting_frequency, nav_implied_returns,
                                   pool_vintage_returns, split_by_reporting_frequency)

   print(infer_reporting_frequency(navs))          # months between marks, per vintage
   returns, capital = nav_implied_returns(cf=cash_flows, navs=navs, freq='QE')
   quarterly_returns = pool_vintage_returns(returns, capital)


**One panel, one reporting frequency.** Every return spans exactly one period,
and nothing is forward-filled or interpolated. A panel whose vintages report at
different frequencies raises, naming the offenders:

.. code-block:: python

   groups = split_by_reporting_frequency(cf, navs)
   returns, capital = nav_implied_returns(*groups[6], freq='2QE')   # the semi-annual reporters


Estimating the groups separately is honest. Interpolating them onto a common
grid is an assumption about an unobserved path, and this package does not make
it for you.

Factor loadings come from the shrinkage estimator. The panel is short and the
factors are collinear, so an unconstrained least-squares beta is not usable:

.. code-block:: python

   from privateassets.matf import SignConstraint, fit_factor_betas

   fit = fit_factor_betas(asset_returns=quarterly_returns,
                          factor_returns=quarterly_factor_returns,
                          sign_constraints={'Equity': SignConstraint.POS,
                                            'Credit': SignConstraint.POS},
                          span=None)          # equal weights for in-sample identification
   beta = fit.beta.values


Needs the ``[factors]`` extra. The shrinkage target defaults to zero — a non-zero
prior is an economic view, so it is yours to pass, not the library's to assume.

The multi-factor measure then replaces the benchmark index with a deflator path
and solves the same root-finding step:

.. code-block:: python

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


Passing a single benchmark's reciprocal index ratio as the deflator returns the
classical Direct Alpha, to within root-finder tolerance. That reduction is
pinned by ``test_vintage_direct_alpha_matches_direct_alpha_on_a_benchmark_deflator``.
