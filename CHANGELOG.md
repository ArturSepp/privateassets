# Changelog

## 0.4.0 (2026-07-26)

Closes the last gap between raw fund reporting and the estimator: a fund
publishes marks and cash flows, not returns, and the package can now reconstruct
the return series the loadings are fitted on. Two long-standing defects in that
reconstruction are fixed rather than carried over.

### Added

- `nav_implied_quarterly_returns` — modified Dietz returns per vintage, with the
  denominator returned alongside as the pooling weight.
- `pool_vintage_returns` — capital-weighted aggregation of vintage returns into
  one manager series.

### Fixed

- **An unreported quarter no longer becomes a zero-return quarter.** The
  precursor forward-filled the previous mark, which produces a quarter of
  exactly zero return, manufacturing the smoothness the unsmoothing step then
  tries to remove. That inflates the estimated AR coefficient and so inflates
  the volatility uplift `1 / (1 - theta)`.
  `test_forward_filling_biases_the_smoothing_coefficient_upward` measures the
  effect. The old behaviour is available as `carry_navs_forward=True`.
- **A return spanning a reporting gap is no longer labelled quarterly.** Both
  ends of a quarter must carry a mark, so a two-quarter return is not mixed into
  a quarterly series under a quarterly label. Every annualisation and
  autocorrelation computed on such a series is otherwise wrong. For a genuinely
  infrequent reporter, interpolate with `qis.interpolate_infrequent_returns`.
- **The return index is now a complete quarterly range.** An unreported quarter
  was previously absent from the index rather than present and empty, so a gap
  was invisible to anything downstream.

### Changed

- **One covariance path.** `rolling_ewma_quarterly_covar` (36-month span,
  full-window mean, hand-rolled) and `build_rolling_sigma` (60-month span,
  through `qis`) disagreed numerically, and 0.0.1's config advertised the second
  while running neither. Both are replaced by `rolling_factor_covar`, which
  delegates to `qis.estimate_rolling_ewma_covar` at the documented 60-month
  production span. `DEFAULT_COVAR_SPAN_MONTHS` is now 60.

### Removed

- `rolling_ewma_quarterly_covar`, `build_rolling_sigma`, `DEFAULT_BURNIN_MONTHS`.

## 0.3.0 (2026-07-26)

**The two single-factor benchmarks were understating themselves by the risk-free
rate, which overstated every alpha measured against them.** Both are corrected
here, and the corrections are pinned by economic invariants rather than by
regression values.

### Fixed

- **KN24 benchmark deflator subtracted the risk-free rate twice.** The factor
  panel is excess by construction, so `beta * (r_market - rf)` subtracted rf from
  a series it had already been removed from. At `beta = 1` the deflator returned
  the market's *excess* return where the benchmark portfolio is the market
  itself. `test_unit_beta_benchmark_earns_the_total_market_return` fails under
  the old expression.
- **KN16 equity premium subtracted the risk-free rate twice.** `mu` was computed
  as `log E[exp(r_excess)] - rf`. Since the series is already excess, its
  lognormal mean *is* the premium. On a calibration with a true premium of 5.00%
  and rf of 3.00%, the old expression returned 1.99% against 4.99% corrected.
  Both `gamma = mu / sigma^2` and `delta` inherited the error.
- **KN16 SDF was evaluated on the excess market return.** `delta` is calibrated
  against the *total* return, so the kernel must see `rf + excess`. With the
  premium fix alone the market still priced to 1.0688; with both fixes
  `E[M Rf] = 0.9999` and `E[M Rm] = 1.0000`.
- **A near-degenerate variance no longer produces a garbage SDF.** A constant
  series has sample variance around 3e-36 rather than 0, which passed a
  `> 0` guard and returned `gamma` of order 1e33. Rejected below
  `MIN_ANNUAL_VARIANCE`.

### Added

- `kn24_benchmark_deflator`, `kn16_sdf_params`, `kn16_gpme_deflator` — the
  incumbents the MATF deflator substitutes for, so the comparison runs on the
  same cash flows.
- `bootstrap_factor_betas` and `BetaBootstrap` — block-resampling inference
  through `qis.generate_bootstrapped_indices`, replacing a hand-rolled
  stationary sampler. Indices are drawn once and applied to the asset series and
  the factor panel together, so each draw preserves the pairing the loading
  measures.
- `horizon_indices` — the shared date-to-panel mapping. Every deflator now uses
  one convention, so a KN and a MATF deflator on the same cash flows differ in
  their pricing kernel and in nothing else.
- 22 tests, including that `E[M R] = 1` for both the risk-free asset and the
  market, that a resample with a broken pairing collapses the loading, and that
  the same seed reproduces an interval.

### Changed

- **Resampled results carry their provenance.** `BetaBootstrap` records the
  `qis` version and the seed, because `BootstrapType.STATIONARY` changed its
  wrapping at `qis 5.1.0` and an interval from an earlier version does not
  reproduce.
- **The bootstrap reports `share_at_zero`, not a p-value.** Under a binding sign
  constraint the optimiser piles mass on the boundary, so the old
  `p_two_sided` counted the constraint's own atom as evidence for the null,
  doubled it, and clipped at 1. The quantity is monotone in how binding the
  constraint is, not in the strength of the evidence, and it is now named and
  documented as such.
- Failed resample fits are counted in `num_failed` rather than becoming NaN, so
  a degenerate draw cannot silently shrink the effective sample.

## 0.2.0 (2026-07-26)

**`factorlasso` shipped a breaking rename that silently disabled the estimator.**
The extracted code called `LassoModelType.GROUP_LASSO_CLUSTERS`, which does not
exist in `factorlasso` 0.10.1 — the member is now
`HIERARCHICAL_CLUSTER_GROUP_LASSO`. Any caller of the old estimator raised
`AttributeError` before reaching the fit. Confirmed against the installed
package rather than inferred.

### Added

- `fit_factor_betas` — sign-constrained, cluster-shrunk factor loadings, the
  `beta` the deflator needs. 0.1.0 shipped `matf_deflator` with no way to
  produce its own input; this closes that gap and makes the package usable end
  to end.
- `FactorBetas` — frozen result container carrying the loadings, the economic
  intercept, the observation count, the in-sample factor covariance and the
  signs `factorlasso` derived.
- `SignConstraint` — string enum (`POS`, `NEG`, `ZERO`, `FREE`) replacing the
  bare string sign mask.
- 15 tests, including recovery of a known loading vector, that each sign
  constraint binds rather than passing vacuously, that a heavier penalty shrinks
  towards the origin, and that the estimator's output feeds the deflator without
  adaptation.

### Changed

- `fit_signed_ridge` is renamed `fit_factor_betas`. The old name described
  neither the penalty (a hierarchical cluster group lasso, not a ridge) nor the
  return value.
- **The shrinkage prior has no default.** The extracted code carried
  `{'Private Equity': 0.5}`, an unattributed house view that materially moves the
  loading. It is now a caller argument defaulting to zero.
- `span_freq_dict` and the warm-up are arguments rather than hardcoded
  production constants.
- The covariance is named `sigma_quarterly_insample`, because it is full-sample
  and feeding it to a deflator applied at a date is look-ahead. Use
  `rolling_ewma_quarterly_covar` there.
- `r2` is NaN when `factorlasso` does not report one. The previous fallback
  computed an unweighted in-sample R-squared against a span-weighted fit, which
  is a different statistic.
- `factorlasso` is imported lazily, so the core PME and deflator paths install
  and run without the `[factors]` extra.

### Removed

- The unused `cvxpy` import, and the module docstring's claim that the estimator
  is "cvxpy for sign-constrained ridge regression". It has been a `factorlasso`
  wrapper for some time.

## 0.1.0 (2026-07-26)

**The public surface is narrower than 0.0.1 advertised.** The estimation
pipeline (`Cfg`, `run`, the shrinkage-beta fit, the bootstrap inference layer and
the reporting stack) is withdrawn from the package. It did not run: a
find-and-replace had rewritten a local variable into a package path, raising
`NameError` on the first vintage. It returns in a later release once it runs and
reproduces.

### Added

- `privateassets.matf.__all__` — the package has a declared public API. Every
  name in it resolves and carries a docstring, enforced by
  `test_all_advertised_names_exist` and `test_every_public_callable_is_documented`.
- `matf_deflator` — one MATF deflator, replacing the `matf_multi_factor_deflator`
  and `matf_deflator_fast` pair.
- `vintage_direct_alpha` — Direct Alpha against an arbitrary deflator path,
  renamed from `vintage_da_fast`.
- `build_rolling_sigma` — the `qis.estimate_rolling_ewma_covar` covariance path,
  moved in from the withdrawn `_rolling_covar` module.
- `privateassets/tests/` — 69 tests over a seeded synthetic panel. No network, no
  data files.
- Enforcement tests: no import-time filesystem access, no undocumented public
  callable, no documented argument absent from its signature, no proprietary
  identifier, no competing analytics stack, no `optimalportfolios` import.
- Packaging metadata: licence, classifiers, `[factors]`, `[excel]`, `[plot]` and
  `[dev]` extras, `CITATION.cff`, CI.

### Changed

- **`matf_deflator` returns NaN for a cash-flow date before the factor panel
  starts.** `matf_deflator_fast` clipped the index to zero, pinning such a date to
  the first quarter end and returning a finite deflator for a horizon that does
  not exist. Any vintage whose first call preceded the panel changes value.
- **`vintage_direct_alpha` uses `xtol=1e-8, maxiter=200`.** `vintage_da_fast` used
  `1e-7, 100`, a tolerance ten times looser than the point estimate it was
  bootstrapping. Alphas move in the eighth decimal.
- **`vintage_direct_alpha` raises on misaligned deflators** rather than returning
  NaN, so a length mismatch is a bug report and not a missing number.
- The deflator covariance calls `qis.compute_ewm_covar` rather than
  `factorlasso.compute_ewm_covar`. Verified bit-identical on a 200x4 panel at
  span 36, maximum absolute difference 0.0, so no result changes.
- `load_navs` defaults to the column `Net Asset Value Amount`. 0.0.1 defaulted to
  `Net Asset Value Amout`, a typo carried from one workbook. Sheet and column
  names are now arguments, and a missing column raises `ValueError` naming it.
- `compute_vintage_stats` and `cf_with_terminal_for_vintage` drop their unused
  `asof` argument.
- `requires-python` lowered to `>=3.10`, matching the rest of the stack.
- `factorlasso` moved from a mandatory dependency to the `[factors]` extra. The
  core PME and deflator paths do not import it.

### Removed

- `_unsmooth.apply_fixed_theta_unsmooth` and `PRODUCTION_THETA`. The function is
  `qis.unsmooth_returns_glm(returns, ar_order=1, theta=...)`, and the constant was
  an estimate from licensed data rather than a parameter of the method. Estimate
  it with `fit_panel_ar1`.
- `_pipeline` and `_rolling_covar` as public modules.
- The `illustrations` CLI, which 0.0.1's changelog advertised and the package did
  not contain.

### Fixed

- Importing the package no longer creates an `outputs/` directory in the caller's
  working directory, and no longer resolves paths from `os.getcwd()` at import.
- `xirr` raises on misaligned inputs instead of silently truncating.
