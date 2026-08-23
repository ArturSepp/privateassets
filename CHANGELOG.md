# Changelog

## Unreleased

### Changed

- Moved the installable package to `src/privateassets/` and the automated suite to
  top-level `tests/` without changing the `privateassets` import name or public API.

## 0.6.2 (2026-08-22)

### Changed

- Enforced the separation between automated pytest modules and source-only development runners.
  Future component runners use `privateassets/run/<subject>_local.py` with `Locals` and
  `run_local(local=...)`, and are excluded from built distributions.

## 0.6.1 (2026-07-28)

**The ORCID iD in `CITATION.cff` was not the author's.** It read
0000-0003-4083-4183 from the first release to this one; the correct iD is
0000-0002-7038-1748, which the other repositories in the stack already carry. An
iD in `CITATION.cff` is what Zenodo attaches a minted DOI to, so every release
archived before this one credits a stranger and does not appear on the author's
profile. Anyone who has already cited this package from `CITATION.cff` should
re-export the entry.

### Added

- `test_factorlasso_is_imported_lazily_everywhere` — no module may import the
  GPL-3 optional extra at import time, which is what keeps a core install MIT and
  working. A lazy import inside a function still passes, because that is the
  mechanism.
- `make_collinear_factor_panel` in the test data, and two tests over it. Every
  loading-fit test previously ran on near-orthogonal factors (maximum absolute
  correlation 0.19), so nothing exercised the regime the shrinkage estimator
  exists for.
- `test_the_citation_orcid_is_the_authors` and `test_the_citation_date_is_a_date`
  alongside the existing `test_the_release_triple_agrees`. The wrong iD survived
  every release because it was a string no test read; `date-released` is checked
  for ISO shape because a bare year passes a human's glance and sorts as nothing.
- `CONTRIBUTING.md` and `.github/ISSUE_TEMPLATE/bug_report.yml`. The bug template
  asks which install the reporter is on, because the `factors` extra changes which
  code paths run; `CONTRIBUTING.md` states the MIT/GPL-3 boundary as a rule a
  contributor can break, and asks a pull request to say when a number moves.

### Fixed

- CI installed `.[dev]`, which does not carry `factorlasso`, so `test_betas`,
  `test_estimator` and `test_inference` skipped in full on every push and the run
  reported green on 118 of 171 tests. The workflow now runs both installs, and
  each job asserts whether `factorlasso` is present rather than trusting the
  install to have worked. The lint step was labelled "lint changed files" while
  running `ruff check privateassets`, the whole package; the label now says what
  the step does.
- The `_betas.py` docstring claimed ordinary least squares "does not work" on a
  private-asset panel. Measured, it works and is simply less accurate: mean
  absolute loading error 0.0797 against 0.0483 at zero correlation, and 0.1696
  against 0.1290 at 0.98, with a wrong-signed loading in 1 of 200 fits. The
  docstring now carries the table and
  `test_shrinkage_beats_least_squares_under_collinearity` reproduces it.

### Changed

- Dependency floor raised to `qis>=5.2.1`, which carries the AR-residual fix
  reported under 0.6.0.

**No number in this package changes.** `qis 5.2.1` differs from `5.1.0` in
`bootstrap_numba.py` alone, and within that module only in `compute_ar_residuals`
and the index draw inside `bootstrap_ar_process`. This package calls neither. Its
resampling goes through `qis.generate_bootstrapped_indices`, which is untouched,
so bootstrap intervals are bit-identical across the two versions. The suite as it
stood at 0.6.0 — 169 tests — passed against `5.2.1` before anything else in this
release was written.

The floor is raised anyway because the defects were silent. `compute_ar_residuals`
left NaN in the residuals whenever the input had gaps, treated the observations
either side of a gap as one period apart, and returned one fewer row than
`bootstrap_ar_process` then drew indices over — and the consumer is `@njit` with
bounds checking off, so the out-of-bounds row was read rather than raising. A
future release of this package that reaches for the AR path should not be able to
resolve an older `qis`.

## 0.6.0 (2026-07-26)

The smoothing coefficient can now be bias-corrected, and the part of its bias
that no correction reaches is separated out and measured.

### Added

- `BiasCorrection`, `kendall_corrected_theta`, `bootstrap_corrected_theta`,
  `simulate_panel_ar1`, and a `bias_correction=` argument on `fit_panel_ar1` and
  `estimate_matf_alpha`. The default is `NONE`: an estimate is never adjusted
  without being asked.
- `theta_raw` on the result, alongside the measured `bias`, so a corrected
  number always carries what it was corrected from.

### Two biases, only one correctable

Demeaning a short AR(1) biases its coefficient by about `-(1 + 3 theta) / n`.
Measured over 60 replications on seeded panels, mean absolute error:

| true theta | n | raw | Kendall | bootstrap |
|---|---|---|---|---|
| 0.15 | 40 | 0.0359 | 0.0014 | 0.0001 |
| 0.30 | 40 | 0.0403 | 0.0091 | 0.0015 |
| 0.45 | 40 | 0.0446 | 0.0169 | 0.0030 |
| 0.30 | 80 | 0.0120 | 0.0125 | 0.0030 |

The parametric bootstrap dominates. The analytic correction is first-order and
slightly overcorrects by n = 80, so `BOOTSTRAP` is the one to use.

The second bias is measurement error, and it is the larger one. Modified Dietz
returns are noisiest while capital is still being called and the denominator is
dominated by the call itself. Error in a regressor attenuates its coefficient,
and simulating from the fitted model reproduces the fitted persistence rather
than the true one, so no small-sample correction recovers it. On the end-to-end
synthetic panel with a true theta of 0.30:

    raw estimate                    0.1606
    demeaning bias removed          0.1958   (+0.0352)
    truth                           0.3000   (+0.1042 measurement error)
    volatility uplift 1/(1-theta)   1.191 -> 1.244 -> 1.429

The measurement-error component is three times the small-sample one.
`test_the_bias_correction_closes_part_of_the_theta_gap_and_names_the_rest` pins
both, and the ordering, so neither can be quietly forgotten.

### Reported upstream, not worked around

`qis.compute_ar_residuals` raises `KeyError: 0` on pandas 3.0: line 9 reads
`ar_model.params[0]`, but `params` is a labelled Series (`const`, `x.L1`), so
that is a label lookup rather than a positional one. It takes
`qis.bootstrap_ar_process` down with it. The fix belongs in the `qis` repository,
and shipped there in `5.2.1` along with three silent defects found alongside it.
This package uses a parametric simulation instead, which is the right method for
a bias correction regardless.

## 0.5.0 (2026-07-26)

The estimator runs end to end in one call. `estimate_matf_alpha` composes
reported returns, the smoothing coefficient, the loadings, the point-in-time
covariance, the deflator and the aggregate, and returns every intermediate
rather than printing it.

### Added

- `estimate_matf_alpha` and `MatfResult` — the whole chain, from marks and cash
  flows to a capital-weighted alpha, as one frozen result.
- `beta=` — price a panel against supplied loadings instead of fitting them. This
  is how a production loading vector is applied to a new panel, and how the
  deflator path is tested without the in-sample fit moving under it.
- `theta=` — fix the smoothing coefficient instead of estimating it, recorded in
  `provenance` because it makes the result depend on data not supplied.
- `MatfResult.provenance` — `qis`, `factorlasso` and `privateassets` versions,
  the seed, the specification, and the two standing caveats
  (`betas_are_in_sample`, `covariance_is_point_in_time`).
- `make_factor_driven_panel` — a synthetic panel whose funds earn exactly what
  the MATF benchmark portfolio earns times `exp(alpha)`, then smoothed by a known
  AR(1). Recovering alpha requires every stage to be right at once.
- 24 tests, including recovery of an injected alpha, a one-for-one response to
  changing it, and the end-to-end no-look-ahead check.

### Fixed

- **The covariance inside the deflator is point-in-time.** The precursor
  estimated one full-sample matrix and applied it to every cash flow, so a 2001
  contribution was deflated with a covariance that knew about 2008 and 2020.
  `test_no_look_ahead_end_to_end` deletes every observation after a vintage
  closes and asserts its alpha is unchanged to 1e-12; it fails under the old
  behaviour.
- **Nothing is printed and no file is written.** The precursor's `run()` emitted
  about forty lines to stdout, wrote a spreadsheet and a PDF, and returned none
  of its inference. Bootstrap intervals and the loading table were unreachable
  programmatically.
- **A vintage that cannot be priced keeps its row with a NaN alpha** rather than
  disappearing, so the aggregate's denominator and the vintage count agree.

### Known and documented

- The loadings are in-sample by construction: one beta over the whole panel,
  applied to every vintage. That is the identification design, and it is why
  `betas_are_in_sample` is in `provenance`.
- The estimated smoothing coefficient is attenuated. Modified Dietz returns are
  noisiest while capital is still being called, and that measurement noise
  attenuates an autoregressive estimate on top of the Kendall demeaning bias.
  `test_theta_is_attenuated_by_the_j_curve` pins the direction. Understating
  theta understates the volatility uplift `1 / (1 - theta)`.

## 0.4.0 (2026-07-26)

Closes the gap between raw fund reporting and the estimator: a fund publishes
marks and cash flows, not returns. The reconstruction assumes **one panel, one
reporting frequency**, and says so out loud rather than smoothing over a panel
that violates it.

### Added

- `nav_implied_returns` — modified Dietz returns per vintage on any supported
  reporting frequency, with the denominator returned alongside as the pooling
  weight.
- `pool_vintage_returns` — capital-weighted aggregation into one manager series.
- `infer_reporting_frequency` — median months between marks, per vintage. Read
  it before choosing the frequency to estimate on.
- `split_by_reporting_frequency` — partition a mixed panel into groups that each
  report at one frequency, so each is estimated on its own grid.

### Fixed

- **A return spanning a skipped period is rejected, not relabelled.** The
  precursor computed the return across a gap and filed it at the panel
  frequency, mixing one-period and two-period returns under one label and
  corrupting every annualisation and autocorrelation built on the series. Both
  ends of a period must now be marked. A panel with gaps raises, naming the
  vintages and how many periods each skips.
- **Nothing is forward-filled.** Carrying a mark forward produces a period of
  exactly zero return, manufacturing the smoothness the unsmoothing step then
  removes, which inflates the AR coefficient and the volatility uplift
  `1 / (1 - theta)`. There is no option to re-enable it.
- **The return index is a complete period range**, so a gap is visible rather
  than absent from the index.

### Changed

- **One covariance path.** `rolling_ewma_quarterly_covar` (36-month span,
  full-window mean, hand-rolled) and `build_rolling_sigma` (60-month span,
  through `qis`) disagreed numerically, and 0.0.1's config advertised the second
  while running neither. Both are replaced by `rolling_factor_covar`, which
  delegates to `qis.estimate_rolling_ewma_covar` at the documented 60-month
  production span. `DEFAULT_COVAR_SPAN_MONTHS` is now 60.
- **No interpolation anywhere in the package.** A series reported less often
  than the estimation grid is estimated on its own grid, or split out. Inventing
  an intermediate path is an assumption about unobserved data, and the package
  does not make it on the caller's behalf.

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
