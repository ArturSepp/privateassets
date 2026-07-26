# Changelog

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
