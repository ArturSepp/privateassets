# privateassets

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-06-28](https://github.com/ArturSepp/privateassets/commit/37e8fee827a5d5630398cbf20dfdb3b5df876181)*

Implemented in [privateassets](https://github.com/ArturSepp/privateassets).
Software citation: [CITATION.cff](https://github.com/ArturSepp/privateassets/blob/main/CITATION.cff).

**Multi-factor money-weighted PME for private-asset cash flows: risk-adjusted alpha and factor exposures**

You supply fund cash flows and NAVs together with benchmark levels for classical PME, or factor
index levels and the matching risk-free-rate series for MATF estimation. No fund records, market
data, or licensed datasets ship with the package.

**Install:** `pip install privateassets` · **Import:** `privateassets` · **Status:** Alpha

[![PyPI](https://img.shields.io/pypi/v/privateassets?style=flat-square)](https://pypi.org/project/privateassets/)
[![Python](https://img.shields.io/pypi/pyversions/privateassets?style=flat-square)](https://pypi.org/project/privateassets/)
[![CI](https://github.com/ArturSepp/privateassets/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ArturSepp/privateassets/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/privateassets/badge/?version=latest)](https://privateassets.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/ArturSepp/privateassets.svg?style=flat-square)](LICENSE.txt)
[![Downloads](https://static.pepy.tech/badge/privateassets)](https://pepy.tech/project/privateassets)
[![Monthly](https://static.pepy.tech/badge/privateassets/month)](https://pepy.tech/project/privateassets)

**Documentation:** [privateassets.readthedocs.io](https://privateassets.readthedocs.io/en/latest/) ·
[core workflows](https://privateassets.readthedocs.io/en/latest/workflows.html) ·
[offline first success](examples/first_success.py)

---

## Why privateassets

A single-benchmark PME divides fund cash flows by the return of one index. That
charges the fund for one exposure and credits everything else to skill. The MATF
deflator divides them by the return of a *tradable multi-factor portfolio*, so a
distressed-credit fund is measured against the credit and equity basket it
actually loaded on rather than against equities alone.

**privateassets generalises Direct Alpha, KS-PME and GPME from one benchmark to
a multi-factor deflator**, and ships the classical measures alongside so the two
can be compared on the same cash flows.

### Key differentiators

**Fund reporting to alpha in one call.** `estimate_matf_alpha` takes cash flows,
NAVs and factor levels and returns capital-weighted and per-vintage alpha,
factor loadings, bootstrap intervals and full provenance. It returns every
intermediate it computed, and writes nothing. Each stage — NAV-implied returns,
unsmoothing, factor betas, deflators — is also usable on its own.

**Point-in-time by construction.** The covariance inside the deflator uses only
returns observed by each quarter end: deleting every observation after a vintage
closes leaves its alpha unchanged to 1e-12, which
`test_no_look_ahead_end_to_end` asserts.

**One panel, one reporting frequency.** Nothing is forward-filled or
interpolated; a panel whose vintages report at different frequencies raises,
naming the offenders. Interpolating them onto a common grid is an assumption
about an unobserved path, and this package does not make it for you.

**The incumbents ship alongside.** `kn24_benchmark_deflator` and
`kn16_gpme_deflator` price the same cash flows against one market index, so the
multi-factor result is reported next to what it replaces — and passing a single
benchmark's reciprocal index ratio as the deflator reproduces classical Direct
Alpha to root-finder tolerance, pinned by a test.

**Built on the stack, not duplicating it.** Unsmoothing, covariance estimation
and block resampling delegate to [`qis`](https://pypi.org/project/qis/);
sign-constrained shrinkage betas to
[`factorlasso`](https://pypi.org/project/factorlasso/) through the optional
`[factors]` extra.

### When to use it — and when not

Use `privateassets` to estimate risk-adjusted alpha and systematic factor
exposures of private-equity and private-credit funds from their cash flows and
NAVs, to unsmooth appraisal-based NAV returns with bias-corrected AR(1)
estimates, and to report multi-factor and classical PME side by side on the
same panel.

Do not reach for it for portfolio construction — that is its sibling
[`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios). The
reporting and factsheet layer is not in this release, no data ships with the
package, and the factor loadings are in-sample by construction — one beta over
the whole panel — so it is a measurement tool, not a live risk system. The
caveats travel with every number in `provenance`.

## Installation

```bash
pip install privateassets
```

Sign-constrained shrinkage betas need the `factors` extra:

```bash
pip install "privateassets[factors]"
```

Built on [`qis`](https://pypi.org/project/qis/) for unsmoothing, covariance
estimation and resampling, and optionally on
[`factorlasso`](https://pypi.org/project/factorlasso/) for sign-constrained
shrinkage betas. It does not depend on `optimalportfolios`, which is a sibling.

**Licence note.** This package is MIT. The `factors` extra requires
`factorlasso>=1.0.4.post1`, the first verified MIT FactorLasso release; earlier published GPL
FactorLasso releases retain their licences. The core PME and deflator paths remain usable
without FactorLasso. Dependencies retain their own licences and notices, which the
[supplemental third-party notices](THIRD_PARTY_NOTICES.md) retain for the reviewed Windows core
and factors environments; this does not make the entire installed environment MIT.

**Data.** No data ships with this repository, and none may be added. Every input is licensed and
read from a path you supply. Fund-level analysis requires user-supplied cash flows and NAVs;
classical PME also needs benchmark levels, while MATF estimation needs factor levels and a
matching risk-free-rate series. See [`DATA_README.md`](DATA_README.md).

## Table of contents

1. [Why privateassets](#why-privateassets)
2. [Installation](#installation)
3. [Statement of need](#statement-of-need)
4. [Overview](#overview)
5. [Five-minute quickstart](#five-minute-quickstart)
6. [Examples](#examples)
7. [Testing](#testing)
8. [Changelog](#changelog)
9. [Feedback & contributing](#feedback-contributing)
10. [Citation](#citation)
11. [Ecosystem](#ecosystem)
12. [License](#license)
13. [Disclaimer](#disclaimer)

## Statement of need

Private-asset performance is money-weighted and benchmark-relative: a public-market equivalent
(PME) deflates each cash flow by the growth of a reference index and asks what is left. With one
index, Direct Alpha, KS-PME and the generalised PME of Korteweg and Nagel charge the fund for one
exposure and credit everything else to skill. `privateassets` replaces the single index by a
tradable multi-factor deflator whose loadings are estimated from the fund's own NAV-implied,
unsmoothed returns, point-in-time at every quarter end, and reports the classical measures next
to the multi-factor result on the same cash flows. The methods paper is in preparation; its
workspace is local by policy ([paper index](papers/README.md)).

| Paper | Implemented analytics | Where |
|---|---|---|
| [TODO: Sepp et al., *A Multi-Factor Money-Weighted Alpha for Private Capital*, working title; reference once public] | The multi-factor money-weighted deflator, capital-weighted and per-vintage alpha, factor loadings with bootstrap intervals, AR(1) unsmoothing with bias correction, and the single-benchmark incumbents (`kn24_benchmark_deflator`, `kn16_gpme_deflator`) for side-by-side reporting | `privateassets.matf`, [core workflows](https://privateassets.readthedocs.io/en/latest/workflows.html) |

## Overview

The installable package is `src/privateassets/`, with the analytics in the `privateassets.matf`
subpackage: KS-PME, Direct Alpha and vintage statistics, the one-call multi-factor alpha
estimator, NAV-implied returns, pooled-cohort extracts, shrinkage factor loadings, the
multi-factor deflator, the single-benchmark incumbents, and the bootstrap inference. The
[core workflows guide](https://privateassets.readthedocs.io/en/latest/workflows.html) walks
through them in order, the [conventions guide](https://privateassets.readthedocs.io/en/latest/conventions.html)
fixes the return, frequency and sign conventions, and the
[API reference](https://privateassets.readthedocs.io/en/latest/api.html) lists every public
object. Loading fits accept `ExpertPrior` to centre a selected factor on its weighted
univariate regression and impose a sign-oriented uncertainty floor; the returned
`PriorDiagnostics` makes that convention inspectable, and the loading bootstrap re-estimates it
on each draw. This reproduces an expert loading rule; the floor does not establish that the
multifactor exposure is well identified.

### Analytics at a glance

| Area | What the package computes |
|---|---|
| Classical PME | KS-PME (`ks_pme`), Direct Alpha (`vintage_direct_alpha`) and vintage statistics on user-supplied cash flows, NAVs and benchmark levels |
| Multi-factor alpha | `estimate_matf_alpha`: cash flows, NAVs and factor levels → capital-weighted and per-vintage alpha, factor loadings, bootstrap intervals and provenance, every intermediate returned, nothing written |
| NAV-implied returns | Modified Dietz returns from reported NAVs and cash flows, one panel at one reporting frequency; mixed-frequency panels raise and name the offenders |
| Unsmoothing | `fit_panel_ar1` across a panel of funds, with `BiasCorrection.BOOTSTRAP` (parametric simulation from the fitted model) or the analytic `KENDALL` formula; the inversion itself is `qis.unsmooth_returns_glm`; `theta_raw` is always reported |
| Factor loadings | Sign-constrained shrinkage betas through `factorlasso` (`[factors]` extra), `ExpertPrior` centring and `PriorDiagnostics`; point-in-time covariance using only returns observed by each quarter end |
| Deflators | The multi-factor MATF deflator; `kn24_benchmark_deflator` and `kn16_gpme_deflator` (with `kn16_sdf_params`) for the single-index incumbents, taking the equity factor as an excess log return and adding the risk-free leg back where the economics needs a total return |
| Inference | `bootstrap_factor_betas`: block resampling through `qis` with the asset and its factors resampled together; `share_at_zero` reports how often a sign constraint binds (not a p-value) |

Two caveats travel with every number and are recorded in `provenance`: the loadings are
in-sample, and the smoothing coefficient is attenuated by measurement noise in the J-curve period
even after bias correction.

## Five-minute quickstart

The repository's deterministic example uses only core dependencies and synthetic values. From a
source checkout (the `examples/` directory is not included in the wheel), run:

```console
python examples/first_success.py
```

Expected evidence for this release:

```text
privateassets 0.8.0: core PME calculation succeeded
```

See the [docs quickstart](https://privateassets.readthedocs.io/en/latest/quickstart.html) for the complete inputs and call, included from [`examples/first_success.py`](examples/first_success.py).

## Examples

### Highlighted examples

**Unsmoothing.** Appraisal NAVs are reported with a lag and anchored to the previous mark, so
reported returns are a moving average of true ones. Estimate the AR(1) coefficient across a
panel of funds here, then apply it with `qis`:

```python
import qis
from privateassets.matf import fit_panel_ar1

result = fit_panel_ar1(demeaned_series_by_fund)
unsmoothed = qis.unsmooth_returns_glm(returns, ar_order=1, theta=result['theta_hat'])
```

The inversion itself is `qis.unsmooth_returns_glm`. This package does not carry a second copy of
it. **The estimate is biased down by two separate mechanisms, and only one is correctable.**
Demeaning a short AR(1) biases the coefficient by about `-(1 + 3θ)/n`; pass
`bias_correction=BiasCorrection.BOOTSTRAP` to remove it — a parametric simulation from the
fitted model, which cuts mean absolute error by roughly an order of magnitude and handles
heterogeneous series lengths that the analytic `KENDALL` formula only approximates. What remains
is measurement error: Modified Dietz returns are noisiest while capital is still being called,
and error in a regressor attenuates its coefficient. On the end-to-end synthetic panel with a
true θ of 0.30, the raw estimate is 0.161, correcting the demeaning bias gives 0.196, and the
remaining 0.104 is measurement error that no small-sample correction reaches. It is the larger
of the two. Corrections are off by default and `theta_raw` is always reported.

**Comparing against the single-factor incumbents.** `kn24_benchmark_deflator` and
`kn16_gpme_deflator` price the same cash flows against one market index, so the multi-factor
result can be reported next to what it replaces:

```python
from privateassets.matf import kn16_gpme_deflator, kn16_sdf_params

delta, gamma, sigma2 = kn16_sdf_params(equity_excess_log_returns, rf_quarterly)
kn_deflators = kn16_gpme_deflator(cf_dates=dates, t0=dates[0],
                                  cum_log_equity_excess=cum_log_equity,
                                  cum_log_rf=cum_log_rf, quarter_ends=quarter_ends,
                                  delta=delta, gamma=gamma)
kn_alpha = vintage_direct_alpha(cf_v, rvpi_nav, dates, kn_deflators)
```

**Inference.** Loadings are resampled in blocks through `qis`, with the asset and its factors
resampled together:

```python
from privateassets.matf import bootstrap_factor_betas

boot = bootstrap_factor_betas(asset_returns, factor_returns,
                              num_samples=1000, block_size=12, seed=1)
print(boot.lower, boot.upper, boot.qis_version)
```

`share_at_zero` reports how often a sign constraint binds. It is not a p-value: under a binding
constraint the mass sits on the boundary, so the quantity tracks the constraint, not the
evidence.

### Runnable examples

- [`examples/first_success.py`](examples/first_success.py) — the minimal offline KS-PME
  calculation on synthetic values, using only core dependencies; the
  [docs quickstart](https://privateassets.readthedocs.io/en/latest/quickstart.html) includes it.

The [core workflows guide](https://privateassets.readthedocs.io/en/latest/workflows.html) carries
the end-to-end multi-factor alpha workflow from fund reporting to `estimate_matf_alpha`.

## Testing

```bash
uv sync --locked --group test
uv run --no-sync pytest
uv sync --locked --group test --extra factors
uv run --no-sync pytest
uv run --locked --only-group lint ruff check src/privateassets tests examples docs/conf.py
```

The suite uses no network or data files. `tests/synthetic_data.py` draws a seeded panel carrying
the defects real panels carry: irregular cash-flow dates, a J-curve, unrealised residual NAVs, and
a factor panel that starts after the first fund does. Tests needing the `[factors]` extra skip
rather than fail, so a core install stays green. Enforcement tests fail the suite if package
imports have filesystem side effects, documentation drifts from signatures, proprietary
identifiers or competing-stack imports enter the package, release metadata diverges, or
automated tests, development runners, production modules and built distributions cross their
declared boundaries. Development runners (`src/privateassets/run/<subject>_local.py`, excluded
from distributions) are described in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Changelog

Release history is maintained in [CHANGELOG.md](CHANGELOG.md).

## Feedback & contributing <a name="feedback-contributing"></a>

- **Bug:** [report a reproducible bug](https://github.com/ArturSepp/privateassets/issues/new?template=bug_report.yml), using synthetic or anonymised inputs and including the package version, Python/platform, and expected versus actual result.
- **Feature:** [request a feature](https://github.com/ArturSepp/privateassets/issues/new?template=feature_request.yml), explaining which fund-reporting convention or benchmark input is blocking adoption, the current workaround, and the smallest useful API.
- **Question or methodology:** search or open an [issue](https://github.com/ArturSepp/privateassets/issues) and name the measure, convention, or deflator involved.
- **Contribution:** read [CONTRIBUTING.md](CONTRIBUTING.md) for the public-data boundary, development commands, numerical-change rules, and pull-request guidance.

## Citation

Machine-readable metadata is available in [`CITATION.cff`](CITATION.cff). A copyable software
citation is:

```bibtex
@software{sepp2026privateassets,
  author = {Sepp, Artur},
  title = {privateassets: Multi-factor Money-weighted PME for Private-asset Cash Flows},
  year = {2026},
  version = {0.8.0},
  url = {https://github.com/ArturSepp/privateassets}
}
```

[TODO: add the methods paper entry once it is public.]

## Ecosystem

This package is part of an open-source Python stack for quantitative finance. The
[ArturSepp profile](https://github.com/ArturSepp) is the canonical catalogue:

| Package | Purpose |
|---|---|
| [`qis`](https://github.com/ArturSepp/QuantInvestStrats) | Performance analytics, backtesting, and factsheet reporting |
| [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios) | Portfolio construction and rolling backtesting |
| [`factorlasso`](https://github.com/ArturSepp/factorlasso) | Sparse factor-model estimation |
| [`bbg-fetch`](https://github.com/ArturSepp/BloombergFetch) | Bloomberg data in pandas DataFrames |
| [`option-chain-analytics`](https://github.com/ArturSepp/OptionChainAnalytics) | Point-in-time option-chain data and queries |
| [`vanilla-option-pricers`](https://github.com/ArturSepp/VanillaOptionPricers) | Numba-vectorised BSM and Bachelier pricing |
| [`stochvolmodels`](https://github.com/ArturSepp/StochVolModels) | Stochastic-volatility pricing and calibration |
| [`trendfollowing`](https://github.com/ArturSepp/TrendFollowingSystems) | Closed-form trend-following analytics |
| [`privateassets`](https://github.com/ArturSepp/privateassets) *(this package)* | Multi-factor PME for private assets |
| [`goal-based-allocation`](https://github.com/ArturSepp/GoalBasedAllocation) | Goal-based allocation under regime-switching jump-diffusions |

`privateassets` uses `qis` for time-series analytics and resampling. The optional `factors`
extra adds `factorlasso` for sign-constrained shrinkage betas. `optimalportfolios` is a sibling
for portfolio construction, not a dependency.

## License

This project is licensed under the MIT License; see [`LICENSE.txt`](LICENSE.txt).
The optional FactorLasso version floor and dependency notices are described under
[Installation](#installation).

## Disclaimer

`privateassets` is distributed FREE & WITHOUT ANY WARRANTY under the MIT License. See
[LICENSE.txt](LICENSE.txt) in the release for details. Use the routes in
[Feedback & contributing](#feedback-contributing) for bugs, feature requests and
methodology questions.
