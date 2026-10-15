# Papers and publication tracks

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Project: [PrivateAssets](https://github.com/ArturSepp/privateassets).
Software citation: [CITATION.cff](https://github.com/ArturSepp/privateassets/blob/main/CITATION.cff).

All paper workspaces are local and Git-ignored, including summaries, manuscripts,
figures, presentations, correspondence, replication code, tests and data. Only
this index and the shared [paper contract](AGENTS.md) are tracked.

## Workspaces

| Track | Working title | Scope | Current status |
|---|---|---|---|
| `multi_factor_direct_alpha` | A Multi-Factor Money-Weighted Alpha for Private Capital | The methods paper. The multi-factor money-weighted deflator, its theory and inference, a Monte Carlo study, and cohort-level empirics. Named object: the MATF deflator (`privateassets.matf.matf_deflator`). | Workspace created 2026-10-13, to draft |
| `jfqa_matf_pme` | — | Earlier workspace for the same methods paper | Superseded by `multi_factor_direct_alpha`, to be retired once its records move |
| `faj_application` | — | The application paper. Manager skill dispersion, manager and vintage selection, and the allocation consequences, on fund-level cash flows. Cites the estimator rather than restating it. | To draft, gated on fund-level data being licensable for central use and for publication |
| `matf_alpha` | MATF-alpha | The single-manager precursor manuscript and its review rounds | Internal, migrated 2026-09-27, no venue |

Each workspace follows [AGENTS.md](AGENTS.md): `paper`, `drafts`, `presentations`,
`private`, `replication` and `agents`. Every section and all future paper folders
remain ignored until an explicit publication decision changes the repository
policy. Local paper folders are retained when their files are removed from tracking.

## Pipeline

Research proposed and recorded in the private working notes, with preliminary
investigations where noted and final publication studies still pending.
Each entry becomes a workspace only when it has a named object, a stated
substitution for the result it displaces, and a canonicalization vehicle in the
package. Data sources are named by class, per the [data policy](../DATA_README.md).

| # | Proposal | Home | Data | Gate |
|---|---|---|---|---|
| P1 | Omitted-factor bias of single-benchmark PME, with the smoothing-robustness and cohort-aggregation results, estimation and inference, Monte Carlo, and the bias quantified across strategies | `multi_factor_direct_alpha` | commercial cohort cash flows, strategy x vintage | Proofs. Prior-art sweep for the three results (done only for the smoothing-coefficient bias). Choice of the lead exhibit. |
| P2 | Skill dispersion across managers, alpha of a fund as cohort baseline plus idiosyncratic skill, manager and vintage selection, allocation consequences | `faj_application` | fund-level cash flows under licence | Licence terms for central use and publication. Cash-flow form, dated transactions or reconstructed. Cohort-construction specification. |
| P3 | Alpha and loadings by vintage year on the cohort universe, separating skill erosion from loading drift and benchmark effects. Candidate headline exhibit of P1. | `multi_factor_direct_alpha` | commercial cohort cash flows | Preliminary rolling and cohort-vintage investigations exist privately. The final vintage exhibit under the settled NAV loading rule remains to be prepared. |
| P4 | Comparing NAV-derived and cash-flow-derived loadings for private-asset capital market assumptions. Practitioner paper proposal. | none yet | commercial cohort cash flows | Production-chain reconciliation and identification assessment. A general replacement of NAV-derived loadings is not an established deliverable. |
| P5 | Estimator comparison on an anonymised panel. The smoothing-coefficient bias decomposition, shrinkage against least squares, and a four-way deflator comparison on identical cash flows. | none yet | calibrated synthetic panel, real-panel aggregates as a check | Release format and licence gates, recorded in the private anonymisation specification. |
| P6 | Cash-flow modelling. Drawdown and distribution paths, commitment pacing, vintage aggregation, and liquidity and undrawn-commitment constraints inside the allocation problem, with a closed-form liquidity cap as the bound. | none yet | commercial cohort cash flows for calibration | Nothing built. No stack implementation of pacing, J-curve or liquidity constraints. |
| P7 | Extension of the single-manager precursor to other strategies and a cross-manager comparison on one factor stack | folds into `faj_application` | fund-level cash flows under licence | Same gate as P2. |
| P8 | Liquid replication of private-asset factor exposure with listed instruments, as a substitute for market exposure rather than for the illiquidity premium | none yet | listed factor and index data | Proof of concept not started. |

The [repository data policy](../DATA_README.md) remains binding. Paper folders
are absent from fresh Git checkouts and Python distributions. Keep private backups
of local material; untracking a file does not remove its earlier Git history.
