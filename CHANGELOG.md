# Changelog

## 0.0.1 (2026-06)

Initial release of the private-asset analytics package.

- `matf`: multi-factor, money-weighted PME. Tradable multi-factor deflator,
  KS-PME / Direct Alpha / Long-Nickels, panel-MLE unsmoothing, rolling EWMA
  covariance, sign-coherent shrinkage betas, and the MATF Direct Alpha pipeline.
- 14 illustration scripts under `matf/illustrations/` that regenerate the
  precursor figures via `python -m privateassets.matf <id>`.
- Built on `qis` and `factorlasso`. No dependency on `optimalportfolios`
  (sibling, not child).
- Flat package layout at the repository root, matching the `optimalportfolios`
  house style.
