# Data — licensed, never committed

**No data files are stored in this repository, and none may be added.** Every
input is obtained under licence and read from a path you supply. `data/` is
git-ignored in full.

The rule is that code is public and data is not. Before committing anything,
confirm it contains no vendor or LP records, and no intermediate file from which
they could be reconstructed. That includes calibrated constants: a coefficient
estimated on licensed data is a result, not a parameter, and does not belong in
a default argument.

| Source | Granularity | Publication status |
|---|---|---|
| Commercial cohort cash-flow datasets | Strategy x vintage cohorts | Aggregated derived results publishable under the vendor's attribution and pre-publication review terms |
| Commercial fund-level cash-flow datasets | Named fund level | Per-licence; confirm publication terms before use |
| Direct LP cash flows received under NDA | Fund level | Internal only, never publishable |
| Proprietary factor levels and rate series | — | Internal only, never publishable |

Vendor contract terms, pricing, and the status of any commercial negotiation are
not recorded in this repository.

## Input shapes

The readers in `privateassets.matf` take sheet and column names as arguments and
raise `ValueError` naming any column they cannot find. Adapt them to your own
schema rather than renaming your files.

- **Cash flows** — one row per dated flow, carrying a fund identifier, a date, and
  a signed amount. Contributions are negative and distributions positive.
- **NAVs** — one row per reported NAV, carrying a fund identifier, a date, and the
  NAV.
- **Factor levels** — a daily panel of tradable factor index levels. Excess of the
  risk-free rate, because the deflator adds the risk-free leg separately.
- **Risk-free rate** — a quarterly simple yield on a quarter-end grid.
