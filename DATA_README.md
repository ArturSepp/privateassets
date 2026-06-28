# Data — licensed, not committed

**No data files are stored in this repository.** Every input is obtained under
licence and placed in `data/` locally; `data/` is git-ignored in full. The public
package must never contain vendor or LP data, nor any derived file from which it
could be reconstructed.

| Source | Used by | Granularity | Publication status |
|---|---|---|---|
| **MSCI Private Capital — CFO dataset** | JFQA paper, SAA/CMA | Cohort (strategy × vintage), 1990+ | Aggregated derived results publishable **with MSCI attribution + standard pre-publication review** |
| **Preqin (BlackRock) — fund-level cash flows** | FAJ paper, selection | Named fund-level | Publication terms **pending** (Boris Hamm / BlackRock) — confirm before use |
| **Oaktree / Crown LP cash flows** | Precursor draft only | Fund-level | **Internal only — never publishable** |
| MATF factor levels, rate series | all | — | LGT-internal; not for public release |

Rule of thumb: code is public, data is not. Before committing anything, confirm
it contains no vendor records and no intermediate file from which they could be
recovered.
