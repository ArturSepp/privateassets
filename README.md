# privateassets

Quantitative analytics for **private-asset returns** — private equity, private
credit, and other private-market strategies.

Core method: **MATF** — a multi-factor, money-weighted PME that estimates
risk-adjusted alpha and systematic factor exposures (β) directly from private-asset
cash flows, generalising Direct Alpha / KS-PME / GPME from a single benchmark to a
multi-factor deflator. Additional analytics (unsmoothing, volatility/beta
estimation) currently live within `matf` and may be promoted to top-level
submodules as they generalise.

Built on [`qis`](https://pypi.org/project/qis/) and
[`factorlasso`](https://pypi.org/project/factorlasso/); it does **not** depend on
`optimalportfolios` (sibling, not child).

## Install

```bash
pip install privateassets        # once published
# or, from source:
pip install -e .
```

```python
import privateassets
from privateassets.matf import ...   # estimator API
```

## Layout

```
privateassets/                 # MATF estimator engine (deflator, PME, panel-MLE
│   ├── __init__.py            #   unsmoothing, rolling covar, shrinkage, pipeline)
│   └── matf/
│       └── illustrations/     # 14 figure scripts (python -m privateassets.matf <id>)
scripts/                       # reproduction runners
paper_code/                    # publication tracks: jfqa_matf_pme, faj_application (unwritten)
data/                          # git-ignored, private: licensed inputs (see DATA_README.md)
outputs/                       # git-ignored: generated artifacts
projects/                      # git-ignored, private: mandate engagements (not part of the package)
```

## Papers

`paper_code/` holds the clean, public publication tracks. The JFQA paper
(estimator + theory, MSCI cohort data) and the FAJ paper (fund-level application,
Preqin data) are not yet written. The precursor Oaktree manuscript runs on
proprietary LP data and is kept in the private `projects/` tree, out of this
public package.

## Data

No data is stored in this repository. All vendor / LP data is obtained under
licence and placed in `data/` (git-ignored). See `DATA_README.md` for the
per-source licensing rules before adding or committing anything.
