"""
privateassets.matf — multi-factor, money-weighted PME.

Estimates risk-adjusted alpha and systematic factor exposures (beta) directly
from private-asset cash flows. Generalises Direct Alpha, KS-PME and GPME from a
single benchmark to a tradable multi-factor deflator, with sign-coherent
shrinkage betas (factorlasso / HCGL), panel-MLE AR(1) unsmoothing, and
block-bootstrap inference.

Modules
-------
  _pme.py            KS-PME, Direct Alpha, Long-Nickels, and XIRR
  _pipeline.py       core estimator: shrinkage betas, deflator, Direct Alpha ladder
  _deflator.py       rolling-Sigma multi-factor deflator
  _unsmooth.py       fixed-theta AR(1) Getmansky-Lo-Makarov unsmoothing
  _panel_mle.py      panel MLE for the AR(1) unsmoothing coefficient
  _rolling_covar.py  qis-canonical rolling EWMA factor covariance

Data and outputs
----------------
Run from the repository root. Inputs are read from data/ and results written to
outputs/, both resolved against the current working directory (override with
PRIVATEASSETS_ROOT). Input data is licensed and is never shipped with the
package. See DATA_README.md.

Built on qis and factorlasso. No dependency on optimalportfolios.
"""
from pathlib import Path
import os

# Repo root: where data/ and outputs/ live. Default is the current working
# directory (run the CLI from the repo root); override with PRIVATEASSETS_ROOT.
# Never resolved relative to the installed package, so a pip-installed copy
# does not read from or write into site-packages.
PROJECT_ROOT = Path(os.environ.get("PRIVATEASSETS_ROOT", os.getcwd())).resolve()
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

# Ensure outputs dir exists at import time
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Convenience handles to the three data files
DATA_XLSX = DATA_DIR / 'data.xlsx'
FACTORS_CSV = DATA_DIR / 'factors.csv'
RF_CSV = DATA_DIR / 'rf_rate.csv'

__version__ = "0.0.1"
