"""Exercise a minimal offline PME calculation using only core dependencies."""

# packages
from enum import Enum

import numpy as np
import pandas as pd
# qis / project
import privateassets
from privateassets.matf import ks_pme


class Locals(Enum):
    """available local example workflows."""

    CORE_PME = 1


def run_local(local: Locals) -> None:
    """run one finite core calculation without files, network, or optional extras."""
    if local != Locals.CORE_PME:
        raise ValueError(f"unsupported local case: {local}")

    dates = pd.Series(pd.to_datetime(['2020-01-01', '2021-01-01']))
    amounts = pd.Series([-100.0, 110.0])
    benchmark = pd.Series([100.0, 110.0], index=pd.DatetimeIndex(dates))

    pme = ks_pme(
        cf_dates=dates,
        cf_amounts=amounts,
        rvpi_nav=0.0,
        rvpi_date=dates.iloc[-1],
        bench_idx=benchmark,
    )
    if not np.isfinite(pme) or pme <= 0.0:
        raise RuntimeError(f'core PME calculation returned {pme!r}')
    print(f'privateassets {privateassets.__version__}: core PME calculation succeeded')


if __name__ == '__main__':
    run_local(local=Locals.CORE_PME)
