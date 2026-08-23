"""Exercise a minimal offline PME calculation using only core dependencies."""

# packages
import numpy as np
import pandas as pd
# qis / project
import privateassets
from privateassets.matf import ks_pme


def main() -> None:
    """Run one finite core calculation without files, network, or optional extras."""
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
    main()
