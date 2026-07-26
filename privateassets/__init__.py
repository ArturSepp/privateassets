"""privateassets — quantitative analytics for private-asset returns.

Core method: MATF, a multi-factor money-weighted PME that estimates risk-adjusted
alpha and factor exposures from private-asset cash flows.

Built on ``qis`` and ``factorlasso``. No dependency on ``optimalportfolios``,
which is a sibling rather than a parent.

    from privateassets.matf import ks_pme, direct_alpha, matf_deflator

Importing this package has no side effects and reads nothing from disk.
"""
__version__ = "0.1.0"

__all__ = ['__version__']
