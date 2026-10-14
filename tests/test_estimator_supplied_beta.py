"""Supplied-beta pricing regressions also run without the optional loading fitter."""

# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
from privateassets.matf import estimate_matf_alpha
from tests.synthetic_data import make_factor_driven_panel


@pytest.mark.parametrize('order', [['Equity', 'Credit'], ['Credit', 'Equity'],
                                 ['Commodities', 'Credit', 'Rates', 'Equity']])
def test_subset_and_reordered_beta_price_like_the_full_vector(order):
    """Fails on the old all-factor covariance paired with a shorter or reordered beta."""
    panel = make_factor_driven_panel()
    beta = pd.Series({'Equity': 0.7, 'Rates': 0.0, 'Credit': 0.5, 'Commodities': 0.0})
    full = estimate_matf_alpha(*panel, beta=beta, theta=0.0, sigma_default=np.zeros((4, 4)))
    selected = estimate_matf_alpha(*panel, beta=beta.loc[order], theta=0.0,
                                   sigma_default=np.zeros((len(order), len(order))))
    np.testing.assert_allclose(selected.vintage_alpha.matf_alpha, full.vintage_alpha.matf_alpha,
                               rtol=0, atol=1e-10)
    assert all(sigma.shape == (len(order), len(order)) for sigma in selected.sigma_by_period.values())


def test_supplied_beta_covariance_shape_error_names_both_shapes():
    """Fails if an invalid seed covariance reaches matrix multiplication or is unused silently."""
    with pytest.raises(ValueError, match=r'shape \(4, 4\).*shape \(2, 2\)'):
        estimate_matf_alpha(*make_factor_driven_panel(), beta=pd.Series({'Credit': .5, 'Equity': .7}),
                            sigma_default=np.zeros((4, 4)))
