Conventions
===========

- Factor inputs are **excess** log returns. The risk-free rate enters the
  deflator once, through its own term.
- Covariance is **point-in-time**: the matrix at a quarter end uses only returns
  observed by that date. ``test_rolling_covariance_is_point_in_time`` enforces it.
- Deflator matrices are in **quarterly** units and scale by the horizon in
  quarters.
- Day counts are ACT/365.25 throughout.
- A cash-flow date before the factor panel starts returns NaN. It is not pinned
  to the first quarter end.
- There is one covariance path, ``rolling_factor_covar``, which delegates to
  ``qis.estimate_rolling_ewma_covar`` at a 60-month span. The 36-month
  full-window-mean variant that shipped in 0.1.0 is gone.
- One panel, one reporting frequency. Every return spans exactly one period,
  and nothing is forward-filled or interpolated.
