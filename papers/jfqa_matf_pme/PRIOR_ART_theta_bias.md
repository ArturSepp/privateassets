# Prior-art sweep: bias-corrected AR(1) estimation for appraisal-smoothed returns

Recorded 2026-07-27. Gates the novelty claim in the estimation section of the
methods paper, under the rule that a "to our knowledge, first" claim requires the
supporting search to have been performed and written down.

## The claim under test

Two candidate claims sit in the `fit_panel_ar1` work:

- **C1.** A bias-corrected estimator of a common AR(1) coefficient across a panel
  of appraisal-priced funds, by parametric bootstrap, handling heterogeneous
  series lengths and variances.
- **C2.** The decomposition of the fitted coefficient into a small-sample
  demeaning component and a measurement-error component, with the second
  measured and shown to be the larger.

## The five nearest results

**1. Marriott and Pope (1954), and Kendall (1954).** The analytic finite-sample
bias of the serial correlation coefficient under an estimated mean, of order
`-(1 + 3 theta) / n`. This is exactly what `BiasCorrection.KENDALL` implements,
including the constant. *Difference from C1: none.* The formula is theirs, is
seventy years old, and the package cites it rather than claiming it.

**2. Nickell (1981).** The panel form of the same problem: the within
transformation induces an `O(1/T)` bias in the coefficient on the lagged
dependent variable in a fixed-effects dynamic panel. Demeaning each fund's series
before pooling *is* the within transformation, so the bias measured on the
synthetic panels is the Nickell bias, not a new phenomenon. *Difference from C1:
none in kind.* The panel setting was solved before the private-asset application
existed.

**3. Everaert and Pozzi (2007).** An iterative bootstrap bias correction for the
fixed-effects estimator in dynamic panels. This is, in method, what
`BiasCorrection.BOOTSTRAP` does: simulate from the fitted model, measure the
resulting bias, and invert it. **De Vos, Everaert and Ruyssen (2015)** ship it as
the Stata command `xtbcfe`, simplify the algorithm through an invariance
principle, and extend it to unbalanced panels with resampling schemes that
accommodate heteroskedasticity and cross-sectional dependence. *Difference from
C1: the extension I expected to be the novel part — unbalanced panels with
heterogeneous variances — is the stated contribution of the 2015 paper.*

**C1 does not survive.** The estimator is a re-implementation, in Python, of an
established econometric correction, applied to a new class of series. That is
worth a citation and a sentence, not a contribution claim. The package should
cite Everaert and Pozzi for the method and Marriott and Pope for the analytic
alternative, and describe `fit_panel_ar1` as applying them.

**4. Staudenmayer and Buonaccorsi (2005).** Measurement error in linear
autoregressive models: the coefficient is attenuated when the series is observed
with error, and the paper gives the correction. *Difference from C2: the
attenuation mechanism is established.* What the paper does not supply is the
error variance for a Modified Dietz return, which is what makes the correction
operational here and is where the private-asset specificity lives.

**5. Couts, Goncalves and Rossi (2024).** The closest paper in the application
domain. Funds holding illiquid assets report spuriously autocorrelated returns;
existing unsmoothing methods miss that funds holding *similar* assets share a
common source of that autocorrelation, and correcting for it raises measured
systematic risk. They generalise the Getmansky-Lo-Makarov technique and apply it
to hedge funds and commercial real estate. *Difference from C2: they estimate a
richer smoothing structure and do not, in the 2019 working-paper version,
treat finite-sample bias in the smoothing coefficient or errors-in-variables
attenuation of it.* Their common-component insight and the panel-common-theta
assumption here are close relatives, and the paper is the incumbent this work
must position against rather than merely cite.

## Verdict

- **C1 is dead.** Do not claim it. Cite Everaert and Pozzi (2007) for the
  bootstrap correction, De Vos et al. (2015) for the unbalanced and
  heteroskedastic extensions, Marriott and Pope (1954) and Nickell (1981) for the
  analytic bias.
- **C2 is narrower than it looked, and survives in that narrower form.** Not the
  existence of either bias, both of which are established, but the *measured
  decomposition on private-asset cash flows*: on the synthetic panel at a true
  theta of 0.30, the fitted 0.1606 splits into 0.0352 of demeaning bias and
  0.1042 of measurement error, so the uncorrectable component is three times the
  correctable one. The mechanism — Modified Dietz noise is largest while capital
  is still being called, because the denominator is dominated by the call — is
  specific to fund cash flows and is not in any of the five. Frame it as a
  measurement result about private-asset return series, not as an estimator.
- **Consequence for the paper.** The estimation section shrinks from a claimed
  contribution to a correctly-cited application, and the decomposition moves from
  a footnote to the result it supports. That is a smaller claim and a defensible
  one.

## What was and was not checked

Bibliographic details below were verified against publisher or repository
records. **Full texts of Everaert and Pozzi (2007) and of the published Couts,
Goncalves and Rossi (2024) are paywalled and were not read.** For Everaert and
Pozzi the method description comes from the De Vos et al. (2015) account of it;
for Couts et al. the statement about finite-sample bias comes from the 2019
working-paper version, and the published version may differ. Both must be read in
full before the estimation section is written, and this file updated. Treat the
two `[UNVERIFIED-FULLTEXT]` marks below as blocking for submission.

## References

- Kendall, M. G. (1954). Note on bias in the estimation of autocorrelation.
  *Biometrika* 41(3-4), 403-404.
- Marriott, F. H. C. and Pope, J. A. (1954). Bias in the estimation of
  autocorrelations. *Biometrika* 41(3-4), 390-402.
- Nickell, S. (1981). Biases in dynamic models with fixed effects.
  *Econometrica* 49(6), 1417-1426.
- Everaert, G. and Pozzi, L. (2007). Bootstrap-based bias correction for dynamic
  panels. *Journal of Economic Dynamics and Control* 31(4), 1160-1184.
  `[UNVERIFIED-FULLTEXT]`
- De Vos, I., Everaert, G. and Ruyssen, I. (2015). Bootstrap-based bias
  correction and inference for dynamic panels with fixed effects.
  *Stata Journal* 15(4), 986-1018.
- Staudenmayer, J. and Buonaccorsi, J. P. (2005). Measurement error in linear
  autoregressive models. *Journal of the American Statistical Association*
  100(471), 841-852.
- Getmansky, M., Lo, A. W. and Makarov, I. (2004). An econometric model of serial
  correlation and illiquidity in hedge fund returns. *Journal of Financial
  Economics* 74(3), 529-609.
- Couts, S. J., Goncalves, A. S. and Rossi, A. (2024). Unsmoothing returns of
  illiquid funds. *Review of Financial Studies* 37(7), 2110-. `[UNVERIFIED-FULLTEXT]`
