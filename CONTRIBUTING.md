# Contributing to privateassets

Thanks for your interest in `privateassets`. The package estimates risk-adjusted alpha and factor
exposures from private-asset cash flows, so a change here can move a number in a published table.
Read the licence note below before you start: this repository is MIT, and its `factors` extra
pulls in GPL-3 code.

## Scope

In scope:

- Bug fixes in the PME estimators, the panel AR(1) fit, the bias corrections or the resampling
  inference
- New estimators with a reference for the definition used, and a test pinning them to a published
  or derived value
- Documentation, worked examples and tests

Out of scope — these will be declined, so please open an issue to discuss before writing code:

- New hard runtime dependencies. Optional functionality belongs behind an extra in
  `[project.optional-dependencies]` with a lazy, guarded import
- Performance statistics, factsheets and plotting, which belong in
  [`qis`](https://github.com/ArturSepp/QuantInvestStrats)
- Portfolio optimisation, which belongs in
  [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios)
- Factor model estimation, which belongs in
  [`factorlasso`](https://github.com/ArturSepp/factorlasso)
- Data vendor integrations, and examples needing a paid data subscription to run

## The licence boundary

`privateassets` is MIT. `factorlasso`, behind the `factors` extra, is GPL-3. The core install must
therefore stay importable and useful without it: no module may import `factorlasso` at import
time, only inside the function that needs it.
`test_factorlasso_is_imported_lazily_everywhere` enforces this, and CI runs both installs so
neither can quietly become the other. A change that moves a `factorlasso` import to module scope
makes an MIT install carry GPL-3 obligations, and will be declined.

## Reporting a bug

Open an issue using the bug report template. A report needs the `privateassets` version, your
Python version, whether the `factors` extra is installed, a minimal self-contained reproducer, and
the full traceback or the incorrect numbers. Cash-flow data is usually confidential — please use
`privateassets.tests.synthetic_data`, which generates panels with the defects real ones carry.

## Asking a question

Open an issue and describe what you are trying to do. Questions about methodology are welcome;
where a question is really about the working paper, please say which section you are reading.

## Development setup

```bash
git clone https://github.com/ArturSepp/privateassets.git
cd privateassets
pip install -e ".[dev]"              # core install: three test modules importorskip
pip install -e ".[dev,factors]"      # the full suite, including the loading fit
pytest                               # tests live inside the package
ruff check privateassets
```

Both installs must be green before a pull request. The core one is not optional: it is the install
that proves the MIT claim.

`AGENTS.md` in this repository documents the layout, commands, conventions and constraints in more
detail — it is written for AI coding agents but is equally useful to human contributors.

## Pull requests

- One topic per pull request. Unrelated changes in the same PR make review slower and are likely
  to be asked to split.
- Add or update tests for behaviour you change. A bug fix should come with a test that fails
  before the fix.
- **State whether a number moves.** If a change alters an estimate, a standard error or a
  bootstrap interval, say so in the PR description and show the before and after. A silent
  numerical change is the one thing review here cannot catch by reading.
- Run both installs and `ruff` before submitting.
- Do not bump the version in `pyproject.toml` or `CITATION.cff`; releases are cut separately.
- Do not commit generated output: figures, tables, estimation results or data files.

## Conduct

Be civil and assume good faith. Technical disagreement is welcome; personal remarks are not.

## Licence

This project is MIT licensed. By contributing, you agree that your contributions are licensed
under the MIT licence of this project.
