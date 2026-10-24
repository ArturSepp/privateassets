# Contributing to privateassets

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-07-28](https://github.com/ArturSepp/privateassets/commit/44d705027e530c3ec04b1207b28aecc3b757faed)*

Implemented in [privateassets](https://github.com/ArturSepp/privateassets).
Software citation: [CITATION.cff](https://github.com/ArturSepp/privateassets/blob/main/CITATION.cff).

Thanks for your interest in `privateassets`. The package estimates risk-adjusted alpha and factor
exposures from private-asset cash flows, so a change here can move a number in a published table.
Read the licence note below before you start: this repository is MIT, and the development
branch's `factors` extra requires the verified MIT FactorLasso release or later.

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

`privateassets` is MIT. The development branch's `factors` extra requires
`factorlasso>=1.0.4.post1`, the first verified MIT FactorLasso release. Preserve
dependency notices; historical GPL releases retain their licences.
The core install must stay importable and useful without optional estimators:
no module may import `factorlasso` at import time, only inside the function that needs it.
`test_factorlasso_is_imported_lazily_everywhere` enforces this, and CI runs both installs so
neither can quietly become the other. A change that moves a `factorlasso` import to module scope
breaks the core-install contract and will be declined.

## Reporting a bug

Open an issue using the bug report template. A report needs the `privateassets` version, your
Python version, whether the `factors` extra is installed, a minimal self-contained reproducer, and
the full traceback or the incorrect numbers. Cash-flow data is usually confidential — please use
`tests.synthetic_data`, which generates panels with the defects real ones carry.

## Asking a question

Open an issue and describe what you are trying to do. Questions about methodology are welcome;
where a question is really about the working paper, please say which section you are reading.

## Starter tasks

These are bounded proposals, not assigned issues or confirmed defects. Recheck source and open
work before agreeing one task. Use the existing [feature form](https://github.com/ArturSepp/PrivateAssets/issues/new?template=feature_request.yml),
Area **Documentation or example**. Include the task ID, owner, exact paths, reproduction, acceptance and focused
check under Proposed behavior; retain required boundary checkboxes. Use the bug form only for
observed incorrect behaviour. Keep numerical code, public signatures, dependencies, fixtures and
canonical examples unchanged for these documentation tasks.

### PA-1: clarify PME inputs

Owner: PrivateAssets. Edit only `docs/quickstart.rst` to clarify one observed cash-flow-sign or terminal-NAV ambiguity. Acceptance: distinguish contributed capital, distributions, residual NAV and dimensionless PME without changing the example or estimator.

```console
python examples/first_success.py
python -m pytest tests/test_pme.py -q
```

### PA-2: explain optional factor setup

Owner: PrivateAssets. Edit only `docs/install.rst` to clarify one core-versus-factors setup ambiguity. Acceptance: a reader can run core PME without FactorLasso and identify when the optional extra is needed; do not add an import or dependency.

```console
python -m pytest tests/test_public_api.py -k factorlasso -q
```

### PA-3: check release navigation

Owner: PrivateAssets. Edit only `docs/quickstart.rst` to clarify one released-versus-main boundary. Acceptance: retain the pinned canonical example, old page anchor and private-data boundary, with no changes to computed results.

```console
python -m pytest tests/test_docs_site.py -q
```

Run the package's existing strict HTML command after a documentation edit and inspect the
changed passage. On the maintainer's OneDrive checkout, dot-source the shared
`Enter-AgentRepo.ps1` first and use `C:\Python\PrivateAssets312\Scripts\python.exe`.
Keep exports, caches, builds and run output on C. The `uv` setup below is for ordinary contributor
checkouts outside OneDrive; never create `.venv` in the synchronized checkout.

Follow the [shared review guidance](https://github.com/ArturSepp/ArturSepp/blob/main/docs/contributing_to_stack.md#review-attribution-and-automation-assisted-changes).
Record the revision, actual versions, exact checks and skips, rendered review and material
automation assistance. Preserve MIT core and the optional FactorLasso boundary licensing and attribution. Automated checks do not
establish independent reader success. Review time and acceptance are not promised.

## Development setup

```bash
git clone https://github.com/ArturSepp/privateassets.git
cd privateassets
uv sync --locked --group test
uv run --no-sync pytest                          # core: optional factor tests skip
uv sync --locked --group test --extra factors
uv run --no-sync pytest                          # full: includes the loading fit
uv run --locked --only-group lint ruff check src/privateassets tests examples docs/conf.py
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
