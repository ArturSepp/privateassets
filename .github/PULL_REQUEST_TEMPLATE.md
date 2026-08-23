## What changed

Describe the problem and the smallest coherent change that solves it.

## Verification

List the exact commands run and their results.

## Checklist

- [ ] Tests cover the changed public behavior or defect.
- [ ] Return frequency, annualisation, day count, and excess-versus-total conventions remain explicit.
- [ ] No private or licensed data, local paths, generated outputs, or agent reports are included.
- [ ] Core tests pass without factorlasso; full tests pass with the factors extra.
- [ ] uv run --locked --only-group lint ruff check src/privateassets tests examples docs/conf.py passes.
- [ ] User-visible changes are documented in CHANGELOG.md and relevant docs.
- [ ] New runtime dependencies or public-signature changes are called out explicitly.
