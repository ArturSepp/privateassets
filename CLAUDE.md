# privateassets — repository map and conventions

`privateassets` is a public, installable library for private-asset return
analytics. It is a sibling of `optimalportfolios`, not a child: it depends only
on `qis` + `factorlasso` (+ numpy/pandas/scipy/cvxpy/openpyxl) and imports
nothing from `optimalportfolios`.

## Layout

The package sits flat at the repository root (same pattern as
`optimalportfolios`), not under `src/`.

```
privateassets/          PUBLIC package (tracked). The MATF engine + illustrations.
scripts/                PUBLIC reproduction runners.
paper_code/             PUBLIC. Clean publication tracks (JFQA, FAJ). Unwritten.
projects/               PRIVATE, untracked (git-ignored). Mandate engagements.
data/                   PRIVATE, untracked (git-ignored). Licensed / LP data.
outputs/                Generated artifacts (git-ignored).
```

`projects/` and `data/` are developed alongside the package for convenience and
are excluded by `.gitignore`. They are not part of the published package, the
same way `rosaa` nests inside the `optimalportfolios` working directory without
being tracked there.

## Rules

- Code is public, data is not. No vendor or LP data, and no file from which it
  could be reconstructed, may enter `privateassets/`, `scripts/`, or
  `paper_code/`. See `DATA_README.md`.
- The precursor Oaktree manuscript runs on proprietary data and lives only in
  `projects/`. The publication-track manuscripts in `paper_code/` are the clean,
  releasable artifacts.
- Run the CLI from the repository root. Path resolution is cwd-based, overridable
  with the `PRIVATEASSETS_ROOT` environment variable.
- Mark every unverified figure or number `[TODO]`. No invented citations or
  results.
