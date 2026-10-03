# Paper workspace contract

Root AGENTS.md and DATA_README.md control numerical invariants and the stricter
data policy: code is public; vendor data, LP records and reconstructable inputs
or results are never committed. This layout does not relax that rule.

## Layout

Use `papers/<paper_id>/` outside the installable package.

| Section | Purpose | Git policy |
|---|---|---|
| `paper/` | Current manuscript, PDF and `figures/` | Always ignored under the current policy |
| `drafts/<version>/` | Prior drafts with their own figures and dependencies | Always ignored |
| `presentations/<event>/` | Slides and their figures | Always ignored under the current policy |
| `private/` | Editorial correspondence, referee reports, replies and permissions | Always ignored |
| `replication/` | Local research code, `data/`, automated `tests/` | Entire section ignored |
| `agents/` | Roadmaps, audits, prior-art searches and execution records | Always ignored |

Do not track empty placeholders. Each paper's ignored `agents/` owns paper-specific
working records, overriding the generated shared core's root-only record location.
Repository-wide working records remain in root `agents/`.

## Publication boundaries

- Every current and future `papers/<paper_id>/` workspace is fully ignored,
  including `jfqa_matf_pme`, `faj_application` and `matf_alpha`. This includes
  summaries, nested policy files, code, tests and every other paper asset.
  Only the shared `papers/AGENTS.md` and `papers/README.md` remain tracked.
- Keep the application track's data/publication gate. The layout does not grant
  permission to publish the planned methods paper or any empirical result.
  SSRN availability alone is not a GitHub publication permission.
- Reopening any paper workspace requires an explicit publication decision and
  coordinated updates to the root ignore rules, publication checker and index.
  Nested exceptions or force-adding files cannot override this local-only policy.
- All `replication/data/` content stays ignored, including `data/local/`.
  Keep input documentation in `replication/README.md`; never add data exceptions.
  Licensed data can remain in existing ignored root `data/` or external storage;
  do not duplicate it to populate these folders. Root `projects/` stays private.
- Public tests use the existing synthetic-data generator or independently
  specified synthetic inputs in code, never calibrated constants from licensed
  records. Existing `tests/synthetic_data.py` and numerical goldens stay unchanged.
- A prior-art search is an agent working record. Retain its evidence locally;
  approved manuscript text should cite independently verified primary sources.
- Preserve files when untracking. Ignored files need private backup; removing a
  file from the current index does not erase earlier Git history.

## Replication and validation

Paper-specific code and tests go under `replication/` and `replication/tests/`.
Existing package tests remain in root `tests/`. No paper-specific pipeline exists
yet; do not invent reproduction claims or duplicate package calculations.

Use the root-prescribed external Python environment. Before Python/builds on this
host, configure the shared `Enter-AgentRepo.ps1` runtime with this repository's
`-RepoPath` and the stack `-RepositoriesRoot`. Generate figures, caches and builds
outside the checkout and OneDrive; promote publication assets only after review.
Never create an environment inside OneDrive.

Run `.github/scripts/check_paper_policy.py --worktree` for an unstaged preview and
without options for the actual index and indexed ignore rules. CI repeats the
checker and its regressions. Forced additions of private sections, data files or
agent reports must fail even if nested ignore rules try to reopen them.
The public-prose leak scan follows Git-visible documents, so local ignored records
do not become public test inputs; force-added records remain visible to enforcement.

Both wheels and source archives must exclude papers, agent records, root data,
projects and outputs. Check both with `--artifacts <directory>`. Preserve the
existing core/factors test matrix; layout work must not change the optional extra
contract, numerical code, seeds or baseline values.
