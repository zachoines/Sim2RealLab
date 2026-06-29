# Establish a clean `flake8` / `black` baseline across the Python source

**Type:** task / refactor (tech-debt)
**Owner:** Either — the lint set spans Jetson-owned `strafer_ros`, shared `strafer_shared`, and DGX-owned `strafer_lab/scripts`; see Context for the cross-lane split.
**Priority:** P3
**Estimate:** M — mechanical but wide; the violation count is unknown and expected large (`make lint` / `make format-check` have never been run to green).
**Branch:** task/python-lint-format-baseline

## Story

As a **maintainer of this repo**, I want **`make lint` and `make format-check` to
pass cleanly on the tracked Python source**, so that **a lint/format gate can be
added to CI and future PRs stop accreting style drift** — instead of the current
state where neither target has ever been run green and running them now surfaces
a large backlog.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md) — the
  reformat crosses three lanes; respect the boundaries (see Context).
- [context/conventions.md](../../context/conventions.md) — the source-comment
  norms (no transient refs, no env names) this pass must not violate.
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

The `Makefile` defines but does not currently pass:
- `make lint` → `flake8 source/strafer_ros/ source/strafer_shared/ source/strafer_lab/scripts/ --max-line-length 100 --extend-ignore=E203,W503`
- `make format-check` → `black --check` over the same three trees (black reads
  `pyproject.toml` → `[tool.black] line-length = 120`).

Neither has been run to green; running them now is expected to surface many
violations. Three things must be sorted before the cleanup is durable:

1. **The two tools disagree on line length.** `flake8` is invoked with
   `--max-line-length 100`, but `black` (and the configured-but-unused `ruff`,
   `[tool.ruff] line-length = 120`) format to **120**. Left as-is, `black` will
   reflow lines up to 120 that `flake8` then flags as `E501` — the two gates
   fight forever. **Resolve the width first** (recommendation: align `flake8`'s
   `--max-line-length` to **120** to match the formatter, since `black` is
   authoritative on wrapping; or set `black`/`ruff` to 100 — pick one and apply
   everywhere, including the Makefile flags).

2. **The toolchain isn't installed by default here.** On the DGX, conda `base`
   has neither `black` nor `flake8` (`make lint` errors with `No module named
   flake8`; `black` needs `make install-tools`). The Makefile comment claims
   "flake8 is already available via ROS" — true on the Jetson, not on the DGX.
   The baseline must pin/install the toolchain and document where it comes from
   per host so the gate is reproducible.

3. **Cross-lane reformat.** A single repo-wide formatting PR touches Jetson-owned
   `strafer_ros`, shared `strafer_shared`, and DGX-owned `strafer_lab/scripts` in
   one diff. A pure-formatting diff is low-risk but large and crosses ownership
   boundaries. **Prefer splitting per-package** (one PR per tree) so each lane
   reviews its own diff, or explicitly coordinate a single repo-wide pass.

This is the prerequisite for the CI lint gate tracked in
[`test-ci-workflow`](test-ci-workflow.md) (which adds a GitHub Actions matrix over
`make test-*`); a `flake8`/`black --check` step can be folded in once the baseline
is green so it can't regress. Cross-lane gate mechanics overlap with
[`jetson-test-gate-cross-lane-deps`](jetson-test-gate-cross-lane-deps.md).

## Approach

- Resolve the line-length conflict in config + the Makefile flags first.
- Ensure `black` / `flake8` / `autopep8` are installed/pinned and documented per
  host (`make install-tools` covers `black`/`autopep8`).
- Run `make format` (`black`) for the bulk auto-fix, then `make lint` and clear
  the residual `flake8` findings (unused imports `F401`, residual `E501` if any,
  etc.) with `make lint-fix` (`autopep8`) plus hand cleanup.
- **Formatting/imports only — no logic changes.** Review the diff to confirm; if
  the formatter surfaces a genuine bug, file it separately, don't fix it here.
- Land per-package where practical to keep each lane's diff reviewable.

## Acceptance criteria

- [ ] `make lint` exits 0 on the tracked Python source.
- [ ] `make format-check` exits 0.
- [ ] The `flake8`/`black` line-length conflict is resolved to a single
      configured width, reflected in `pyproject.toml` and the Makefile flags.
- [ ] The lint/format toolchain is installed/pinned and how to obtain it per host
      is documented (Makefile / README / env setup).
- [ ] No behavior change: the diff is formatting/imports only, and the touched
      packages' smokes stay green (`make test-lab-pure`, `make test-autonomy`,
      and — on the Jetson — `make test-ros` as applicable to what was reformatted).
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Out of scope

- The CI enforcement step itself — that's [`test-ci-workflow`](test-ci-workflow.md).
  This brief just makes the baseline green so the gate has something to enforce.
- Migrating off `flake8` + `black` to `ruff` (already a listed dep with
  `line-length = 120` but unused) — a separate tooling decision; note it but don't
  do it here.
- Fixing any logic bug a formatter/linter incidentally surfaces — file separately.
- The two failing `test_sim` suites surfaced alongside this (env-registration
  `EXPECTED_ENVS` drift; the `curriculums` Kit subprocess crash) — those are
  test-correctness issues, not style, and are tracked separately.
