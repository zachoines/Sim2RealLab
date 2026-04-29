# Establish task → branch → PR convention; retrofit active briefs

**Status:** Shipped 2026-04-28 in `91b11b8` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/11

**Type:** task / refactor / docs
**Owner:** Jetson agent
**Priority:** P1
**Estimate:** S (~few hours; one context module + brief retrofit pass)
**Branch:** task/branch-per-task-convention

## Story

As a **maintainer of this multi-agent monorepo**, I want **each task
brief in `docs/tasks/*.md` to map to one short-lived branch off
`main` and one PR**, so that **long-lived shared branches like
`phase_15-isaaclab3` don't accumulate ~100 commits of unrelated work
across multiple months before they land**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)
- [context/conventions.md](context/conventions.md)

## Context

Pre-convention, `phase_15-isaaclab3` accumulated 109 commits / +31k
LOC / 183 files spanning Isaac Lab 3.0 migration, sim-in-the-loop
bridge work, autonomy stack polish, the docs/tasks system itself, the
headless Foxglove visualizer, and the SLAM floor-leak fix. Reviewing
that as a single PR is hard; reverting any one piece is harder; and
the divergence from `main` made every cross-host change feel like a
risky merge.

The convention this brief lands fixes that going forward. The merge
of `phase_15-isaaclab3 → main` is the last monolithic PR; from there
each task uses its own `task/<brief-slug>` branch off `main`, opens a
PR with `gh pr create`, and merges via `--merge` (not squash) so
per-commit history survives.

This brief is **self-bootstrapping**: it's the first brief written
under the convention it establishes. Branch name (`task/branch-per-
task-convention`) matches filename. PR will be the first one opened
under the convention.

## Acceptance criteria

- [ ] New module `docs/tasks/context/branching-and-prs.md` exists
      and covers: branch naming, branch-off point, per-task workflow,
      PR composition rules, merge style, what's NOT in scope.
- [ ] [`docs/tasks/context/README.md`](context/README.md) lists the
      new module under "Current modules".
- [ ] [`docs/tasks/context/conventions.md`](context/conventions.md)'s
      "Closed task brief lifecycle" section includes a `**PR:**` line
      in the Shipped-stamp template, with a link to the new module.
- [ ] [`docs/tasks/README.md`](README.md)'s "Brief format" template
      adds `**Branch:** task/<brief-slug>` as a required frontmatter
      field, and the bundle example links the new module.
- [ ] Every active brief in `docs/tasks/*.md` (NOT
      `docs/tasks/completed/*.md`) carries a `**Branch:**` line
      matching `task/<filename-stem>`. Briefs in `completed/` are
      historical and not retrofitted.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.
- [ ] PR opens via `gh pr create --base main --head
      task/branch-per-task-convention …` and merges with `--merge`
      (preserve history).

## Investigation pointers

- The existing context-module style is in
  [`context/conventions.md`](context/conventions.md) — match its
  voice (terse, declarative, link-heavy).
- The brief format is documented in [`README.md:55+`](README.md).
- Active briefs to retrofit (as of this brief's authoring; verify
  with `ls docs/tasks/*.md` before starting):
  - `async-camera-publishers.md`
  - `branch-per-task-convention.md` (this brief)
  - `d555-distortion-model-explicit.md`
  - `integration-prompts-refresh.md`
  - `nav2-far-goal-staging.md`
  - `planner-far-target-staging.md`
  - `real-d555-depth-range-survey.md`
  - `vlm-bbox-overlay.md`
- `gh` is installed on both DGX and Jetson. Run `gh auth login`
  once per host.

## Out of scope

- Retrofitting briefs already in `docs/tasks/completed/`. They
  shipped before the convention existed; their record is git
  history. Don't rewrite the past.
- GitHub branch-protection rules, required-reviews, or CI gating.
  Those are one-time GitHub-UI configuration, not a brief.
- Linters or git hooks that mechanically enforce the convention.
  Author discipline + PR review is sufficient for now; revisit
  only if the convention starts being breached.
- Bulk-renaming any non-brief artifact (CLAUDE.md, repo-root
  README) to mention the convention. The README in
  `docs/tasks/` is the canonical entry point.
