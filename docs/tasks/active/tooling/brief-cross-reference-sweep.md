# Sweep legacy flat-slug brief cross-references after the epic-structure reorg

**Type:** bug / docs hygiene
**Owner:** Either (mechanical sweep + manual sanity check on
ambiguous cases)
**Priority:** P2 (no blocked work, but every broken link is a
discoverability tax on every agent that picks a brief)
**Estimate:** S (~half day; grep + per-link replacement + smoke
that no remaining `<epic>-` legacy prefixes resolve to
non-existent files)
**Branch:** task/brief-cross-reference-sweep

## Story

As an **agent picking up a brief who clicks the
`[harness-mission-generator](harness-mission-generator.md)`-
style links to read sibling context**, I want **every brief
cross-reference in `docs/tasks/` to resolve to the actual file
under the post-reorg `<epic>/<basename>.md` layout, not to the
legacy flat-naming slugs that no longer exist as filenames**, so
that **the task-board cross-reference graph isn't quietly broken
across most epics, and the audit/onboarding cost of navigating
the queue stops being a low-grade time tax**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`../README.md`](../../README.md) §"Directory layout" —
  defines the post-reorg `<epic>/<basename>.md` convention
  these links should target.

Sibling briefs that show the reorg's tip-of-iceberg breakage:
- [`teleop-driver`](../harness/teleop-driver.md) (and its
  harness epic peers) — every cross-brief link previously
  referenced `<epic>-<basename>` slugs; the harness-audit PR
  fixed them in place but the same pattern exists in
  multi-room, trained-policy, clip-validation, reliability,
  investigations.

## Context

### The breakage

[`completed/task-board-epic-structure.md`](../../completed/task-board-epic-structure.md)
moved every active / parked brief from `docs/tasks/<flat>.md`
to `docs/tasks/<active|parked>/<epic>/<basename>.md` in a single
PR. The brief contents themselves were not rewritten — they
still link to siblings via the legacy flat slug, e.g.:

```markdown
[`harness-mission-generator`](harness-mission-generator.md)
[`multi-room-scene-connectivity-validation`](multi-room-scene-connectivity-validation.md)
[`strafer-vla-v2-architecture`](strafer-vla-v2-architecture.md)
```

None of those filenames exist anymore. The actual paths are:

```
docs/tasks/active/harness/mission-generator.md
docs/tasks/active/multi-room/scene-connectivity-validation.md
docs/tasks/parked/experimental/vla-v2-architecture.md
```

The harness-epic audit fixed this in place for the five harness
briefs. The same pattern remains everywhere else.

### What this brief does

A mechanical sweep + a thin manual layer for ambiguous cases:

1. **Enumerate broken links.** For every `.md` under
   `docs/tasks/active/`, `docs/tasks/parked/`, and
   `docs/tasks/completed/`, grep for markdown links to `.md`
   files. For each link, check whether the resolved path
   exists. List the broken ones.
2. **Build the rename map.** Most broken links resolve
   mechanically:
   - `<epic>-<basename>.md` → `<epic>/<basename>.md` (same dir
     if same epic, `../<epic>/<basename>.md` otherwise).
   - `strafer-<basename>.md` → check whether it landed in
     `trained-policy/`, `experimental/`, or `harness/`; most
     are unambiguous.
   - `clip-<basename>.md` → `clip-validation/<basename>.md`
     (active or parked).
   - `multi-room-<basename>.md` → `multi-room/<basename>.md`.
3. **Manual triage for ambiguous cases.** Some legacy slugs may
   not map cleanly (e.g., a brief was renamed in the reorg, or
   merged into another, or shipped + lives under
   `completed/`). For those, the brief author resolves manually.
4. **Add a lint to prevent recurrence.** A simple Bash one-liner
   in CI (or a pre-commit hook) that greps `docs/tasks/` for
   `(]\([a-z-]+\.md\))` patterns and reports any link that
   doesn't resolve to a file. Even a non-fatal warning gets the
   problem caught before merge.

### Why this is one brief, not per-epic

The renames are mechanical and run faster as one sweep. Per-epic
PRs would re-do the same grep + replace 7 times. The audit
landed the harness fixes in the audit PR because the audit
needed the link graph correct to reason about cross-brief
dependencies; for the rest of the queue, one consolidated PR is
cleaner.

## Acceptance criteria

- [ ] **Broken-link enumeration.** A grep pass over
      `docs/tasks/` listing every markdown link whose target
      doesn't resolve. Posted as a comment on the PR (or a
      one-off `link-audit.txt` artifact dropped under the PR
      branch).
- [ ] **Renamed in place.** Every broken link is updated to its
      resolved path. Ambiguous cases (target brief was renamed
      AND moved) are resolved per the brief author's judgement;
      list them in the PR description with the chosen target.
- [ ] **Lint helper.** A `tools/check_brief_links.sh` (or
      similar) script that re-runs the broken-link enumeration
      in CI / locally, with a non-zero exit on breakage.
      Documented in [`docs/tasks/README.md`](../../README.md)'s
      "How to write a good brief" section.
- [ ] **Sweep verification.** Re-run the lint after rename;
      report zero broken links.
- [ ] **No source changes.** This brief touches `docs/` only and
      optionally `tools/` for the lint script. No
      `source/`-tree edits.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- One-liner enumeration starter:
  ```
  rg -nP '\]\(([A-Za-z0-9_./-]+\.md)\)' docs/tasks/ -o -r '$1' \
    | sort -u \
    | while read link path; do
        target=$(dirname "$link")/$path
        [ -f "$target" ] || echo "BROKEN: $link -> $path"
      done
  ```
  Refine to handle relative paths and same-dir refs correctly.
- The renaming map can mostly be built from the post-reorg
  `git log` for the epic-structure PR — the PR's diff shows
  every old→new path explicitly.
- Briefs in `docs/tasks/completed/` reference paths under
  `active/<epic>/` in their `Follow-ups` lines; those links are
  also subject to the same fix.

## Out of scope

- **Adding a brief-format linter** beyond the broken-link
  check. Format consistency (frontmatter ordering, acceptance
  bullets, etc.) is a future brief if drift becomes a real
  problem.
- **Auto-generating the BOARD.md index** from brief
  frontmatter. The board is updated by hand per the
  maintenance contract; replacing that with generation is a
  larger architectural change.
- **Renaming briefs** to match a stricter convention.
  Filenames are what they are; this brief only fixes links to
  them.
- **Touching `source/` or other docs outside `docs/tasks/`.**
  Stay in the task-board surface.
