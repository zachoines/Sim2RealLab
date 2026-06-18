# Archive / rationalize the interim top-level architecture docs

**Type:** docs lifecycle (cross-epic)
**Owner:** DGX agent
**Priority:** P3 — blocks no acceptance bar, but the interim docs are
actively misleading (they predate the epics that now own their
subjects) and the Tier 2 harness review surfaced three of them being
half-maintained inside an implementation PR.
**Estimate:** M (the move is small; the reference-rehoming for
`MISSION_VALIDATION_ARCHITECTURE.md` is the real work — 26 referrers).
**Branch:** task/archive-interim-architecture-docs

## Story

As a **reader trying to understand the system**, I want **the top-level
`docs/*.md` architecture write-ups that predate the `docs/tasks/`
epics either archived or trimmed to what is actually true today**, so
that **I'm not led by an interim doc that describes a design the epics
have since superseded, and the epics own the authoritative write-ups.**

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)

## Context

The `docs/tasks/` epic structure (and its `context/` modules) became
the authoritative home for design intent. Several top-level `docs/*.md`
files were written *before* that — bringup guides and interim
architecture sketches — and have drifted. The Tier 2 harness PR (#88)
review caught three being edited to stay current inside a driver PR,
which is the wrong vehicle. This brief is the right one.

The principle (from the project's docs-era model): **an implementation
PR should not maintain an interim top-level doc; either the doc is
archived, or its authoritative successor lives in the owning epic.**

### The three docs and their disposition

**1. `docs/INTEGRATION_SIM_IN_THE_LOOP.md` → archive.** Served its
purpose during Isaac-ROS bridge bringup; day-to-day sim-in-the-loop is
now driven from `docs/example_commands_cheatsheet.md` copy-paste
commands, and `docs/HARNESS_DATA_CAPTURE.md` owns the capture story.
Low referrer count. Move to `docs/archived/` with a header stamp (date,
why, successor pointers). A future *clean* first-run `SIM_IN_THE_LOOP`
user-guide rewrite is out of scope here — file it parked
(`sim-in-the-loop-user-guide`, filed-on-trigger) if wanted.

**2. `docs/SYSTEM_FLOW_DIAGRAMS.md` → audit + trim (not archive).** It
depicts separate "VLM flow" and "CLIP flow" data paths. That is **not**
how the harness epic works: there is one unified LeRobot v3 collection
scheme across drivers (teleop / bridge / scripted), and the per-model
training divergence is downstream and **pending** in the clip-validation
epic. Remove the stale separate-flow content; keep only the unified
capture flow that is true today; leave authoritative per-consumer
training flows to the clip-validation briefs to add when those features
are set. If after trimming little remains, archive it instead.

**3. `docs/MISSION_VALIDATION_ARCHITECTURE.md` → stamp interim now;
full-archive gated on clip-validation.** This is the load-bearing one:
**26 referrers** including active `validator-evaluation`,
`mission-generator`, `autonomy-stack`, the `perception-backbone`
context spine, and four parked clip-validation briefs — several citing
specific sections (§2.6, §3.6, §4). A blind `git mv` to `archived/`
breaks all of those links. Disposition:
- **Now:** add a top-of-file banner — "INTERIM. Predates the
  clip-validation epic; superseded section-by-section as those briefs
  ship. The authoritative mid-mission-validation architecture will live
  in the clip-validation epic." Do **not** move the file.
- **Later (gated):** full archive + reference rehoming once the
  clip-validation epic has produced its authoritative replacement.
  Capture that as the un-park trigger on a parked sibling
  (`mission-validation-doc-archive`) rather than forcing it now.

This split honors "stop treating it as authoritative" immediately
without dangling 26 references.

## Acceptance criteria

- [ ] `docs/archived/` created; `INTEGRATION_SIM_IN_THE_LOOP.md` moved
      there with a header stamp (date, reason, successor pointers to
      `HARNESS_DATA_CAPTURE.md` + `example_commands_cheatsheet.md`).
- [ ] Any referrer to `INTEGRATION_SIM_IN_THE_LOOP.md` updated to the
      archived path or re-pointed at the successor. Specifically:
  - [`harness-architecture`](../harness/harness-architecture.md)
    acceptance criterion that says "update
    `INTEGRATION_SIM_IN_THE_LOOP.md`" is rewritten to "the unified
    `capture.py` entry point is documented in `HARNESS_DATA_CAPTURE.md`
    + `example_commands_cheatsheet.md`; `INTEGRATION_SIM_IN_THE_LOOP.md`
    archived."
  - [`mission-generator`](../harness/mission-generator.md)'s "Doc
    surface" acceptance item (Stage 5b/5c on
    `INTEGRATION_SIM_IN_THE_LOOP.md`) re-pointed at the successor.
- [ ] `SYSTEM_FLOW_DIAGRAMS.md` trimmed to the unified LeRobot v3
      capture flow; the separate VLM-flow / CLIP-flow depiction removed;
      a one-line note that per-consumer training flows are owned by the
      clip-validation epic. (Or archived if little remains.)
- [ ] `MISSION_VALIDATION_ARCHITECTURE.md` gains the INTERIM banner;
      file stays in place; **no referrers broken**. A parked
      `mission-validation-doc-archive` brief is filed with the un-park
      trigger "clip-validation epic has shipped the authoritative
      mid-mission-validation write-up."
- [ ] `grep -rn "INTEGRATION_SIM_IN_THE_LOOP" docs/ source/` returns no
      live (non-archived) pointer except the archived copy itself.
- [ ] If your work invalidates a fact in any context module, package
      README, or guide under `docs/`, update it in the same commit.

## Out of scope

- The clean first-run `SIM_IN_THE_LOOP` user-guide rewrite (file
  `sim-in-the-loop-user-guide` parked if wanted).
- Writing the authoritative mission-validation architecture — that is
  the clip-validation epic's, when its features are set.
- Any code change. Docs only.
