# Archive / rationalize the interim top-level architecture docs

**Type:** docs lifecycle (cross-epic)
**Owner:** DGX agent
**Priority:** P3 — blocks no acceptance bar, but the interim docs are
actively misleading (they predate the epics that now own their
subjects) and the Tier 2 harness review surfaced three of them being
half-maintained inside an implementation PR.
**Estimate:** Phase 1 (stamp + sweep-exemption) ✅ done; Phase 2 (physical
relocation + referrer rehoming) is the remaining M — the rehoming for
`MISSION_VALIDATION_ARCHITECTURE.md` is the real work (26 referrers).
**Branch:** task/archive-interim-architecture-docs

## Status

**Phase 1 shipped 2026-06-19** (the churn-stopper): all five interim docs
carry an `> **INTERIM**` banner, and `context/conventions.md`'s
user-facing-docs maintenance sweep was edited to **exempt** banner-stamped
docs and drop them from the sweep table. That removes the contractual
pressure that kept dragging implementation PRs into these files (the
recurring annoyance). Phase 2 — physically moving them to `docs/archived/`
and rehoming referrers — is the remaining work below, deferrable now that
nothing re-edits them.

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

**Root cause found + fixed (Phase 1).** These docs were edit-magnets
*because* `context/conventions.md`'s user-facing-docs maintenance sweep
listed them as surfaces every brief MUST update. The sweep now exempts
any `> **INTERIM**`-bannered doc and no longer lists them, so the
contractual pull is gone. The banner + exemption is the durable fix;
the physical move is now just tidiness.

### The five interim docs and their disposition

The set grew from three to five (operator add 2026-06-19:
`STRAFER_AUTONOMY_NEXT.md`, `SIM_TO_REAL_TUNING_GUIDE.md`). All five are
banner-stamped + sweep-exempt as of Phase 1; the per-doc Phase-2 plan:

| Doc | Phase 2 disposition | Successor |
|---|---|---|
| `INTEGRATION_SIM_IN_THE_LOOP.md` | move to `docs/archived/` + rehome ~18 referrers | `HARNESS_DATA_CAPTURE.md` + `example_commands_cheatsheet.md` |
| `SYSTEM_FLOW_DIAGRAMS.md` | trim to the unified LeRobot v3 flow, or archive if little remains | the unified capture flow; clip-validation owns per-consumer training flows |
| `MISSION_VALIDATION_ARCHITECTURE.md` | **stay in place** (26 referrers); full archive gated on clip-validation shipping its authoritative replacement | the clip-validation epic |
| `STRAFER_AUTONOMY_NEXT.md` | move to `docs/archived/` + rehome ~16 referrers | the multi-room epic (`autonomy-stack` + BOARD) |
| `SIM_TO_REAL_TUNING_GUIDE.md` | **verify staleness first** — no staleness markers found at stamp time; if still a current reference, drop its banner + re-add to the sweep instead of archiving | sim→real promotion: `nav2-sim-real-promotion-architecture` |

### Original three-doc disposition (kept for context)

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

### Phase 1 — stamp + sweep-exemption (✅ done 2026-06-19)

- [x] All five interim docs carry an `> **INTERIM**` top-of-file banner
      (strong "not maintained" for INTEGRATION / SYSTEM_FLOW /
      STRAFER_AUTONOMY_NEXT; soft "superseded section-by-section" for
      MISSION_VALIDATION; "verify before relying" for SIM_TO_REAL_TUNING_GUIDE).
- [x] `context/conventions.md` user-facing-docs sweep drops the interim
      docs from its table and adds the **banner-exemption** rule, so no
      implementation PR is contractually pulled into them.
- [x] No referrers broken (nothing moved in Phase 1).

### Phase 2 — physical relocation + referrer rehoming (deferred)

- [ ] `docs/archived/` created; `INTEGRATION_SIM_IN_THE_LOOP.md` and
      `STRAFER_AUTONOMY_NEXT.md` moved there (header stamp already present).
- [ ] Their referrers (~18 and ~16) rehomed to the archived path or the
      successor. Specifically `harness-architecture`'s acceptance line and
      `mission-generator`'s "Doc surface" item re-pointed off
      `INTEGRATION_SIM_IN_THE_LOOP.md`.
- [ ] `SYSTEM_FLOW_DIAGRAMS.md` trimmed to the unified LeRobot v3 flow (or
      archived if little remains).
- [ ] `MISSION_VALIDATION_ARCHITECTURE.md` stays in place; full archive
      gated on the clip-validation epic shipping its authoritative
      replacement (file the un-park trigger on a parked
      `mission-validation-doc-archive` then).
- [ ] `SIM_TO_REAL_TUNING_GUIDE.md` staleness confirmed before archiving —
      if it is still a current reference, drop its banner + re-add it to the
      conventions sweep instead.
- [ ] `grep` for each archived doc name returns no live (non-archived)
      pointer except the archived copy.
- [ ] If your work invalidates a fact in any context module / README /
      guide, update it in the same commit.

## Out of scope

- The clean first-run `SIM_IN_THE_LOOP` user-guide rewrite (file
  `sim-in-the-loop-user-guide` parked if wanted).
- Writing the authoritative mission-validation architecture — that is
  the clip-validation epic's, when its features are set.
- Any code change. Docs only.
