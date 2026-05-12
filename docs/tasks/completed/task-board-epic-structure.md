# Reorganize the task board into epic subdirectories with a parked sibling

**Status:** Shipped 2026-05-12 in `b32537f` (Either).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/29

**Type:** task / convention
**Owner:** Either (the change is to `docs/tasks/` structure — no source-tree work — but lands in this PR with cross-host implications since both agents pick from this board)
**Priority:** P2 (active-queue growth is making `ls active/` hard to scan; left untreated, the trend continues. Not blocking any feature today, but the cost of fixing it grows with brief count.)
**Estimate:** S–M (~half a day; mostly a big-bang `git mv` + link rewrites + README/board updates. No source code touched.)
**Branch:** task/task-board-epic-structure

## Story

As **either-lane agents picking work off the task board**, we want
**briefs grouped by feature epic in the filesystem AND in
BOARD.md, with filed-on-trigger / dependency-blocked work moved out
of `active/` into a `parked/` sibling**, so that **`ls
docs/tasks/active/` answers "what's pickable right now grouped by
the feature it serves" rather than "what's the alphabetical list of
every brief that's been filed since the project started, including
ones that aren't really pickable."**

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)
- [README.md](../README.md) — the existing convention this brief
  amends. `## File naming`, `## Shipping a brief`, and the agent-
  launcher template are the three sections that need surgery.
- [BOARD.md](../BOARD.md) — the existing index. Every link in it
  needs path updates; an `## Epics` section gets added on top of
  the existing priority-first view.

## Context

The board has 31 active briefs as of this filing. Symptoms that
motivated this brief:

- `ls docs/tasks/active/` returns 31 alphabetically-sorted entries
  with no visible grouping. A fresh agent has to scan all of them
  to find work in a specific feature area, or open BOARD.md to get
  the grouping the filesystem doesn't provide.
- Roughly 6 of the 31 are filed-on-trigger or blocked-on-deps —
  they're already listed under BOARD.md's `## Blocked` section,
  but they sit alphabetically among pickable work in the
  filesystem, polluting the at-a-glance pickup view.
- Tech-debt / refactor briefs are sprinkled among feature briefs.
  Reasoning about "what tech debt is queued vs. what's net-new
  capability" requires a manual pass through 31 file names.

The current convention (one flat `active/` directory + a rich
`BOARD.md` index) was correct when the project had ~5 active
briefs. At ~30 it's straining. At ~50, which the current pace will
hit within a quarter, it will be actively confusing.

Two interventions:

1. **Epic subdirs** in `active/` (and a mirrored `parked/`). Nine
   epics, decided in the filing session:
   - `multi-room/` — cross-room navigation as MVP default
   - `trained-policy/` — RL-policy training + Jetson deployment
   - `harness/` — training-data pipeline
   - `clip-validation/` — mid-mission validation via CLIP
   - `sim-performance/` — bridge throughput + renderer
   - `reliability/` — nav + executor bugs + autonomy refactors
   - `tooling/` — testing, CI, workstation bringup
   - `experimental/` — long-horizon research (VLA v2 etc.)
   - `investigations/` — one-shot measurements with no
     committed implementation
2. **Parked sibling.** `docs/tasks/parked/` mirrors `active/`'s
   subdir structure. Six briefs migrate here (see Appendix A
   below). Un-park by `git mv parked/<epic>/<name>.md
   active/<epic>/<name>.md`.

Branch naming convention is unchanged: `task/<basename>` (the
subdirectory does not leak into the branch name). The filesystem
path is for organization; the branch is for the worktree.

## Approach

Big-bang migration in one PR. The cost of a partial migration
(half-flat half-epic during a transition window) is worse than the
cost of the one large diff.

### Step 1: Author the convention brief

This file (the one you're reading). Lives at top-level
`active/task-board-epic-structure.md` rather than inside an epic
dir, since it's the brief that defines the subdir convention —
chicken-and-egg if it sat inside one. On ship it moves to
`completed/task-board-epic-structure.md` (completed/ stays flat;
see Out of scope).

### Step 2: Create epic subdirs and move briefs

```
docs/tasks/active/
├── multi-room/
│   ├── autonomy-stack.md                 (was: multi-room-autonomy-stack.md)
│   ├── scene-connectivity-validation.md  (was: multi-room-scene-connectivity-validation.md)
│   └── planner-far-target-staging.md
├── trained-policy/
│   ├── export-onnx-depth.md              (was: policy-export-onnx-depth.md)
│   ├── loader-recurrent-state.md         (was: policy-loader-recurrent-state.md)
│   ├── inference-package.md              (was: strafer-inference-package.md)
│   ├── goal-noise-training.md            (was: policy-goal-noise-training.md)
│   └── subgoal-env.md                    (was: strafer-lab-subgoal-env.md)
├── harness/
│   ├── teleop-driver.md                  (was: harness-teleop-driver.md)
│   ├── mission-generator.md              (was: harness-mission-generator.md)
│   ├── behavior-cloning-data-expansion.md  (was: harness-behavior-cloning-data-expansion.md)
│   └── trajectory-first-captioning.md    (was: harness-trajectory-first-captioning.md)
├── clip-validation/
│   └── validator-evaluation.md           (was: clip-mid-mission-validator-evaluation.md)
├── sim-performance/
│   ├── bridge-throughput-toward-25hz.md
│   └── isaac-sim-rt-2-default-renderer.md
├── reliability/
│   ├── nav-deadline-sim-time-audit.md
│   ├── executor-prefer-rotate-then-translate.md
│   ├── rtabmap-cold-start-determinism.md
│   ├── planner-rotate-direction-prompt.md
│   └── grounding-publisher-extraction.md
├── tooling/
│   ├── unify-test-targets-and-ci.md
│   └── windows-workstation-bringup.md
└── investigations/
    ├── next-integration-round.md
    ├── real-d555-depth-range-survey.md
    └── training-throughput-profile-and-investigate.md
```

Filename simplifications happen opportunistically when the brief's
old name redundantly encoded the now-explicit epic prefix:

- `harness-teleop-driver.md` → `harness/teleop-driver.md`
- `policy-export-onnx-depth.md` → `trained-policy/export-onnx-depth.md`
- `multi-room-autonomy-stack.md` → `multi-room/autonomy-stack.md`
- `clip-mid-mission-validator-evaluation.md` → `clip-validation/validator-evaluation.md`
- `strafer-inference-package.md` → `trained-policy/inference-package.md`
- `strafer-lab-subgoal-env.md` → `trained-policy/subgoal-env.md`

Other briefs keep their full name (no redundant prefix to strip),
to avoid invasive rename churn for slim semantic gain.

### Step 3: Create `parked/` with mirrored subdirs, move 6 briefs

```
docs/tasks/parked/
├── harness/
│   └── oracle-driver.md                  (was: harness-oracle-driver.md)
├── reliability/
│   ├── nav-stall-multilayer-watchdog.md
│   └── perception-side-bearing-service.md
├── clip-validation/
│   └── cotrained-retrieval-augmented.md  (was: clip-cotrained-retrieval-augmented.md)
├── trained-policy/
│   └── hybrid-mode.md                    (was: strafer-inference-hybrid-mode.md)
└── experimental/
    └── vla-v2-architecture.md            (was: strafer-vla-v2-architecture.md)
```

After this step, `active/` has 25 briefs, `parked/` has 6.

### Step 4: Update cross-references

Three classes of links break:

- **Brief → context module.** `../context/foo.md` becomes
  `../../context/foo.md` (one extra `../`).
- **Brief → external doc.** `../../INTEGRATION_*.md` becomes
  `../../../INTEGRATION_*.md`.
- **Brief → other brief.** `../completed/foo.md` becomes
  `../../completed/foo.md`. `../active/foo.md` (rare, used in a
  few completed briefs that point back at active follow-ups)
  becomes `../active/<epic>/<new-name>.md` or
  `../parked/<epic>/<new-name>.md`.

Surfaces to sweep:
- All briefs in `active/<epic>/` and `parked/<epic>/`.
- All briefs in `completed/` that reference `../active/...`
  (a handful — see Investigation pointers).
- `docs/tasks/README.md` — the file-naming section and the
  shipping-order-of-operations example paths.
- `docs/tasks/BOARD.md` — every link in the priority/lane tables
  needs path updates; an `## Epics` section gets added at the top.
- `docs/tasks/context/branching-and-prs.md`, `context/README.md`
  — any example brief paths.
- Root `Readme.md`, `docs/STRAFER_AUTONOMY_NEXT.md`,
  `docs/INTEGRATION_SIM_IN_THE_LOOP.md`,
  `docs/MISSION_VALIDATION_ARCHITECTURE.md`, package READMEs —
  anything that hardcodes `docs/tasks/active/<name>.md` paths.

A `grep -rln "docs/tasks/active"` + `grep -rln "../active/" docs/`
sweep catches the bulk.

### Step 5: Update README.md

Add a `## Directory layout` section explaining `active/<epic>/`,
`parked/<epic>/`, and the un-park `git mv` workflow. Update the
file-naming section to note that the slug stays kebab-case but the
epic prefix can be elided if the subdirectory already conveys it.
Update the shipping order-of-operations example paths.

### Step 6: Update conventions.md

Add a `## Task board structure` section pointing at README.md for
the directory layout and stating: (a) the nine epics are a fixed
small set, not free-form tags; (b) adding a new epic requires its
own convention amendment, same as any directory-level change; (c)
parked → active is a `git mv` operation, no other ceremony.

### Step 7: Restructure BOARD.md

Two changes:
1. Add an `## Epics` section near the top, between
   `## In flight` and `## Ready to pick up`, with one table per
   epic listing its briefs (active + parked). Rough priority +
   state visible per row.
2. Update every existing link to use the new paths.

The priority-first `## Ready to pick up` view stays — it answers
"what should I work on next" better than the epic view does. The
epic view answers "what's the state of feature X." Both are useful;
neither replaces the other.

## Acceptance criteria

- [ ] `docs/tasks/active/` contains exactly nine subdirectories
      named per Step 2 above, with the 25 briefs distributed per
      the migration table.
- [ ] `docs/tasks/parked/` contains exactly five subdirectories
      (mirroring the relevant subset of active/), with the 6
      parked briefs distributed per Step 3.
- [ ] `docs/tasks/active/task-board-epic-structure.md` (this
      brief) sits at the top level of `active/`, not inside an
      epic dir. On ship it lands at
      `docs/tasks/completed/task-board-epic-structure.md`.
- [ ] All relative links between briefs, from briefs to context
      modules, and from briefs to external docs resolve correctly
      after the move. Spot-check by `find docs -name "*.md" -exec
      grep -l "](../" {} \;` and rendering a sample of the
      results.
- [ ] `docs/tasks/README.md` has a `## Directory layout` section
      describing the new structure. The shipping order-of-operations
      example path now reads
      `git mv docs/tasks/active/<epic>/<brief>.md docs/tasks/completed/<brief>.md`.
- [ ] `docs/tasks/context/conventions.md` has a `## Task board
      structure` section pointing at README.md and stating the
      epic-set is fixed.
- [ ] `docs/tasks/BOARD.md` has an `## Epics` section listing
      all 31 briefs (active + parked) grouped by epic; the
      existing `## Ready to pick up` / `## Blocked` /
      `## Quick wins` sections still exist with updated paths.
- [ ] Stale `align-after-scan-grounding.md` row removed from
      BOARD.md (shipped in PR #28 but the row wasn't deleted —
      drive-by housekeeping inside the BOARD.md rewrite).
- [ ] Branch naming convention is unchanged. A spot-check in
      `branching-and-prs.md` confirms `task/<basename>` (not
      `task/<epic>/<basename>`).
- [ ] Root `Readme.md`, package READMEs under `source/*/`, and
      docs under `docs/*.md` no longer reference any of the moved
      brief paths at their old locations.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit.

## Investigation pointers

- `docs/tasks/README.md:47-59` — file naming section.
- `docs/tasks/README.md:230-289` — shipping order-of-operations,
  hardcodes the `active/`→`completed/` mv command.
- `docs/tasks/README.md:307-314` — agent-launcher template,
  hardcodes `docs/tasks/active/<brief-name>.md`.
- `docs/tasks/README.md:401-415` — examples reference flat
  `active/...` paths.
- `docs/tasks/BOARD.md` — 31 brief rows, each with a
  `active/<name>.md` link.
- `docs/tasks/completed/goal-projection-depth-range.md`,
  `docs/tasks/completed/policy-export-tooling.md` — known
  completed→active back-references.
- `grep -rln "docs/tasks/active" docs/` — surfaces every external
  doc that hardcodes a brief path.
- `/home/jetson/.claude/projects/-home-jetson-workspaces/memory/MEMORY.md`
  — agent memory references `docs/tasks/context/branching-and-prs.md`
  (which doesn't move) but no specific active briefs that this
  brief moves. Sanity-check anyway.

## Out of scope

- **Restructuring `docs/tasks/completed/`.** Completed briefs are a
  historical record, not a pickup queue. Browsing them by epic adds
  no value the alphabetical layout doesn't already give. Keep flat.
- **Restructuring `docs/tasks/context/`.** Context modules are
  cross-epic by definition (every brief references
  `repo-topology.md`). Subdirectorizing them would be inverted —
  the modules don't belong to epics, the briefs do.
- **Adding a tenth epic.** Nine is what the spread justifies today.
  If a future filing has no natural home, the right move is to
  amend the convention (this brief's successor) rather than
  invent on the spot. Refuse the urge to create a `misc/` bucket.
- **Renaming completed briefs that match the rename pattern
  applied to their active counterparts.** Their old names appear
  in commit messages, PR descriptions, and external references; a
  rename creates back-pointer rot for no gain. Renames apply only
  in active/ and parked/.
- **Migrating filed-on-trigger semantics into frontmatter.** The
  `parked/` directory is the new home for that signal. Briefs
  don't need a `status: parked` field; the path encodes it. Same
  reason briefs don't have a `status: active` field today.
- **Automating the BOARD.md sync.** The board is hand-maintained
  per the existing convention; this brief doesn't try to change
  that. A future tooling brief could generate the Epics section
  from frontmatter, but only after the manual version has bedded
  in.

## Appendix A: Migration table

Full source → destination mapping for the migration commits. The
agent executing this brief generates these mv commands
programmatically from this table.

### `active/` → `active/<epic>/`

| From | To |
|---|---|
| `active/multi-room-autonomy-stack.md` | `active/multi-room/autonomy-stack.md` |
| `active/multi-room-scene-connectivity-validation.md` | `active/multi-room/scene-connectivity-validation.md` |
| `active/planner-far-target-staging.md` | `active/multi-room/planner-far-target-staging.md` |
| `active/policy-export-onnx-depth.md` | `active/trained-policy/export-onnx-depth.md` |
| `active/policy-loader-recurrent-state.md` | `active/trained-policy/loader-recurrent-state.md` |
| `active/strafer-inference-package.md` | `active/trained-policy/inference-package.md` |
| `active/policy-goal-noise-training.md` | `active/trained-policy/goal-noise-training.md` |
| `active/strafer-lab-subgoal-env.md` | `active/trained-policy/subgoal-env.md` |
| `active/harness-teleop-driver.md` | `active/harness/teleop-driver.md` |
| `active/harness-mission-generator.md` | `active/harness/mission-generator.md` |
| `active/harness-behavior-cloning-data-expansion.md` | `active/harness/behavior-cloning-data-expansion.md` |
| `active/harness-trajectory-first-captioning.md` | `active/harness/trajectory-first-captioning.md` |
| `active/clip-mid-mission-validator-evaluation.md` | `active/clip-validation/validator-evaluation.md` |
| `active/bridge-throughput-toward-25hz.md` | `active/sim-performance/bridge-throughput-toward-25hz.md` |
| `active/isaac-sim-rt-2-default-renderer.md` | `active/sim-performance/isaac-sim-rt-2-default-renderer.md` |
| `active/nav-deadline-sim-time-audit.md` | `active/reliability/nav-deadline-sim-time-audit.md` |
| `active/executor-prefer-rotate-then-translate.md` | `active/reliability/executor-prefer-rotate-then-translate.md` |
| `active/rtabmap-cold-start-determinism.md` | `active/reliability/rtabmap-cold-start-determinism.md` |
| `active/planner-rotate-direction-prompt.md` | `active/reliability/planner-rotate-direction-prompt.md` |
| `active/grounding-publisher-extraction.md` | `active/reliability/grounding-publisher-extraction.md` |
| `active/unify-test-targets-and-ci.md` | `active/tooling/unify-test-targets-and-ci.md` |
| `active/windows-workstation-bringup.md` | `active/tooling/windows-workstation-bringup.md` |
| `active/next-integration-round.md` | `active/investigations/next-integration-round.md` |
| `active/real-d555-depth-range-survey.md` | `active/investigations/real-d555-depth-range-survey.md` |
| `active/training-throughput-profile-and-investigate.md` | `active/investigations/training-throughput-profile-and-investigate.md` |

### `active/` → `parked/<epic>/`

| From | To |
|---|---|
| `active/harness-oracle-driver.md` | `parked/harness/oracle-driver.md` |
| `active/nav-stall-multilayer-watchdog.md` | `parked/reliability/nav-stall-multilayer-watchdog.md` |
| `active/perception-side-bearing-service.md` | `parked/reliability/perception-side-bearing-service.md` |
| `active/clip-cotrained-retrieval-augmented.md` | `parked/clip-validation/cotrained-retrieval-augmented.md` |
| `active/strafer-inference-hybrid-mode.md` | `parked/trained-policy/hybrid-mode.md` |
| `active/strafer-vla-v2-architecture.md` | `parked/experimental/vla-v2-architecture.md` |
