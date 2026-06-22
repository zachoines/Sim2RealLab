# Sub-group `scripts/` by sub-system + amend the placement rule

**Type:** refactor / tooling (convention amendment + history-preserving moves)
**Owner:** DGX agent (lane: `source/strafer_lab/scripts/`, `source/strafer_vlm/`, top-level `Makefile`, the workstation-facing `docs/`)
**Priority:** P3 — tooling polish; doesn't block features.
**Estimate:** M (a `git mv` + path-sweep like its parent brief, plus the one-paragraph rule amendment; the harness-cluster sequencing and the final taxonomy are the only judgement parts).
**Branch:** task/script-tool-subsystem-grouping

> **Scope note (consolidated with [`tools-package-reorg`](../../parked/tooling/tools-package-reorg.md)):**
> the **`tools/`** sub-grouping is owned by [`tools-package-reorg`](../../parked/tooling/tools-package-reorg.md)
> (filed first, more detailed on the importer sweep). This brief owns the
> **`scripts/`** sub-grouping **and the shared placement-rule amendment** that
> governs *both* dirs. The two are complementary — land independently; the rule
> here is the contract `tools-package-reorg` conforms to.

## Story

As **an agent adding or finding a workstation-side script**, I want
**runnable `scripts/` sub-divided by sub-system (policy / scene-gen / harness /
diagnostics)**, so that **a ~14-entry flat `scripts/` (growing to ~20+ as the
`eval_*` / `build_*` wave lands) stays navigable, and a new file's home is
obvious from its sub-system** — with one amended placement rule that also
governs the parallel `tools/` reorg.

This is the natural sequel to
[`script-tooling-layout-consolidation`](../../completed/script-tooling-layout-consolidation.md),
which dissolved the three scattered locations into one flat `scripts/` + one
flat `tools/` per package. That brief deliberately stopped at *flat* (only
`asset_authoring/` was carved out). This brief decides whether — and how — to
go one level deeper for `scripts/`, and writes the rule for both.

## Context bundle

Read these before starting:
- [`../../completed/script-tooling-layout-consolidation.md`](../../completed/script-tooling-layout-consolidation.md) — the parent brief; its **target layout**, the **placement rule** it wrote into `conventions.md`, and its **per-file move table** (the caller map this brief re-walks).
- [`context/conventions.md`](../../context/conventions.md) — the **`## Script and tool placement`** section this brief **amends** (flat → sub-system folders), plus the "no `misc/` bucket" warning under `## Task board structure` (the same anti-pattern applies to a `scripts/misc/`).
- [`context/repo-topology.md`](../../context/repo-topology.md) — the "Key entry-point scripts" table + the `$ISAACLAB -p <path>` invocation contract (every sub-folder move changes a documented path).
- **Sibling briefs — coordinate:**
  - [`tools-package-reorg`](../../parked/tooling/tools-package-reorg.md) — the **`tools/`** half. Lands under the rule amended here. Sequence per its own brief (it's gated behind in-flight `tools/`-touching PRs).
  - [`../harness/harness-architecture.md`](../harness/harness-architecture.md) — owns `capture.py`, `teleop_capture.py`, `collect_demos.py`, `run_sim_in_the_loop.py`. The `scripts/harness/` group **must be sequenced behind it** (see [Coordination](#coordination--sequencing)).
  - [`unify-test-targets-and-ci.md`](../../completed/unify-test-targets-and-ci.md) / [`strafer-lab-test-tree-unification.md`](../../completed/strafer-lab-test-tree-unification.md) — own `make test-*` + test files; this brief only updates import targets, never test files.

## The problem

After the parent brief, `scripts/` is correct-but-flat: ~14 runnable entry
points (plus `asset_authoring/`, `type_stubs/`, `retired/`). The placement
rule's tie-breaker ("imported or run?") decides `scripts/` vs `tools/`, but
**once a file is a script there's no further guidance** — and the incoming
epics add `eval_transit_monitor`, `eval_room_state`, `build_mission_corpus`,
`export_backbone_onnx` (and more), pushing toward ~20.

Flat is fine at this size; the bet is that the `eval_*`/`build_*` wave makes
sub-system grouping pay off, and that **deciding the taxonomy now** (before
that wave lands) avoids a third reorg. (`tools/` has the same shape and is
handled by [`tools-package-reorg`](../../parked/tooling/tools-package-reorg.md).)

**The cost of getting it wrong:** sub-folders re-introduce a placement
question ("which sub-system?") that the parent brief's flat rule had
eliminated. So the amendment must ship an **enumerable sub-system list with a
defined default**, or it trades one coin-flip for another.

## Proposed `scripts/` taxonomy (open for review)

| Group | Members | Notes |
|---|---|---|
| `scripts/policy/` | `train_strafer_navigation`, `play_strafer_navigation`, `export_policy`, `benchmark_policy` | The policy lifecycle. **Open sub-decision:** split `training/` (train/play) from `policy/` (export/benchmark)? |
| `scripts/scene_gen/` | `prep_room_usds`, `postprocess_scene_usd`, `generate_scenes_metadata`, `extract_scene_metadata`, `validate_scene_connectivity` | The scene-generation pipeline. **Naming note:** prefer `scene_gen/` over `infinigen/` — the group mixes *Infinigen-coupled* steps (`prep_room_usds`, and the prim-name parsing inside `generate_scenes_metadata` / `extract_scene_metadata`) with *source-agnostic* ones (`postprocess_scene_usd` attaches colliders to any USD; `validate_scene_connectivity` authors `connectivity[]` + `occupancy.npy` over any USD — it is a producer by *pipeline ordering*, not by Infinigen coupling). The Infinigen-specific *logic* lives inside those scripts + the `infinigen_label_parser` tool, which the parallel [`tools-package-reorg`](../../parked/tooling/tools-package-reorg.md) isolates. `validate_scene_connectivity` was previously unassigned — without a home it falls to the flat default; it belongs here (the scene-gen pipeline it runs in). |
| `scripts/diagnostics/` | `test_strafer_env`, `roller_bounce_probe` | Smoke / physics-regression probes. |
| `scripts/harness/` | `capture`, `teleop_capture`, `collect_demos`, `run_sim_in_the_loop` | Data-capture + sim bridge. **Sequenced behind harness-architecture.** `capture`↔`teleop_capture` must stay siblings (subprocess resolver) — move together. `collect_demos` is RL demo-collection (DAPG/GAIL), grouped here for the shared gamepad path, not because it is harness-epic. |
| `scripts/asset_authoring/` | *(unchanged)* | Already carved out by the parent brief. |
| `scripts/type_stubs/`, `scripts/retired/` | *(unchanged)* | Dev tooling; harness-emptied holding area. |

**Default-home decision (must be in the rule):** a new runnable that fits no
group goes … where? Options: (a) top-level `scripts/` (flat fallback — keeps a
home, no `misc/`), (b) a new named group if a third sibling appears. Recommend
**(a) flat fallback**, and "promote to a group on the third member."

## The rule amendment (the durable contribution — governs both `scripts/` and `tools/`)

Amend `## Script and tool placement` in
[`conventions.md`](../../context/conventions.md): `scripts/` and `tools/` are
sub-divided by **sub-system** (an enumerable, small set — list it), not flat.
A new file picks the sub-system folder; if none fits, it stays at the dir top
level (no `misc/` — the urge to add one is the signal a new sub-system folder
is genuinely needed, promoted on the third member). Keep the existing
`tools/` vs `scripts/` tie-breaker and the `retired/` semantics unchanged.
The concrete `tools/` taxonomy lives in [`tools-package-reorg`](../../parked/tooling/tools-package-reorg.md);
this rule is the convention both reorgs satisfy.

## Coordination & sequencing

- **Non-harness `scripts/` groups first** (`policy/`, `infinigen/`,
  `diagnostics/`): no harness dependency — pickable as soon as this brief is.
- **`scripts/harness/` is gated on `harness-architecture`.** It re-moves
  files that brief owns + are Makefile-wired (`sim-bridge`/`-gui`/`-harness`
  → `Makefile` invoke `run_sim_in_the_loop.py`; `capture.py`↔`teleop_capture.py`
  subprocess hand-off). Land it after harness-architecture stabilizes (or
  bundle with one of its tier PRs), updating `harness-architecture.md`'s paths
  in the same PR — same contract the parent brief used for the `capture.py`
  move.
- **The rule amendment can land first / standalone** — it's a one-paragraph
  convention edit and unblocks both this brief's moves and `tools-package-reorg`.
- **`tools/` is [`tools-package-reorg`](../../parked/tooling/tools-package-reorg.md)'s job**, sequenced
  per that brief (behind in-flight `tools/`-touching PRs + the harness-retirement
  PR that deletes `dataset_export.py`).

## Acceptance criteria

- [ ] `conventions.md`'s placement rule is amended to the sub-system layout with an enumerated sub-system set and a defined default home; the "no `misc/`" guard is restated for `scripts/`/`tools/`. (This is the shared contract `tools-package-reorg` also conforms to.)
- [ ] Each `scripts/` group exists; every moved path is updated in the same commit (Makefile, cheatsheet, READMEs, repo-topology, context modules, active/parked briefs); `make sim-bridge`/`-harness` + the training/export/capture smokes run from the new paths.
- [ ] `scripts/harness/` moves are coordinated with `harness-architecture` and its references repointed in the same PR; `capture.py`↔`teleop_capture.py` resolver still resolves.
- [ ] No source file references a pre-move path by a stale string; `tools/check_brief_links.sh` no worse than baseline; Mermaid edge-label sweep clean.

## Out of scope

- **`tools/` sub-grouping** — owned by [`tools-package-reorg`](../../parked/tooling/tools-package-reorg.md); this brief only writes the shared rule it conforms to.
- **`strafer_ros/`** and the Jetson lane.
- **Re-litigating `scripts/` vs `tools/`** (the parent brief's tie-breaker stands; this brief only adds the second axis).
- **`make` wrappers / invocation unification** — separate concern.
- **Creating the future `eval_*`/`build_*` scripts** — this brief only ensures they have an unambiguous sub-system home when they land.
