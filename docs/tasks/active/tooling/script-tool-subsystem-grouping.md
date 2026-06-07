# Sub-group `scripts/` and `tools/` by sub-system

**Type:** refactor / tooling (convention amendment + history-preserving moves)
**Owner:** DGX agent (lane: `source/strafer_lab/scripts/`, `source/strafer_lab/strafer_lab/tools/`, `source/strafer_vlm/`, top-level `Makefile`, the workstation-facing `docs/`)
**Priority:** P3 — tooling polish; doesn't block features.
**Estimate:** M (Part A `scripts/` is a `git mv` + path-sweep like its parent brief; Part B `tools/` is smaller in file count but wider in import blast radius. The harness-cluster sequencing and the final taxonomy are the only judgement parts.)
**Branch:** task/script-tool-subsystem-grouping

## Story

As **an agent adding or finding a workstation-side script or tool**, I want
**runnable `scripts/` and importable `tools/` sub-divided by sub-system
(policy / scene-gen / harness / diagnostics; lerobot / teleop / scene /
perception)**, so that **a ~14-entry flat `scripts/` (growing to ~20+ as the
`eval_*` / `build_*` wave lands) and a ~13-module flat `tools/` stay navigable,
and a new file's home is obvious from its sub-system.**

This is the natural sequel to
[`script-tooling-layout-consolidation`](../../completed/script-tooling-layout-consolidation.md),
which dissolved the three scattered locations into one flat `scripts/` + one
flat `tools/` per package. That brief deliberately stopped at *flat* (only
`asset_authoring/` was carved out). This brief decides whether — and how — to
go one level deeper.

## Context bundle

Read these before starting:
- [`../../completed/script-tooling-layout-consolidation.md`](../../completed/script-tooling-layout-consolidation.md) — the parent brief; its **target layout**, the **placement rule** it wrote into `conventions.md`, and its **per-file move table** (the caller map this brief re-walks).
- [`context/conventions.md`](../../context/conventions.md) — the **`## Script and tool placement`** section this brief **amends** (flat → sub-system folders), plus the "no `misc/` bucket" warning under `## Task board structure` (the same anti-pattern applies to a `scripts/misc/`).
- [`context/repo-topology.md`](../../context/repo-topology.md) — the "Key entry-point scripts" table + the `$ISAACLAB -p <path>` invocation contract (every sub-folder move changes a documented path).
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md) — `strafer_autonomy/tests/` (Jetson-adjacent) imports `strafer_lab.tools.*`; **Part B moves break those import sites** unless repointed.
- **Sibling briefs — coordinate:**
  - [`../harness/harness-architecture.md`](../harness/harness-architecture.md) — owns `capture.py`, `teleop_capture.py`, `collect_demos.py`, `run_sim_in_the_loop.py`. The `scripts/harness/` group **must be sequenced behind it** (see [Coordination](#coordination--sequencing)).
  - [`unify-test-targets-and-ci.md`](unify-test-targets-and-ci.md) / [`strafer-lab-test-tree-unification.md`](strafer-lab-test-tree-unification.md) — own `make test-*` + test files; this brief only updates import targets, never test files.

## The problem

After the parent brief, both dirs are correct-but-flat:

- `source/strafer_lab/scripts/` — 14 runnable entry points (plus `asset_authoring/`, `type_stubs/`, `retired/`). The placement rule's tie-breaker ("imported or run?") decides `scripts/` vs `tools/`, but **once a file is a script there's no further guidance** — and the incoming epics add `eval_transit_monitor`, `eval_room_state`, `build_mission_corpus`, `export_backbone_onnx` (and more), pushing toward ~20.
- `source/strafer_lab/strafer_lab/tools/` — 13 importable modules (plus `retired/`). Flat; spans four+ unrelated concerns (LeRobot IO, teleop input, scene accessors, perception writers).

Flat is fine at this size; the bet is that the `eval_*`/`build_*` wave makes sub-system grouping pay off, and that **deciding the taxonomy now** (before that wave lands) avoids a third reorg.

**The cost of getting it wrong:** sub-folders re-introduce a placement question ("which sub-system?") that the parent brief's flat rule had eliminated. So the amendment must ship an **enumerable sub-system list with a defined default**, or it trades one coin-flip for another.

## Proposed taxonomy (open for review)

### Part A — `scripts/` (cheap: scripts are not imported)

| Group | Members | Notes |
|---|---|---|
| `scripts/policy/` | `train_strafer_navigation`, `play_strafer_navigation`, `export_policy`, `benchmark_policy` | The policy lifecycle. **Open sub-decision:** split `training/` (train/play) from `policy/` (export/benchmark)? |
| `scripts/infinigen/` | `prep_room_usds`, `postprocess_scene_usd`, `generate_scenes_metadata`, `extract_scene_metadata` | The Infinigen scene-gen pipeline (note spelling: infinigen). |
| `scripts/diagnostics/` | `test_strafer_env`, `roller_bounce_probe` | Smoke / physics-regression probes. |
| `scripts/harness/` | `capture`, `teleop_capture`, `collect_demos`, `run_sim_in_the_loop` | Data-capture + sim bridge. **Sequenced behind harness-architecture.** `capture`↔`teleop_capture` must stay siblings (subprocess resolver) — move together. `collect_demos` is RL demo-collection (DAPG/GAIL), grouped here for the shared gamepad path, not because it is harness-epic. |
| `scripts/asset_authoring/` | *(unchanged)* | Already carved out by the parent brief. |
| `scripts/type_stubs/`, `scripts/retired/` | *(unchanged)* | Dev tooling; harness-emptied holding area. |

**Default-home decision (must be in the rule):** a new runnable that fits no
group goes … where? Options: (a) top-level `scripts/` (flat fallback — keeps a
home, no `misc/`), (b) a new named group if a third sibling appears. Recommend
**(a) flat fallback**, and "promote to a group on the third member."

### Part B — `tools/` (expensive: every importer moves too)

| Group | Members |
|---|---|
| `tools/lerobot/` | `lerobot_writer`, `lerobot_depth` |
| `tools/teleop/` | `gamepad_reader`, `teleop_buttons`, `teleop_mission_picker` |
| `tools/scene/` | `scene_labels`, `scene_paths`, `spatial_description`, `infinigen_label_parser` |
| `tools/perception/` | `perception_writer`, `bbox_extractor` |
| top-level `tools/` | `phase_profiler` (lone profiling util — flat until a sibling appears) |
| `tools/retired/` | *(unchanged)* `dataset_export` |

Each sub-package needs an `__init__.py`, and **every import site updates**:
`teleop_capture` imports 6 tools modules; `run_sim_in_the_loop` imports
`perception_writer`; the retired scripts import scene/spatial modules; and
**`strafer_autonomy/tests/` imports `strafer_lab.tools.{bbox_extractor,scene_labels,spatial_description,infinigen_label_parser,perception_writer}`** (cross-lane — coordinate). This is why Part B is phased after Part A.

## The rule amendment (the durable contribution)

Amend `## Script and tool placement` in
[`conventions.md`](../../context/conventions.md): `scripts/` and `tools/` are
sub-divided by **sub-system** (an enumerable, small set — list it), not flat.
A new file picks the sub-system folder; if none fits, it stays at the dir top
level (no `misc/` — the urge to add one is the signal a new sub-system folder
is genuinely needed, promoted on the third member). Keep the existing
`tools/` vs `scripts/` tie-breaker and the `retired/` semantics unchanged.

## Coordination & sequencing

- **Part A non-harness first** (`policy/`, `infinigen/`, `diagnostics/`): no
  harness dependency — pickable as soon as this brief is.
- **`scripts/harness/` is gated on `harness-architecture`.** It re-moves
  files that brief owns + are Makefile-wired (`sim-bridge`/`-gui`/`-harness`
  → `Makefile` invoke `run_sim_in_the_loop.py`; `capture.py`↔`teleop_capture.py`
  subprocess hand-off). Land it after harness-architecture stabilizes (or
  bundle with one of its tier PRs), updating `harness-architecture.md`'s paths
  in the same PR — same contract the parent brief used for the `capture.py`
  move.
- **Part B (`tools/`) last**, as its own PR — the importer + autonomy-test
  blast radius is the bulk of the risk and is independent of Part A.

## Acceptance criteria

- [ ] `conventions.md`'s placement rule is amended to the sub-system layout with an enumerated sub-system set and a defined default home; the "no `misc/`" guard is restated for `scripts/`/`tools/`.
- [ ] Part A: each `scripts/` group exists; every moved path is updated in the same commit (Makefile, cheatsheet, READMEs, repo-topology, context modules, active/parked briefs); `make sim-bridge`/`-harness` + the training/export/capture smokes run from the new paths.
- [ ] `scripts/harness/` moves are coordinated with `harness-architecture` and its references repointed in the same PR; `capture.py`↔`teleop_capture.py` resolver still resolves.
- [ ] Part B: each `tools/` sub-package has `__init__.py`; **every** importer (incl. `strafer_autonomy/tests/`) is repointed; `make test-dgx` + `make test-harness` stay green.
- [ ] No source file references a pre-move path by a stale string; `tools/check_brief_links.sh` no worse than baseline; Mermaid edge-label sweep clean.

## Out of scope

- **`strafer_ros/`** and the Jetson lane.
- **Re-litigating `scripts/` vs `tools/`** (the parent brief's tie-breaker stands; this brief only adds the second axis).
- **`make` wrappers / invocation unification** — separate concern.
- **Creating the future `eval_*`/`build_*` scripts** — this brief only ensures they have an unambiguous sub-system home when they land.
