# Organize strafer_lab `tools/` into purpose subpackages

**Type:** refactor (pure-mechanical module moves + import-path updates; no behavior change)
**Owner:** DGX agent
**Priority:** P3 — pure cleanliness; blocks no acceptance bar. The flat `tools/` works today; this makes the package name signal domain instead of being an undifferentiated grab-bag.
**Estimate:** M — mechanical but wide: ~13 module moves + every importer updated (~26 files import `strafer_lab.tools` across `source/` + `Scripts/`). The risk is import churn / merge conflicts, not logic.
**Branch:** `task/tools-package-reorg`

**Blocked on / sequencing:** Filed-on-trigger / parked. This is a wide import-churn change that conflicts easily, so land it only when **no large `tools/`-touching PR is in flight**. Known `tools/` churn ahead of it: the R1 first-class `observation.detections.*` harness column and [`depth-ffv1-video-column`](../harness/depth-ffv1-video-column.md) both edit the LeRobot writer/depth modules; let those settle first. Do **not** interleave this with the harness-retirement PR that deletes `dataset_export.py` (see Acceptance).

## Story

As a **developer navigating `strafer_lab`** I want **`tools/` split into purpose-named subpackages (harness writers vs teleop input vs scene metadata vs profiling)** so that **the import path signals what a module is for, and the package stops accreting unrelated domain-specific helpers behind one generic name.**

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`script-tool-subsystem-grouping`](../../active/tooling/script-tool-subsystem-grouping.md) — writes the shared `## Script and tool placement` rule amendment (flat → sub-system) that this brief's `tools/` move conforms to; that brief owns the parallel `scripts/` reorg. Land this `tools/` move under that rule.

## Motivation

`source/strafer_lab/strafer_lab/tools/` has grown to ~13 modules spanning at least four unrelated domains:

| Module | Domain |
|---|---|
| `lerobot_writer.py`, `lerobot_depth.py`, `perception_writer.py`, `bbox_extractor.py` | harness dataset capture |
| `gamepad_reader.py`, `teleop_buttons.py`, `teleop_mission_picker.py` | teleop input / UI |
| `scene_labels.py`, `scene_paths.py`, `infinigen_label_parser.py`, `spatial_description.py` | scene metadata |
| `phase_profiler.py` | profiling |
| `dataset_export.py` | **retirement-pending** (on the harness-architecture retirement list) |

The generic name `tools/` invites unrelated modules and obscures domain at the import site (`from strafer_lab.tools.X import ...` tells you nothing about what layer `X` belongs to). The `teleop_` / `scene_` / `lerobot_` filename prefixes are already doing subpackage-naming work by hand — this brief promotes those informal prefixes to real subpackages.

## Proposed grouping (implementer finalizes)

The **principle** is purpose-grouping; the exact tree is the implementer's call. A reasonable starting split:

```
strafer_lab/tools/
  harness/     lerobot_writer, lerobot_depth, perception_writer, bbox_extractor
  teleop/      gamepad_reader, teleop_buttons, teleop_mission_picker
  scene/       scene_labels, scene_paths, infinigen_label_parser, spatial_description
  profiling/   phase_profiler
```

`dataset_export.py` is deliberately **not** placed — see Acceptance.

## Acceptance

- [ ] Modules moved into purpose subpackages under `tools/`; **pure mechanical move + import-path rewrite — no logic edits, no behavior change.** A diff that touches anything other than module locations + import statements + `__init__.py` re-exports is out of scope.
- [ ] **Every importer updated.** Grep `strafer_lab.tools` across `source/` and `Scripts/` (~26 files) and update each; no lingering old import path resolves only by accident. No back-compat shim / alias `tools/__init__.py` re-export — this is a clean move, callers migrate in-PR (matches the project's clean-break convention).
- [ ] **`dataset_export.py` is NOT re-homed.** It is on the harness-architecture retirement list — giving a doomed module a new home is wasted churn and a merge-conflict magnet against the retirement PR. Leave it at `tools/` root with a one-line "retirement-pending" note, or coordinate its deletion into this PR if the harness-retirement work has already landed. Do not move it.
- [ ] Tests green: `make test-lab-pure` (the pure-Python harness suite) **and** the env-cfg construction path (env cfgs import the writer indirectly). Confirm no test still imports an old `strafer_lab.tools.X` path.
- [ ] Its own PR; no functional change bundled. The PR description states "mechanical reorg, no behavior change" so review is a move-and-import audit, not a logic review.
- [ ] If any context module, package README, or guide under `docs/` names a `strafer_lab.tools.X` path, update it in the same PR per [`conventions.md`](../../context/conventions.md)'s user-facing-documentation-maintenance rule.
- [ ] Brief shipped to [`completed/`](../../completed/) per [`conventions.md`](../../context/conventions.md) inside the shipping PR (stamp with the work commit + PR link).

## Out of scope

- **Logic / behavior changes** of any moved module. Move-only.
- **`dataset_export.py` deletion** — that's the harness-retirement PR's job; this brief only avoids re-homing it.
- **Cross-package moves** (e.g. relocating a module out of `strafer_lab` into `strafer_shared` or `strafer_perception`) — this brief reorganizes *within* `strafer_lab/tools/` only, respecting the DGX/Jetson ownership boundary.
