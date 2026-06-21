# Organize strafer_lab `tools/` into purpose subpackages

**Type:** refactor (pure-mechanical module moves + import-path updates; no behavior change)
**Owner:** DGX agent
**Priority:** P3 — pure cleanliness; blocks no acceptance bar. The flat `tools/` works today; this makes the package name signal domain instead of being an undifferentiated grab-bag.
**Estimate:** M — mechanical but wide: ~13 module moves + every importer updated (~26 files import `strafer_lab.tools` across `source/` + `Scripts/`). The risk is import churn / merge conflicts, not logic.
**Branch:** `task/tools-package-reorg`

**Blocked on / sequencing:** Filed-on-trigger / parked. This is a wide import-churn change that conflicts easily, so land it only when **no large `tools/`-touching PR is in flight**. Known `tools/` churn ahead of it: [`depth-ffv1-video-column`](../harness/depth-ffv1-video-column.md) edits the LeRobot writer/depth modules and `mission-generator` (PR #98) adds `build_mission_queue.py`; let those settle first.

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

_Inventory re-walked 2026-06-21 against the current tree (post #90/#96/#97/#98/#92); `dataset_export.py` is now **deleted**, not retirement-pending._

| Module | Domain |
|---|---|
| `lerobot_writer.py`, `lerobot_depth.py`, `lerobot_detections.py`, `bbox_extractor.py`, `detections_overlay.py`, `mission_queue.py`, `build_mission_queue.py` (lands with `mission-generator`), `grounding_injection.py` | harness dataset capture |
| `gamepad_reader.py`, `teleop_buttons.py`, `teleop_mission_picker.py` | teleop input / UI |
| `scene_metadata_reader.py`, `scene_connectivity.py`, `scene_paths.py`, `scene_labels.py`, `spatial_description.py` | scene metadata — **CONTRACT-side, scene-source-agnostic** (the readers every consumer uses) |
| `infinigen_label_parser.py` + the `room_struct_regex()` half of `scene_classes.py` | scene metadata — **Infinigen-specific producer** (parses `<Factory>_<id>__spawn_asset_<n>_` / `<room>_<i>_<j>_<class>` prim names); imported only by the producers `extract_scene_metadata` / `generate_scenes_metadata` |
| `scene_classes.py`'s `STRUCTURAL_CLASSES` frozenset | the contract's source-agnostic structural-class denylist (stays contract-side) |
| `phase_profiler.py` | profiling |

The generic name `tools/` invites unrelated modules and obscures domain at the import site (`from strafer_lab.tools.X import ...` tells you nothing about what layer `X` belongs to). The `teleop_` / `scene_` / `lerobot_` filename prefixes are already doing subpackage-naming work by hand — this brief promotes those informal prefixes to real subpackages.

**Boundary requirement (per [`SCENE_PROVIDER_CONTRACT.md`](../../../SCENE_PROVIDER_CONTRACT.md)'s Consumer-obligations section):** the reorg must **not** lump the source-agnostic contract-side readers (`scene_metadata_reader`, `scene_connectivity`, `scene_paths`, `scene_labels`) together under one name with the **Infinigen-specific** parser (`infinigen_label_parser`, the `room_struct_regex` half of `scene_classes`). The litmus "what sits in `tools/scene/` next to the agnostic readers is itself source-agnostic" must hold after the move, so a future non-Infinigen scene source's adapter has an obvious, separate home. Split `scene_classes.py`: `STRUCTURAL_CLASSES` (agnostic) stays with the readers; `room_struct_regex()` moves to the Infinigen sub-home.

## Proposed grouping (implementer finalizes)

The **principle** is purpose-grouping; the exact tree is the implementer's call. A reasonable starting split:

```
strafer_lab/tools/
  harness/        lerobot_writer, lerobot_depth, lerobot_detections, bbox_extractor, detections_overlay,
                  mission_queue, build_mission_queue, grounding_injection
  teleop/         gamepad_reader, teleop_buttons, teleop_mission_picker
  scene/          scene_metadata_reader, scene_connectivity, scene_paths, scene_labels, spatial_description,
                  scene_classes (STRUCTURAL_CLASSES only)        # CONTRACT-side, source-agnostic
  scene/infinigen/  infinigen_label_parser, scene_classes.room_struct_regex   # Infinigen-specific producer
  profiling/      phase_profiler
```

The `scene/` vs `scene/infinigen/` split is the boundary requirement above — it is what keeps "in `tools/scene/` = source-agnostic" a reliable read once a second scene source exists. `dataset_export.py` is gone (deleted), so it does not appear.

## Acceptance

- [ ] Modules moved into purpose subpackages under `tools/`; **pure mechanical move + import-path rewrite — no logic edits, no behavior change.** A diff that touches anything other than module locations + import statements + `__init__.py` re-exports is out of scope.
- [ ] **Every importer updated.** Grep `strafer_lab.tools` across `source/` and `Scripts/` (~26 files) and update each; no lingering old import path resolves only by accident. No back-compat shim / alias `tools/__init__.py` re-export — this is a clean move, callers migrate in-PR (matches the project's clean-break convention).
- [ ] **Keep the contract / Infinigen split (per the boundary requirement above).** Source-agnostic readers (`scene_metadata_reader`, `scene_connectivity`, `scene_paths`, `scene_labels`) land in `tools/scene/`; the Infinigen-specific parser (`infinigen_label_parser`, `scene_classes.room_struct_regex`) lands in a separate Infinigen sub-home. After the move, `grep -rE 'infinigen|Factory|spawn_asset|prim_path|bpy' tools/scene/*.py` (excluding the `infinigen/` sub-dir) returns zero — the agnostic readers carry no Infinigen coupling.
- [ ] **Stale-attribution fix.** `tools/__init__.py` still names the deleted `dataset_export` module — drop it in this PR. (`dataset_export.py` was deleted by the harness-retirement work; it is not re-homed because it no longer exists.)
- [ ] Tests green: `make test-lab-pure` (the pure-Python harness suite) **and** the env-cfg construction path (env cfgs import the writer indirectly). Confirm no test still imports an old `strafer_lab.tools.X` path.
- [ ] Its own PR; no functional change bundled. The PR description states "mechanical reorg, no behavior change" so review is a move-and-import audit, not a logic review.
- [ ] If any context module, package README, or guide under `docs/` names a `strafer_lab.tools.X` path, update it in the same PR per [`conventions.md`](../../context/conventions.md)'s user-facing-documentation-maintenance rule.
- [ ] Brief shipped to [`completed/`](../../completed/) per [`conventions.md`](../../context/conventions.md) inside the shipping PR (stamp with the work commit + PR link).

## Out of scope

- **Logic / behavior changes** of any moved module. Move-only.
- **Cross-package moves** (e.g. relocating a module out of `strafer_lab` into `strafer_shared` or `strafer_perception`) — this brief reorganizes *within* `strafer_lab/tools/` only, respecting the DGX/Jetson ownership boundary. The `scene/infinigen/` split is *within* `tools/`; whether the Infinigen producer code eventually moves out to live alongside the `scripts/infinigen/` group ([`script-tool-subsystem-grouping`](../../active/tooling/script-tool-subsystem-grouping.md)) is that brief's call — here it just stops sitting next to the agnostic readers.
