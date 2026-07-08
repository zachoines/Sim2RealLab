# Tools and scripts map

A map of the project's importable **tools** and runnable **scripts** so a
fresh agent knows what already exists before writing a new one, and knows
which modules a behavior change must revisit. This is an **index**: the
authoritative per-tool detail (signatures, args, flags) lives in each
package's README inventory — linked below. The one-line purposes here are
the durable part; when they disagree with the code, the code wins.

For *where a new file belongs* (tool vs script, `scripts/` vs
`tools/`, the `retired/` holding area), see
[`conventions.md`'s Script and tool placement](conventions.md#script-and-tool-placement).
For the entry-point scripts at a glance, [`repo-topology.md`'s Key
entry-point scripts](repo-topology.md#key-entry-point-scripts).

## Authoritative inventories (source of truth)

| Package | Inventory |
|---|---|
| `strafer_lab` | [README → Scripts and tools](../../../source/strafer_lab/README.md) |
| `strafer_autonomy` | [README → Skill registry / Client protocols / CLI](../../../source/strafer_autonomy/README.md) |
| `strafer_vlm` | [README → HTTP endpoints / Operator CLI tools](../../../source/strafer_vlm/README.md) |
| `strafer_ros` | [README → launch / hardware-validation scripts](../../../source/strafer_ros/README.md) |
| `strafer_shared` | [README → shared constants](../../../source/strafer_shared/README.md) |

## `strafer_lab/strafer_lab/tools/` (imported)

**Scene metadata & geometry**
- `scene_metadata_reader` — the single `pxr` touch-point that reads a scene's metadata from USD `customData` (hard-fails when absent); `write_custom_data`, `metadata_hash`.
- `scene_classes` — the shared structural-class set (`floor/ceiling/wall/exterior/staircase`) + room-struct regex; consumed by the label denylist and the spawn-point sampler.
- `scene_labels` — typed accessors over a scene's metadata dict (pure-data `iter_rooms`/`iter_objects`/`get_*_from_data`, plus `pxr`-bound `get_scene_metadata`).
- `scene_paths` — derive which scene USD to read from a `--scene` name / `--scene-usd` override.
- `infinigen_label_parser` — Infinigen factory prim-name → object label / instance id.
- `spatial_description` — Stage-1 factual spatial relations (room, distance, bearing) from a frame + metadata dict.
- `grounding_injection` — hard-negative goal perturbation (`--inject-bad-grounding`) over a scene's `objects[]`.

**LeRobot dataset I/O**
- `lerobot_writer` — the LeRobot v3 dataset writer + per-episode strafer extension parquet; `hash_scene_metadata`.
- `lerobot_depth` — 16UC1 depth PNG sidecar read/write at deterministic paths.
- `lerobot_detections` — the padded `observation.detections.*` columns + `meta/detection_labels.json` vocab (`pack_detections`, `read_detection_labels`).
- `detections_overlay` — render a capture's recorded detections as an annotated MP4 (pure read-side viz; works on any detections-bearing run).

**Perception**
- `bbox_extractor` — wraps the Replicator `bounding_box_2d_tight` annotator → typed `DetectedBbox`; `parse_bbox_data` is pure (no Isaac Sim).

**Teleop & missions**
- `teleop_mission_picker` — scene `objects[]` → filtered/sorted `MissionCandidate` list + console picker.
- `teleop_buttons` — gamepad button state → episode outcome.
- `gamepad_reader` — gamepad input abstraction.
- `mission_queue` — load/validate `mission_queue.yaml`.

**Profiling**
- `phase_profiler` — per-phase timing for the training/capture loops.

`tools/retired/` holds deprecated-but-mined code (e.g. `dataset_export`): not imported by any live entry point; emptied by the owning brief on ship.

## `strafer_lab/scripts/` (run directly via `$ISAACLAB -p` / `python -m` / a `make` target)

- **Training / eval / export:** `train_strafer_navigation`, `play_strafer_navigation`, `export_policy`, `benchmark_policy`, `test_strafer_env`, `collect_demos`.
- **Scene generation:** `prep_room_usds` (generate/ingest; chains the next two), `postprocess_scene_usd` (colliders + lights), `extract_scene_metadata` (embed metadata + `UsdSemantics` labels into the USD), `generate_scenes_metadata` (combined spawn-point manifest).
- **Capture / harness:** `capture` (the `--driver` × `--mission-source` entry point), `teleop_capture`, `run_sim_in_the_loop` (bridge + harness modes), `bridge_harness_smoke` (Jetson-free Kit smoke).
- **Diagnostics:** `roller_bounce_probe`, `probe_rt2_depth_integrity` (RTX Real-Time 2.0 depth-integrity gate; pure analysis in `tools/depth_probe`).
- `scripts/asset_authoring/` — run-by-hand robot/asset USD utilities: `collapse_redundant_xforms`, `inspect_robot_prim_layout`, `run_empty_lab`, `setup_physics`.
- `scripts/retired/` — deprecated data-prep: `finetune_clip`, `generate_descriptions`, `prepare_vlm_finetune_data`.

## Other packages

- `strafer_autonomy` — the executor's skills, planner/VLM/ROS client protocols, and the `strafer-autonomy-cli` / `strafer-executor` console entry points; see its README.
- `strafer_vlm` — the grounding/description/detection HTTP service + operator CLI tools; see its README.
- `strafer_ros` — ROS 2 launch surface + hardware-validation scripts; see its README.
- `strafer_shared` — `constants.py` (chassis/sensor/topic specs) + `policy_interface`; append-only across the lane boundary.

## Maintenance rule

When you change a tool/script's **public API** (a function signature,
a CLI flag, an output schema):

1. Update its row in the owning package's README inventory (the source
   of truth) in the same PR.
2. Re-check every consumer — `rg` the symbol/flag across `source/` and
   `docs/` — and migrate them in the same PR (the clean-break convention).
3. If the change adds/removes/renames a tool or script, update the
   relevant one-liner here.

This map carries only names + purposes; it does not need a row-edit for
an arg-only change (that's the README's job) — but a *new* or *removed*
tool belongs here so the next agent sees it.
