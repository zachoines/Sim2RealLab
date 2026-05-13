# Align the harness output schema with LeRobot v2 / Isaac Lab `RecorderManager`

**Type:** investigation + refactor (filed-on-trigger)
**Owner:** DGX agent
**Priority:** P2 (filed-on-trigger; pick up when the first
downstream training brief — `vla-v2-architecture` or a
GR00T / OpenVLA fine-tune brief — needs to consume the harness
corpus, OR when the harness is about to ship the first 1k+
trajectories and switching format later will mean a re-export)
**Estimate:** M (~half-week; format-decision investigation +
recorder-manager wiring for in-process drivers + exporter
refactor for the bridge driver + back-compat shim)
**Branch:** task/harness-output-format-alignment

## Story

As a **DGX operator who needs the harness corpus to be
consumable by published wheeled-VLA / VLM training pipelines
(GR00T, OpenVLA, π0, Octo) without writing a custom converter
per consumer**, I want **the harness's canonical on-disk schema
to be either LeRobot v2 (parquet + MP4 + `meta/modality.json`)
or Isaac Lab `RecorderManager` HDF5 (robomimic-compatible), with
the existing JSONL layout demoted to a legacy debug
side-channel**, so that **every downstream training brief loads
the corpus with a stock HF / robomimic / Octo loader instead of
its own glue code, and the harness epic stops paying the cost of
maintaining a strafer-specific format the rest of the ecosystem
doesn't speak**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md)

Parent design context:
[`MISSION_VALIDATION_ARCHITECTURE.md` §3.6](../../../MISSION_VALIDATION_ARCHITECTURE.md#36-data-path-options-for-vla-training-the-ones-that-make-33-affordable) —
the four data-collection regimes whose output this brief unifies
under one canonical format.

Sibling briefs:
- [`behavior-cloning-data-expansion`](../../active/harness/behavior-cloning-data-expansion.md) —
  currently specifies a strafer-specific JSONL schema. This
  brief proposes the alignment that will replace its strict-tier
  schema (or be added as an additional output format the brief
  must emit).
- [`teleop-driver`](../../active/harness/teleop-driver.md),
  [`mission-generator`](../../active/harness/mission-generator.md),
  [`trajectory-first-captioning`](../../active/harness/trajectory-first-captioning.md),
  [`oracle-driver`](oracle-driver.md) — all consume the canonical
  schema. Whatever this brief lands on becomes their target.
- [`vla-v2-architecture`](../experimental/vla-v2-architecture.md) —
  the first consumer that exercises the format end-to-end.

## Trigger condition — when to pick this brief up

Pick up when **either** of:

1. The first downstream training brief (`vla-v2-architecture`,
   GR00T fine-tune, OpenVLA fine-tune, or a follow-up
   `mvp-teacher-vla-distillation`) is about to start consuming
   the harness corpus and the operator needs a stable format
   contract.
2. The harness is about to ship the first ≥ 1 k trajectories.
   Re-exporting a small corpus is cheap; re-exporting a large
   one is multi-hour and risks divergence between exports.

Until one fires, this brief stays parked. The audit logged it
because the format decision is **sticky** — every consumer-side
converter written against the strafer-specific JSONL is a
maintenance liability later, so the operator should decide once,
deliberately.

## Context

### Why this brief exists

The current
[`behavior-cloning-data-expansion`](../../active/harness/behavior-cloning-data-expansion.md)
specifies a strafer-specific layout:

```
data/sim_in_the_loop/<scene>/episode_NNNN/
├── frames_tick.jsonl
├── actions.jsonl
├── mission.json
├── frame_*.jpg
├── frame_*.depth.npy
└── progress.jsonl
```

None of the published wheeled-VLA / VLM training pipelines
consume this format. Each pipeline ships its own canonical
loader:

| Pipeline | Canonical format | Loader |
|---|---|---|
| **GR00T (N1.5 / N1.7)** | LeRobot v2 with `meta/modality.json` | `nvidia/isaac-gr00t` + HF `lerobot` |
| **OpenVLA / OpenVLA-OFT** | RLDS (TFDS tfrecord) | `prismatic-vla` + the OXE loader chain |
| **π0 / openpi** | LeRobot v2 | `physical-intelligence/openpi` |
| **Octo** | RLDS | Octo's data-loader |
| **Isaac Lab Mimic / robomimic** | HDF5 (robomimic schema) | Isaac Lab's `record_demos.py` + `train.py` |
| **NaVid / Uni-NaVid** | VLN-CE per-step samples (custom) | per-paper loader |

If we keep the JSONL format, every downstream brief writes a
converter from JSONL → its preferred format. Converters drift.
The harness brief already gates a `mimic-hdf5` + `rlds` exporter
at recommended tier; this brief proposes flipping the default —
make one of those formats the source of truth, demote JSONL to a
debug side-channel.

### The format-choice question

Two paths, both viable; this brief's first job is to pick one
deliberately rather than inherit JSONL by accident.

**Path A — LeRobot v2 as canonical.**

- Pros:
  - GR00T (NVIDIA's wheeled / humanoid foundation model) and π0
    (Physical Intelligence's flow-matching VLA) consume LeRobot
    v2 natively, with `meta/modality.json` specifying the state
    / action / video layout per embodiment.
  - HF `lerobot` library + plugin pipeline lets us drop into
    GR00T or any LeRobot-compatible policy as a finetune target
    with no glue code.
  - Parquet + MP4 is dense (way smaller than JSONL + JPEG +
    `.depth.npy`); MP4 H.264 at high CRF preserves quality
    better than per-frame JPEGs.
  - Versioned schema with a working v3 conversion script —
    long-term ecosystem support.
- Cons:
  - MP4 encoding adds per-frame cost (~5–10 ms on the DGX);
    real-time capture during teleop may need a deferred encode
    pass.
  - Random per-frame access is slower than per-file JPEG. For
    in-training shuffled batches this is fine (LeRobot's loader
    handles it), but for ad-hoc inspection it's worse.
  - Adds a `meta/modality.json` authoring step per scene /
    embodiment.

**Path B — Isaac Lab `RecorderManager` HDF5 (robomimic schema)
as canonical.**

- Pros:
  - Already exists in Isaac Lab —
    `isaaclab.envs.mdp.recorders.recorders.{InitialStateRecorder,
    PreStepActionsRecorder, PostStepStatesRecorder,
    PreStepFlatPolicyObservationsRecorder}` plus
    `RecorderManager` + `HDF5DatasetFileHandler` — and is hooked
    into the existing `record_demos.py` workflow. Less code to
    write.
  - `DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES`
    maps directly to teleop's outcome-tagged buttons
    (succeeded / failed / wrong_instance / wrong_room /
    trajectory_violation), giving us hard-negative file
    separation for free.
  - Compatible with `scripts/imitation_learning/robomimic/
    train.py` in Isaac Lab — we can train a baseline imitation
    policy with NO additional glue.
  - HDF5 is well-supported across the ecosystem and is what
    Isaac Lab Mimic itself emits.
- Cons:
  - GR00T / π0 / OpenVLA all consume LeRobot v2 or RLDS, not
    robomimic HDF5. We'd still need a converter at training
    time.
  - HDF5 is single-file-per-dataset; concurrent writes from
    parallel envs need an external coordinator.
  - The schema is manipulation-flavored — `obs`, `actions`,
    `states` — wheeled SE(2) navigation fits awkwardly.

**Path C — Both, with one canonical and the other auto-exported.**

Pick a canonical (Path A or Path B) but commit to a regenerable
auto-export to the other in `export_bc_dataset.py`. This avoids
locking out either ecosystem. Increases complexity but is the
most pragmatic option if no consumer dominates at pickup time.

### Specific architectural decisions to make

1. **Canonical format.** Path A vs. B vs. C above.
2. **Compose with Isaac Lab `RecorderManager`?** The in-process
   drivers (teleop, oracle, trajectory-first) all run inside an
   Isaac Lab env; they can reuse `RecorderManager` directly,
   avoiding the per-tick writer wheel we'd otherwise reinvent.
   The bridge driver runs cross-host (Jetson → bridge); it
   can't reuse `RecorderManager` and needs its own recorder
   (the existing `perception_writer.py` path). **Architectural
   split**: in-process drivers use `RecorderManager`; bridge
   driver uses the custom recorder; both emit the canonical
   format on disk so consumers see a single contract.
3. **`meta/modality.json` schema for strafer** (if Path A or C).
   Draft:
   ```json
   {
     "state": {
       "robot_pose": {"absolute": true, "rotation_type": "quaternion",
                      "indices": [0, 1, 2, 3, 4, 5, 6]}
     },
     "action": {
       "cmd_vel": {"absolute": false, "indices": [0, 1, 2],
                   "components": ["vx", "vy", "omega_z"]}
     },
     "video": {
       "perception_cam": {"original_key": "observation.images.d555",
                          "fps": 8},
       "policy_cam": {"original_key": "observation.images.d555_policy",
                      "fps": 8}
     },
     "annotation": {
       "mission_text": "annotation.tasks.mission_text"
     }
   }
   ```
4. **Multi-camera handling.** The
   `StraferSceneCfg_InfinigenPerception` scene ships two
   cameras (80×60 policy + 640×360 perception). The format
   supports multi-camera; the brief should decide whether v1
   emits one or both.
5. **Action chunk encoding.** Per the harness brief's audit
   update, action chunks default to 30 ticks (1 s at 30 Hz).
   Document how chunks are encoded in the canonical schema:
   LeRobot v2 stores per-tick actions and lets the loader
   build chunks; Mimic HDF5 stores per-tick `actions` arrays.
   Either way, chunk size goes in metadata, not in the action
   array.
6. **Cross-host bridge driver path.** The bridge driver runs
   on the DGX but the action source (`/cmd_vel`) is published
   from the Jetson over ROS 2. `RecorderManager` cannot
   represent this without a shim. Two options:
   (a) write a custom `RecorderTerm` that pulls
   `/cmd_vel` from the bridge graph each tick and hands it to
   the recorder; (b) keep the current
   `perception_writer.py` path for bridge mode and emit
   LeRobot v2 / HDF5 from the post-hoc converter. Decide at
   investigation time.

## Acceptance criteria

- [ ] **Format decision documented** in the brief PR description
      with explicit rationale: Path A / B / C chosen, why,
      and which downstream consumer drove the choice.
- [ ] **Canonical schema written out** as a short doc under
      `source/strafer_lab/strafer_lab/tools/` or
      `docs/HARNESS_OUTPUT_FORMAT.md`, including the
      `meta/modality.json` content if LeRobot v2 is picked.
- [ ] **In-process driver wiring.** At least the teleop driver
      (or whatever in-process driver is shipped at pickup time)
      writes the canonical format end-to-end via
      `RecorderManager` (Path B) or `lerobot.LeRobotDataset`
      append (Path A). Smoke run produces one episode in the
      target format; a stock loader (robomimic `dataset.load`
      or HF `LeRobotDataset.from_preloaded`) round-trips it.
- [ ] **Bridge driver path.** Either
      [`behavior-cloning-data-expansion`](../../active/harness/behavior-cloning-data-expansion.md)'s
      bridge mode is migrated to emit canonical format
      directly, or `export_bc_dataset.py` is wired as the
      bridge-mode post-processor with a smoke round-trip.
- [ ] **Legacy JSONL kept as side-channel.** Existing consumers
      (`generate_descriptions.py`, `prepare_vlm_finetune_data.py`,
      `dataset_export.py`) continue to work. The legacy file
      either continues to be written, OR a one-shot converter
      from the canonical format → legacy JSONL is provided.
- [ ] **Multi-camera handling decided** and documented; tests
      cover the chosen layout.
- [ ] **Doc surface.** Updates to
      [`source/strafer_lab/README.md`](../../../../source/strafer_lab/README.md)
      "Contracts" and
      [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../../INTEGRATION_SIM_IN_THE_LOOP.md)
      Stage 5/6 reflecting the new format.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- Isaac Lab `RecorderManager` API:
  [`source/isaaclab/isaaclab/managers/recorder_manager.py`](../../../../../IsaacLab/source/isaaclab/isaaclab/managers/recorder_manager.py)
  in the workspace IsaacLab tree.
- Isaac Lab `record_demos.py`:
  [`scripts/tools/record_demos.py`](../../../../../IsaacLab/scripts/tools/record_demos.py).
  Already wires gamepad / keyboard / spacemouse → recorder.
- HDF5 file handler:
  [`source/isaaclab/isaaclab/utils/datasets/hdf5_dataset_file_handler.py`](../../../../../IsaacLab/source/isaaclab/isaaclab/utils/datasets/hdf5_dataset_file_handler.py).
- LeRobot v2 spec:
  [https://huggingface.co/docs/lerobot/lerobot-dataset-v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
  (v3 is current; v2 conversion script noted in the docs).
- GR00T modality.json examples:
  [https://github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
  has worked examples under `getting_started/`.
- π0 / openpi loader:
  [https://github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
  for the LeRobot v2 → action-chunk path.

## Out of scope

- **Migrating completed perception datasets** (the per-skill
  `frames.jsonl` corpus the description / VLM SFT pipeline
  already consumes). That format stays as-is; this brief is
  about the *behavior-cloning* / per-tick / action-stream
  corpus.
- **Choosing a final VLA architecture.** That's
  [`vla-v2-architecture`](../experimental/vla-v2-architecture.md)'s
  decision; this brief only locks the data contract.
- **Real-robot data capture format.** Sim-side only. Real-robot
  capture is a future hardware change; the format choice here
  should be informed by what works on the real robot but not
  blocked by it.
- **Solving the cross-host action-source problem for the bridge
  driver in generality.** Either find a recorder solution or
  ship the post-hoc converter; don't redesign the bridge.
