# Stabilize RTAB-Map cold-start after `make kill` against a populated DB

**Type:** task / bug
**Owner:** Jetson agent (`source/strafer_ros/strafer_slam/`)
**Priority:** P2
**Estimate:** M (~1–2 days; reproduce signature, bisect between bridge-side teleport vs RTAB-Map memory config)
**Branch:** task/rtabmap-cold-start-determinism

## Story

As a **mission operator restarting bringup after `make kill`** against an existing `~/.ros/rtabmap.db`, I want **RTAB-Map to either continue the previous map cleanly OR explicitly declare that this is a fresh-map workflow with the right docs and config**, so that **my second session doesn't burst hundreds of `Not found word` errors, increment to a new map id, and silently rebuild the occupancy grid as if no prior map existed**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)

## Context

**Symptom signature** captured on the operator's 2026-05-10 09:44 launch (sim-in-the-loop bringup against a populated `~/.ros/rtabmap.db` from a prior session that ended via `make kill`):

```
[rtabmap-4] [ERROR] VWDictionary.cpp:741::addWordRef() Not found word 101854 (dict size=31735)
[rtabmap-4] [ERROR] VWDictionary.cpp:741::addWordRef() Not found word 101855 (dict size=31735)
... (many more) ...
[rtabmap-4] [ WARN] Rtabmap.cpp:1441::process() Odometry is reset (identity pose detected). Increment map id to 4!
[rtabmap-4] [ WARN] Memory.cpp:5917::createSignature() Mem/SaveDepth16Format is set to false to use 32bits format but this is not compatible with the compressed depth format chosen (Mem/DepthCompressionFormat=".rvl"), depth images will be compressed in ".png" format instead.
[rtabmap-4] [ WARN] Rtabmap.cpp:3069::process() Rejected loop closure 556 -> 678: Not enough inliers 0/15 (matches=37)
```

Foxglove confirmed a fresh occupancy grid getting built — the "Increment map id" branch did trigger a new disconnected map within the same DB.

**Three suspected root causes, ordered by likelihood:**

1. **Bridge teleports `/strafer/odom` back to identity on every launch.** RTAB-Map's `Odom is reset (identity pose detected)` log is its standard response to an odom message with pose ≈ (0,0,0) when the previous session's last node was at some other pose. Sim teleports the robot to spawn on every launch, so this will fire on every cold start. The "Increment map id" branch is RTAB-Map behaving correctly given that signal — but the operator-visible effect is "my map disappeared."

2. **Stale Working Memory references vs Short-Term Memory pruning.** The `Not found word N (dict size=31735)` burst prints as nodes are loaded from the DB into Working Memory. References inside those nodes point at visual words that have been pruned from the dictionary (e.g., via aggressive `Mem/STMSize=30`). Each missing word is non-fatal but the burst is noisy and may degrade descriptor matching on the first frames.

3. **Depth-compression format mismatch.** `Mem/SaveDepth16Format=false` + `Mem/DepthCompressionFormat=.rvl` is incompatible. RTAB-Map silently falls back to `.png` and emits a one-shot warning. Config cleanup; not a behavioral root cause.

## Approach

Triage in order:

### A. Confirm cause (1) — is the teleport structural?

Capture two consecutive sim sessions (kill between them) with `ros2 topic echo /strafer/odom --once` at the start of each. If both report identity pose, the increment-map-id is fundamentally a consequence of the bridge's spawn behavior. Decide between:
- **A1.** Add an auto-relocalization step at RTAB-Map startup — run loop-closure detection against the existing DB before accepting an identity-pose reset as a new map id.
- **A2.** Run RTAB-Map in localization mode (`localization:=true`) by default on the sim lane after the first session has produced a usable DB. Document the workflow: `make launch-sim` first time → mapping; subsequent launches → localization unless `make clean-map` is run.
- **A3.** Accept the new-map-id behavior, document it as expected for `make kill && make launch-sim`, and update the operator-facing runbooks accordingly.

### B. Tame cause (2) — visual word dictionary noise

Inspect [`config/rtabmap_params.yaml`](../../../../source/strafer_ros/strafer_slam/config/rtabmap_params.yaml) memory settings. `Mem/STMSize: "30"` is aggressive. Candidates:
- Raise `Mem/STMSize` (e.g., 60) so words live longer.
- Set `Mem/InitWMWithAllNodes: "true"` so all nodes preload into WM on start and word references resolve cleanly.

### C. Fix cause (3) — depth compression alignment

Set `Mem/SaveDepth16Format: "true"` OR `Mem/DepthCompressionFormat: ".png"` to silence the warning. Our depth from sim is 32-bit; `.png` is the safer pick (no 65 m truncation risk).

## Acceptance criteria

- [ ] After `make kill && make launch-sim` against an existing DB, the first 60 s of `/rosout` contains zero `addWordRef() Not found word` errors — or the burst is reduced to a documented baseline with a one-line PR rationale.
- [ ] The `Mem/DepthCompressionFormat` warning no longer prints.
- [ ] If cause (1) is confirmed structural, the brief documents the chosen disposition (A1/A2/A3) and ships either the auto-relocalization path OR an updated runbook explaining the new-map-id workflow.
- [ ] No regression on cold-start without a DB (`make clean-map && make launch-sim`): RTAB-Map starts fresh, builds a usable map, executor missions complete.
- [ ] Real-robot bringup unaffected — RTAB-Map state should still persist across power cycles as it does today. Gate any config that risks regressing real-robot if the change is sim-specific.
- [ ] Unit tests cover whichever knobs touched (param presence in `rtabmap_params.yaml`).
- [ ] If your work invalidates a fact in any referenced context module, package README, top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance) for the surface list and trigger heuristics.

## Investigation pointers

- [`source/strafer_ros/strafer_slam/config/rtabmap_params.yaml`](../../../../source/strafer_ros/strafer_slam/config/rtabmap_params.yaml) — Memory and Grid settings.
- [`source/strafer_ros/strafer_slam/launch/slam.launch.py`](../../../../source/strafer_ros/strafer_slam/launch/slam.launch.py) — the launch invocation, including the `localization:=` argument that's currently unused on the sim lane.
- The "Increment map id" log line in `Rtabmap.cpp:1441` is RTAB-Map's standard response to identity-pose-after-existing-graph; if the bridge teleports, that branch fires every session.
- RTAB-Map reference: [Parameters](https://rtabmap.github.io/Parameters.html), `Mem/*` and `RGBD/*` knobs.

## Out of scope

- **Sim bridge teleport behavior.** That's a DGX-lane concern (`source/strafer_lab/`). If we conclude the bridge spawn is the structural root cause, file a follow-up DGX brief — do not piggy-back changes to the bridge here.
- **Real-robot RTAB-Map state determinism.** Real-robot doesn't teleport the chassis across power cycles, so cause (1) doesn't apply. If real-robot regressions surface as a side effect of (2)/(3) config changes, file a third brief; don't expand this one's scope.
