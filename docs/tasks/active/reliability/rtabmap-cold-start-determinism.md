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

1. **Bridge teleports `/strafer/odom` back to identity on every launch.** RTAB-Map's `Odom is reset (identity pose detected)` log is its standard response to an odom message with pose ≈ (0,0,0) when the previous session's last node was at some other pose. Sim teleports the robot to spawn on every launch, so this will fire on every cold start. The "Increment map id" branch is RTAB-Map behaving correctly given that signal — but the operator-visible effect is "my map disappeared." This branch is well-documented upstream as [introlab/rtabmap_ros#80](https://github.com/introlab/rtabmap_ros/issues/80) ("when odometry recovers from being lost, RTAB-Map cannot know there is a discontinuity — it only detects if odometry is reset to Identity, creating a new map"). The fix is either (a) prevent the identity-reset signal at the source, or (b) tell RTAB-Map this is the same map by entering localization mode.

2. **Stale Working Memory references vs Short-Term Memory pruning.** The `Not found word N (dict size=31735)` burst prints as nodes are loaded from the DB into Working Memory. References inside those nodes point at visual words that have been pruned from the dictionary (e.g., via aggressive `Mem/STMSize=30`). Each missing word is non-fatal but the burst is noisy and may degrade descriptor matching on the first frames.

3. **Depth-compression format mismatch.** `Mem/SaveDepth16Format=false` + `Mem/DepthCompressionFormat=.rvl` is incompatible. RTAB-Map silently falls back to `.png` and emits a one-shot warning. Config cleanup; not a behavioral root cause.

**Launch-default context that confirms cause (1) is structural:**
[`bringup_sim_in_the_loop.launch.py:62, 206-208`](../../../../source/strafer_ros/strafer_bringup/launch/bringup_sim_in_the_loop.launch.py)
defaults `localization:=false`. So every `make launch-sim` against an
existing DB *starts in mapping mode*, which is exactly the "identity-
pose teleport interpreted as discontinuity → increment map id" path.
The fix isn't ambiguous: it's disposition **A2** below — flip the
launch default so a populated DB triggers localization mode
automatically.

## Approach

Triage in order:

### A. Confirm cause (1) — is the teleport structural?

Capture two consecutive sim sessions (kill between them) with `ros2 topic echo /strafer/odom --once` at the start of each. If both report identity pose, the increment-map-id is fundamentally a consequence of the bridge's spawn behavior. Decide between:
- **A1.** Add an auto-relocalization step at RTAB-Map startup — run loop-closure detection against the existing DB before accepting an identity-pose reset as a new map id. Possible via `Rtabmap/StartNewMapOnLoopClosure=true` + manual seeding, but RTAB-Map's natural startup pose handling doesn't quite express this — non-trivial integration work.
- **A2.** *(Recommended)* Default `localization:=true` in `bringup_sim_in_the_loop.launch.py` *when a populated DB exists at `database_path`*, falling back to mapping mode only when the DB is missing or `rtabmap_args:=-d` is passed. Document the workflow: first `make launch-sim` against an empty DB → mapping; subsequent launches → localization unless the operator explicitly runs `make clean-map`. This matches the RTAB-Map upstream recommendation for "robots after a first map is created" (see [Official RTAB-Map Forum on localization mode best practices](http://official-rtab-map-forum.206.s1.nabble.com/Memory-management-in-localization-mode-td9886.html)). The same launch path already wires `Mem/IncrementalMemory=false` + `Mem/InitWMWithAllNodes=true` when `localization:=true` ([`slam.launch.py:53-55`](../../../../source/strafer_ros/strafer_slam/launch/slam.launch.py)), so the structural change is one block of launch-time logic, not a config rewrite.
- **A3.** Accept the new-map-id behavior, document it as expected for `make kill && make launch-sim`, and update the operator-facing runbooks accordingly. Rejected as the primary path — the operator-visible "my map disappeared" experience is bad UX even with a runbook.

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
- [`source/strafer_ros/strafer_slam/launch/slam.launch.py:36, 53-55`](../../../../source/strafer_ros/strafer_slam/launch/slam.launch.py) — the `localization:=` argument plumbed in but defaulting to `false`. The localization-mode body already sets `Mem/IncrementalMemory=false` and `Mem/InitWMWithAllNodes=true`; A2's work is one DB-existence check, not a config rewrite.
- [`source/strafer_ros/strafer_bringup/launch/bringup_sim_in_the_loop.launch.py:62, 206-208`](../../../../source/strafer_ros/strafer_bringup/launch/bringup_sim_in_the_loop.launch.py) — the bringup-level `localization` arg, also defaulting to `false`. A2 flips this default.
- The "Increment map id" log line in `Rtabmap.cpp:1441` is RTAB-Map's standard response to identity-pose-after-existing-graph; if the bridge teleports, that branch fires every session.
- Cross-reference: [`executor-slam-tracking-precheck-mid-mission.md`](executor-slam-tracking-precheck-mid-mission.md) — if A2 lands, the precheck mostly catches kidnapped-robot / featureless-corridor cases; if A2 lands *after* the precheck brief, the precheck *is* the cold-start workaround.
- RTAB-Map reference: [Parameters](https://rtabmap.github.io/Parameters.html), `Mem/*` and `RGBD/*` knobs. Upstream issue documenting the identity-reset → new-map branch: [introlab/rtabmap_ros#80](https://github.com/introlab/rtabmap_ros/issues/80).
- Best-practices reference for localization-mode workflow: [Official RTAB-Map Forum thread](http://official-rtab-map-forum.206.s1.nabble.com/Memory-management-in-localization-mode-td9886.html).

## Out of scope

- **Sim bridge teleport behavior.** That's a DGX-lane concern (`source/strafer_lab/`). If we conclude the bridge spawn is the structural root cause, file a follow-up DGX brief — do not piggy-back changes to the bridge here.
- **Real-robot RTAB-Map state determinism.** Real-robot doesn't teleport the chassis across power cycles, so cause (1) doesn't apply. If real-robot regressions surface as a side effect of (2)/(3) config changes, file a third brief; don't expand this one's scope.
