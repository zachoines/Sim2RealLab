# ProcRoom-scene bridge A/B — isolate the scene axis of the v1 drive-fault matrix

**Type:** investigation
**Owner:** DGX (cfg) / operator-run experiment (DGX sim + Jetson autonomy)
**Priority:** P2
**Estimate:** S (cfg shipped; the A/B is one operator session)
**Branch:** task/bridge-procroom

## Story

As a **deployment-validation operator**, I want **the v1 DEPTH_SUBGOAL policy
driven through the deploy pipeline against a training-distribution (ProcRoom)
scene**, so that **I can tell whether the observed drive fault is a domain gap
(deploy scene off-distribution) or a pipeline problem — and route the fix to the
right lane.**

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/env-composition-contract.md](../../context/env-composition-contract.md)
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

The v1 DEPTH_SUBGOAL drive-fault matrix has two axes — **scene** (ProcRoom vs
Infinigen) and **pipeline** (training play-loop vs deploy autonomy stack):

| | training pipeline | deploy pipeline |
|---|---|---|
| **ProcRoom scene** | ✓ works (play videos) | *this experiment* |
| **Infinigen scene** | — | ✗ fails (VLM missions) |

Training-pipeline × ProcRoom drives well (play videos); deploy × Infinigen
fails (missions). Those two cells differ on **both** axes at once, so the
failure is unattributed. This variant fills the untested cell — **deploy
pipeline × ProcRoom scene** — holding the pipeline and the v1 artifact fixed and
moving only the scene onto the training distribution:

- **Drives well** ⇒ the scene axis explains it: the deploy Infinigen scene is
  off-distribution for a ProcRoom-only policy → **domain gap** (route: the
  training lane — ProcRoom-enrichment stories).
- **Drives badly** ⇒ the pipeline axis explains it, and the same session's
  `--gym-dump` join localizes it (the structured train↔deploy obs residual).

Either way, **`Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-v0` then becomes the
standing regression rig** for the domain fix: an enriched-ProcRoom retrain
validates here (training-distribution scene, deploy pipeline) before it is ever
pointed at an Infinigen mission.

### What shipped in this PR (the cfg)

`Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-v0` — the Bridge capture stack
(`rgb_full` + `depth_full` + `depth_policy`) on the `procroom` scene source
instead of `infinigen`:

- `StraferSceneCfg_ProcRoomPerception` extends the ProcRoom scene with the
  640×360 perception camera (`d555_camera_perception`, RGB + depth) the bridge
  streams as `/d555/color/...` + `/d555/depth/image_rect_raw`, keeping the 80×45
  policy camera (`d555_camera`) so the gym-dump / obs shape matches training.
  It mirrors `StraferSceneCfg_InfinigenPerception` but keeps ProcRoom's per-env
  replicated physics (no `replicate_physics=False`).
- `StraferNavCfg_BridgeAutonomy_ProcRoom` composes it (`scene_source
  kind="procroom"`); `_select_scene` grows a `has_perception_cam` branch for the
  procroom source (mirroring the infinigen arm). No composition-machinery
  extension was needed — the procroom source is already wired into every axis
  selector table (events / commands / rewards / terminations / curriculum), so
  this is a leaf-addition, not a new axis.
- **No Infinigen scene machinery runs for this source.** At config time the
  `kind=="procroom"` branch runs only `_apply_procroom_physx_buffers` (never
  `_apply_infinigen_scene_setup`); at runtime the `--scene-usd` override and
  `--mode harness` are gated off the default `--mode bridge` path, and
  `_apply_scene_usd_spawn_override` now fails loud if handed a source with no
  `scene_geometry` prim. ProcRoom generates its geometry in-env and spawns the
  robot from its own BFS free-space events — it never reads
  `scenes_metadata.json` or `Assets/generated/scenes`.

## The experiment protocol (operator/triage — not run in this PR)

- **DGX (sim):**
  `$ISAACLAB -p source/strafer_lab/scripts/run_sim_in_the_loop.py --mode bridge
  --headless --enable_cameras --task
  Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-v0 --decimation 4 --render-interval
  4` — `decimation 4` is the fixed-dynamics config the policy was tuned against.
  `--mode bridge` only: this variant does **not** take `--scene-usd` or
  `--mode harness` (ProcRoom has no loaded scene USD to resolve; the cfg fails
  loud if you try).
  - **Same-session pipeline evidence (optional, gated on the gym-dumper):**
    add `--obs-dump-path <path> --obs-dump-variant DEPTH_SUBGOAL` to emit the
    training-side obs dump for the strict `--gym-dump` parity join. **These two
    flags come from the gym-obs-dumper track (PR #149) and are not on `main`
    yet** — land #149 first, or run the behavioral A/B without them today.
- **Jetson (autonomy):** normal bringup (`hybrid_nav2_strafer` + v1 +
  `replan_period_s=0.2`). **No VLM mission is possible** (ProcRoom has no
  groundable objects) — inject the goal directly:
  `ros2 action send_goal /strafer_inference/navigate_to_pose
  nav2_msgs/action/NavigateToPose "<pose ~2 m ahead in map frame>"` (the
  established smoke pattern; SLAM maps the procedural room fine).
- **Capture:** node obs dump + gym dump (if #149 landed) + the standard bag →
  the session yields BOTH the behavioral A/B and a ProcRoom-scene strict parity
  join.
- **Read-out:**
  - forward-driving + parking ≈ play-video behavior ⇒ **domain gap** (route:
    the training lane — ProcRoom-enrichment stories);
  - reverse/spin persists ⇒ **pipeline** (route: the structured residual from
    the same session's `--gym-dump` join names the offending obs dim).

## Acceptance criteria

- [x] `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-v0` registers and resolves to
      `StraferNavCfg_BridgeAutonomy_ProcRoom` + the CNN depth runner.
- [x] The cfg composes both cameras at the deploy resolutions (80×45 policy,
      640×360 perception, each RGB + depth), the contact sensor, and the
      ProcRoom managers — with no Infinigen `scene_geometry` / `spawn_points_xy`
      / `spawn_z` / ground-lift machinery. Pinned Kit-free in
      [`tests/navigation/test_bridge_procroom_cfg.py`](../../../source/strafer_lab/tests/navigation/test_bridge_procroom_cfg.py)
      and by `test_bridge_procroom_expanded_stack` in the composition-contract
      suite; `EXPECTED_ENVS` updated so the exact-set registration gate passes.
- [ ] **Experiment read-out (operator/triage):** the DGX-sim + Jetson-autonomy
      A/B above is run against a v1 DEPTH_SUBGOAL checkpoint, and the drive
      behavior is classified **domain-gap** vs **pipeline** per the read-out,
      routing the follow-up to the named lane. This closes the brief.
- [x] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit — `context/env-composition-contract.md`'s capture-family
      enumeration was updated here.
- [ ] No regression in the workflows the touched code supports: the Infinigen
      bridge/harness path is unchanged (its scene setup + `--scene-usd`
      re-derivation still run); confirm with the existing
      `test_bridge_spawn_from_occupancy` + `bridge_harness_smoke` smoke.

## Out of scope

- **ProcRoom scene enrichment / retrains.** If the read-out is *domain gap*, the
  fix is a training-lane story (richer ProcRoom obstacle geometry, moving
  obstacles, wider off-path bounds) validated against this rig — not this brief.
- **`--mode harness` / `--scene-usd` on the procroom source.** These are
  Infinigen-substrate affordances (loaded scene USD + `scenes_metadata.json`);
  ProcRoom has no such artifact, so they are intentionally unsupported here.
- **The gym-obs-dumper (`--obs-dump-*` / `--gym-dump`) itself.** That is the
  gym-obs-dumper track (PR #149); this brief only consumes it when it lands.
- **The Infinigen-mission failure diagnosis.** That is the deploy × Infinigen
  cell; this experiment attributes the axis, it does not fix the mission path.
