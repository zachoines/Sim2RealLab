# Validate the trained `NOCAM_SUBGOAL` policy end-to-end through the autonomy stack (hybrid backend, single-room) against the sim-in-the-loop rig

**Type:** task / validation
**Owner:** Either — recording the rosbag, running the sanity-check play episode, and emitting the gym-env ground-truth obs/subgoal require the DGX-side sim bridge running; launching the inference + subgoal-generator nodes, submitting the mission, and running the comparison scripts run on the Jetson side. The brief can be picked up by whichever lane the operator is in when the rig is up.
**Priority:** P2 — the active end-to-end validation milestone for the NOCAM_SUBGOAL deployment track (goal (a)). It converts "hybrid runtime is plumbing-correct + unit-tested" into "the policy actually drove a real mission in the bridge." Matches the sibling [`strafer-direct-sim-validation`](strafer-direct-sim-validation.md) tier.
**Estimate:** M (1–2 days once the rig is up and the deployable NOCAM_SUBGOAL artifact is on the Jetson; longer if a prerequisite needs work).
**Branch:** task/strafer-nocam-subgoal-singleroom-sim-validation

## Story

As a **mission operator validating the NOCAM_SUBGOAL deployment track**, I want **the trained policy to drive a single-room mission end-to-end through the autonomy stack in the sim bridge via a manual `strafer-autonomy-cli submit`, with positive confirmation the policy fired (not a silent Nav2 fallback)**, so that **I have evidence the hybrid runtime is deploy-correct before cross-room and real-robot validation**.

This is goal (a) of the deployment-validation track. The hybrid runtime (`hybrid_nav2_strafer`: Nav2 publishes the global plan, a subgoal generator rolls a subgoal along it off `/plan`, and the NOCAM_SUBGOAL policy does local control) ships as the runtime extension of [`hybrid-mode`](../../parked/trained-policy/hybrid-mode.md) across PRs #119 / #122 / #123. The deployable artifact exists from [`export-nocam-subgoal-variant`](../../completed/export-nocam-subgoal-variant.md) (PR #118).

The load-bearing risk for this validation is a **silent Nav2 fallback**: the inference node ships with an empty-string `model_path` sentinel and a default `policy_variant` of `DEPTH`, the hybrid dispatch falls back to Nav2 if the inference action server is unavailable, and `STRAFER_NAV_BACKEND` is consumed in the executor process — not the submit shell. A run can therefore look completely green while the policy never fires and Nav2 drives the whole mission. Every end-to-end acceptance criterion below is paired with a positive confirm-the-policy-fired check so a green mission cannot mask a fallback.

A confirmed runtime gap surfaced during recon and **must be resolved before scheduling the run** (see Prerequisites): as of the hybrid runtime branch, `mission_runner` always passes a non-`None` per-step `execution_backend` (config default `nav2`) and the plan compiler hardcodes `nav2` into the navigate step, and per-step beats the env var in `_resolve_execution_backend` — so `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` is shadowed on the normal mission path, and the plan validator's recognised-backend set excludes `hybrid_nav2_strafer` entirely. There is no operator-only lever; the navigate leg cannot reach the hybrid backend without a small code change. Resolve and land that wiring before the rig run, or this brief cannot pass.

## Context bundle

Read these before starting:

- [`hybrid-mode`](../../parked/trained-policy/hybrid-mode.md) — the runtime this validates (Nav2 global plan + RL local control); ships as PRs #119 / #122 / #123.
- [`export-nocam-subgoal-variant`](../../completed/export-nocam-subgoal-variant.md) — produces the deployable artifact (`models/strafer_nocam_subgoal_v0.{onnx,pt,json}`).
- [`strafer-direct-sim-validation`](strafer-direct-sim-validation.md) — the lean validation shape this clones (parity / one end-to-end win), retargeted here.
- Full-stack sim-in-the-loop runbook in `docs/example_commands_cheatsheet.md` (DGX shells: `serve-vlm`, `serve-planner`, bridge; Jetson shells: full bringup, inference + subgoal-generator nodes, mission submit).

## What already exists

- Exported NOCAM_SUBGOAL artifact on the DGX at `models/strafer_nocam_subgoal_v0.{onnx,pt,json}` (validated round-trip; `.json` records `policy_variant == NOCAM_SUBGOAL`).
- The hybrid runtime: variant-agnostic obs assembly + node lift (PR #122), the numpy rolling-subgoal generator cross-checked vs torch (PR #119), and `hybrid_nav2_strafer` dispatch + the plan-freshness watchdog (PR #123).
- The registered env ids for ground-truth and the sanity play: `Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-v0` (training/ground-truth) and `Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-v0` (play).
- The sim-in-the-loop rig + runbook in the cheatsheet.

## Prerequisites

1. **Hybrid runtime merged + a backend-selection path that actually reaches `hybrid_nav2_strafer`.** PR #123 merged to the line both the Jetson colcon checkout and the DGX bridge run, and the Jetson rebuilt (`make build`). **Before scheduling, land the executor backend-selection wiring** — as merged, the env var is shadowed by the per-step `nav2` default + the `plan_compiler` hardcode, and `_RECOGNISED_BACKENDS` excludes `hybrid_nav2_strafer`. Capture the agreed lever in this brief before the run (e.g. `executor/main.py` reads `STRAFER_NAV_BACKEND → MissionRunnerConfig.default_navigation_backend`, the `plan_compiler` navigate step stops hardcoding `nav2`, and `_RECOGNISED_BACKENDS` is expanded + reconciled with the dispatch names). An unresolved selection path is a hard blocker, not a runtime detail.
2. **Deployable NOCAM_SUBGOAL artifact on the Jetson.** rsync `models/strafer_nocam_subgoal_v0.onnx` from the DGX to a Jetson-local path and point the inference node `model_path` at the Jetson copy. The policy runs on the Jetson; the empty-string default sentinel must be overridden to this file. (Copy the `.json` sidecar alongside.)
3. **Sim-in-the-loop rig up.** DGX: `serve-vlm`, `serve-planner`, and the bridge (per the cheatsheet runbook). The bridge must republish D555 color so the executor's `scan_for_target` grounding has a frame — `serve-vlm` is required even though the policy is camera-free, because mission grounding is decoupled from the policy.
4. **A single-room scene with a single unambiguous target.** Load `scene_singleroom_000_seed0` via `--scene-usd` / `--scene-name` (generated per the [`single-room-couch-scene-supply`](../../completed/harness/single-room-couch-scene-supply.md) brief: `prep_room_usds.py generate --rooms living-room --quality low --name singleroom`). It is exactly one furnished `living_room` (customData `rooms[]` == 1, 46 objects) so any spawn point is co-room with the target. **The recorded goal-(a) target is `sofa`** — a uniquely-present, groundable object in that scene (count == 1; Infinigen furnished the living room with exactly one, so no forced-object fragility). Phrase the mission "go to the sofa". Cross-room is out of scope here.
5. **Sanity-check the policy in isolation first.** Run `play_strafer_navigation.py --env Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-v0 --checkpoint <converged ckpt>` and confirm the policy drives sensibly (tracks rolling subgoals, no degenerate spin/stall) before spending rig time. This isolates "policy is good" from "deployment plumbing is good."

## Acceptance criteria

### End-to-end single-room mission (gates on the deployable artifact + the rig)

- [ ] **Policy-driven single-room mission completes.** A manual mission submitted via `strafer-autonomy-cli submit "go to the sofa"` (the recorded unique groundable target in `scene_singleroom_000_seed0`) against that single-room scene completes (chassis reaches the target pose within `xy_goal_tolerance`) with the NOCAM_SUBGOAL policy doing local control — **not** a silent Nav2 fallback.
- [ ] **Confirm the policy actually fired (all of these).** This is the load-bearing anti-fallback check:
  - `ros2 action list` shows `/strafer_inference/navigate_to_pose` (advertised only when a policy loaded).
  - The inference node's `ready` param flips `True` after the mission starts (first successful inference).
  - Logs show **neither** "model_path is empty; refusing to advertise…" (at node startup) **nor** "hybrid_nav2_strafer backend selected but the strafer_inference action server (or Nav2 planner) is unavailable; falling back to nav2…" (at dispatch).
  - The hybrid dispatch log "Sending hybrid goal: … Nav2 planner-only + strafer_inference local control." appears.
  - `/strafer/cmd_vel` ticks at the policy rate (~30 Hz) sourced from `strafer_inference`, with `/strafer/subgoal` populated off Nav2 `/plan` — i.e. cmd_vel is coming from the policy, not from Nav2/MPPI.

### Obs parity (rig-only; the artifact need not be loaded for this check)

- [ ] **19-dim NOCAM_SUBGOAL obs parity ≤ 1e-5.** With a recorded sim-in-the-loop rosbag, the inference node's assembled NOCAM_SUBGOAL obs (all 19 dims) matches the gym-env (`Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-v0`) obs at the same sim timestamp within float32 noise (≤ 1e-5 max abs delta). Method: stand up the bringup on the Jetson with the DGX bridge producing `/clock`, `/d555/*`, `/strafer/*` (confirm via `ros2 topic hz`); DGX-side instrument the gym env to dump assembled NOCAM_SUBGOAL obs per step with sim timestamp; Jetson-side add a temporary `--dump-obs-to-jsonl` to emit the assembled obs per tick with sim timestamp; record a ≥30 s rosbag with the robot moving (`/d555/imu/filtered`, `/strafer/joint_states`, `/strafer/odom`, `/strafer/subgoal`, `/clock`); replay and join on sim timestamp asserting the bound. **There is no NOCAM_SUBGOAL deployment-side parity script yet** — landing it under `source/strafer_ros/strafer_inference/scripts/` is a deliverable of this brief. This check does not need the trained checkpoint: set `model_path=""` and record with the action server unadvertised. (There is no depth field for this variant, so the depth-parity check from the sibling does not apply.)

### Subgoal-pick parity (rig-only)

- [ ] **Subgoal-pose parity ≤ 10 cm.** The rolling subgoal the generator publishes on `/strafer/subgoal` off Nav2 `/plan` matches the training-time `SubgoalCommand` / path-cursor selection (the same numpy rolling-subgoal pick that PR #119 cross-checked against torch) within ≤ `MAP_RESOLUTION * 2` (10 cm) of the gym-env subgoal for the same plan/cursor state. Method: emit the gym-env's per-step `SubgoalCommand`/cursor pick alongside the obs dump, and join against the deployed `/strafer/subgoal` on sim timestamp in the same comparison script.

### Plan-freshness / stale-plan behavior

- [ ] **Stale `/plan` → STOP holds.** When Nav2 `/plan` ages past `path_timeout_s`, the subgoal generator suppresses `/strafer/subgoal` and the plan-freshness watchdog zeros the twist — observed `/strafer/cmd_vel` goes to zero (no last-subgoal coasting) until a fresh plan arrives, with the composed stop bounded at ~2.0 s (generator ~1.0 s + inference ~1.0 s). Exercise this by interrupting/aging the plan during a mission (e.g. a brief Nav2 planner stall) and confirm the zero-twist response, then recovery when `/plan` refreshes.

### Maintenance clause

- [ ] If your work invalidates a fact in any referenced context module, package README, top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md).

## Approach

1. **Sanity-play** the converged checkpoint in `Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-v0` (Prereq 5). Stop here if the policy itself is suspect.
2. **Land the backend-selection wiring** (Prereq 1) and confirm a real submitted mission resolves the navigate leg to `hybrid_nav2_strafer` (not `nav2`).
3. **Parity runs (no checkpoint):** record the rosbag with `model_path=""`, dump gym-env obs + subgoal picks DGX-side and assembled obs Jetson-side, land + run the comparison script, assert the 19-dim and 10 cm bounds.
4. **End-to-end run:** rsync the artifact to the Jetson; export `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` in the Jetson `make launch-sim` (executor) shell — not the submit shell — noting this is only effective once the Prereq-1 wiring lands; launch the inference node with `model_path` + `policy_variant=NOCAM_SUBGOAL` and the subgoal generator (neither is auto-launched by the bringup); submit the single-room mission; run the confirm-fired checks; then exercise the stale-plan STOP.

## Investigation pointers

- The bringup launch does **not** start `strafer_inference` or the subgoal generator — both are operator-launched separately on the Jetson with the `model_path` / `policy_variant` overrides (the launch files expose only `config_file` + `log_level`, so use a config overlay).
- `STRAFER_NAV_BACKEND` is read in the executor process (`make launch-sim` shell), but is shadowed on the normal mission path (per-step default + compiler hardcode + the plan validator's recognised-backend set). Resolve the selection lever before the run.
- The separately-launched inference + subgoal-generator nodes do not get `use_sim_time` from their launch files; consider passing `use_sim_time:=true` for `/clock` alignment with the bridge.

## Out of scope

- **Cross-room missions.** Blocked upstream by the multi-room autonomy-stack work; the cross-room reference mission lives in the parked [`strafer-hybrid-sim-validation`](../../parked/trained-policy/strafer-hybrid-sim-validation.md) superset. File any single-room breakage as its own follow-up.
- **Real-robot NOCAM_SUBGOAL validation (goal (b)).** Separate brief.
- **DEPTH_SUBGOAL / depth-variant validation (goal (c)).** Separate brief.
- **Re-tuning the policy if validation surfaces a gap.** Separate DGX-lane training brief.
- **Sim-in-the-loop rig setup.** Already in flight upstream; if broken, file the breakage as its own follow-up.
- **Mission-start latency p95 / per-tick latency benchmarks, the full safety/robustness suite, and trust-boundary documentation.** Deferred to the parked hybrid superset; goal (a) validates that the policy drives single-room end-to-end plus the obs/subgoal parity and stale-plan STOP.
