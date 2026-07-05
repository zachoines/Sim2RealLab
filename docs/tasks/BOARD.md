# Task board

Glanceable index of `docs/tasks/`. Pick from **Ready to pick up** by
matching your lane and session budget; cross-reference with
**By epic** for situational context on a feature area.

This file is updated **in the same commit that files / picks up /
ships a brief** — same maintenance contract as moving briefs to
[`completed/`](completed/). If the board and a brief disagree, the
brief wins; the board is a derived index, not the source of truth.

For brief format, authoring rules, and the agent-launcher template,
see [`README.md`](README.md). For the directory layout and the
parked sibling, see `README.md`'s `## Directory layout` section.
For lane definitions, see
[`context/ownership-boundaries.md`](context/ownership-boundaries.md).

---

## In flight

Open PRs — don't pick these up. Empty when no brief is being
executed (briefs are removed from this section in the same PR
that ships them; see "Shipping a brief: order of operations" in
[`README.md`](README.md)).

| Brief | Owner | PR | State |
|---|---|---|---|
| [`harness-architecture`](active/harness/harness-architecture.md) Tier 1 acceptance run | DGX | post-merge follow-up (PR #63 merged 2026-05-26) | pending operator capture; gated on [`teleop-perf-architecture`](completed/teleop-perf-architecture.md) (shipped 2026-06-01; loop is PhysX-bound, ~10 FPS not the ≥15 target, so a ≥30 ep × ≥2 scene run is faster but still not one-evening). Tier 1 ✓ on harness-architecture.md stays unchecked until artifact lands at `docs/artifacts/teleop_acceptance/<run_id>/`. |
| [`harness-architecture`](active/harness/harness-architecture.md) Tier 2 — bridge driver migration | DGX | post-merge follow-up ([#88](https://github.com/zachoines/Sim2RealLab/pull/88) merged) | pending operator gate — the multi-room end-to-end acceptance is operator-run per the PR test plan. Tier 2 shipped: bridge `--mode harness` → LeRobot v3 writer; both `capture.py` bridge cells wired (queue cell against a hand-authored fixture — mission-generator unshipped); `--inject-bad-grounding` + detections columns; unit suites + Jetson-free Kit smoke green. Brief stays active until Tier 3 ships. |
| [`depth-subgoal-env`](active/trained-policy/depth-subgoal-env.md) | DGX | [#138](https://github.com/zachoines/Sim2RealLab/pull/138) (open) | Un-parked 2026-07-04 (operator epic decision 5 + soft trigger: NOCAM corner-contact is the depth-recoverable failure class). This PR delivers Phases 1–4 (`PolicyVariant.DEPTH_SUBGOAL`, depth-aware obstacle-proximity reward, depth×subgoal env composition + 4 task IDs, Kit-free tests). Brief stays active — Phase 5 (the DEPTH-rate training run + converged checkpoint) is operator-gated and closes it later. |

---

## By epic

Feature-area view. Active briefs are pickable; parked briefs are
filed-on-trigger or blocked-on-deps — see **Parked** below for the
explicit dependencies.

### Multi-room navigation

For how these briefs layer (v1 / v1.5 / v2 / v2.5 / v3 / escape valves) and how the multi-room work relates to the implicit-mapping track in the clip-validation epic, see [`context/multi-room-architecture.md`](context/multi-room-architecture.md).

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`autonomy-stack`](active/multi-room/autonomy-stack.md) | P1 | active | Either |
| [`room-state-eval-harness`](active/multi-room/room-state-eval-harness.md) | P2 | active | DGX |
| [`semantic-region-partition`](active/multi-room/semantic-region-partition.md) | P2 | active | DGX |
| [`planner-far-target-staging`](active/multi-room/planner-far-target-staging.md) | P2 | active | DGX |
| [`semantic-graph-loop-closure`](active/multi-room/semantic-graph-loop-closure.md) | P3 | active | DGX |
| [`llm-guided-frontier-gain`](parked/multi-room/llm-guided-frontier-gain.md) | P2 | parked | DGX |
| [`frontier-cognitive-fsm`](parked/multi-room/frontier-cognitive-fsm.md) | P3 | parked | Either |
| [`room-state-runtime-ergonomics`](parked/multi-room/room-state-runtime-ergonomics.md) | P3 | parked | DGX |
| [`semantic-map-lifecycle-merge`](parked/multi-room/semantic-map-lifecycle-merge.md) | P3 | parked | DGX |
| [`staging-hops-shadow-mode`](parked/multi-room/staging-hops-shadow-mode.md) | P3 | parked | DGX |
| [`planner-scene-graph-expansion`](parked/multi-room/planner-scene-graph-expansion.md) | P3 | parked | DGX |
| [`dynamic-region-granularity`](parked/multi-room/dynamic-region-granularity.md) | P3 | parked | DGX |
| [`learned-spatial-encoder`](parked/multi-room/learned-spatial-encoder.md) | P3 | parked | DGX |

### Trained-policy backend

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`export-sidecar-training-preset`](active/trained-policy/export-sidecar-training-preset.md) | P3 | active | DGX |
| [`policy-export-deprecation-migration`](active/trained-policy/policy-export-deprecation-migration.md) | P3 | active | DGX |
| [`strafer-direct-sim-validation`](active/trained-policy/strafer-direct-sim-validation.md) | P2 | active | Either |
| [`recurrent-state-contract`](active/trained-policy/recurrent-state-contract.md) | P1 | active | Either |
| [`encoder-noise-shared-sample`](active/trained-policy/encoder-noise-shared-sample.md) | P2 | active | DGX |
| [`policy-rate-shared-constants`](active/trained-policy/policy-rate-shared-constants.md) | P2 | active | DGX |
| [`domain-randomization-audit`](active/trained-policy/domain-randomization-audit.md) | P1 | active | DGX |
| [`goal-noise-training`](active/trained-policy/goal-noise-training.md) | P2 | active | DGX |
| [`depth-subgoal-env`](active/trained-policy/depth-subgoal-env.md) | P3 | active | DGX |
| [`depth-subgoal-hybrid-runtime`](parked/trained-policy/depth-subgoal-hybrid-runtime.md) | P3 | parked | Jetson |
| [`batched-gpu-path-planner`](parked/trained-policy/batched-gpu-path-planner.md) | P3 | parked | DGX |
| [`rl-global-nav2-local`](parked/trained-policy/rl-global-nav2-local.md) | P3 | parked | Either |
| [`subgoal-corridor-clearance`](parked/trained-policy/subgoal-corridor-clearance.md) | P3 | parked (filed-on-trigger: GRU arm still cuts corners) | DGX |

### Harness & training data

Five briefs (`behavior-cloning-data-expansion`, `teleop-driver`, `trajectory-first-captioning`, `oracle-driver`, `output-format-alignment`) were consolidated 2026-05-24 into [`harness-architecture`](active/harness/harness-architecture.md). The originals live in [`completed/`](completed/) with retired-not-shipped stamps.

Path-planning consumers here (`mission-generator`'s oracle + waypoint validation, `grounding-negative-taxonomy`'s violation paths) build on the one shared planner — see [`context/path-planning-architecture.md`](context/path-planning-architecture.md) (`subgoal-env`'s grid A* core + per-scene occupancy-grid adapters).

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`harness-architecture`](active/harness/harness-architecture.md) | P1 | active (architecture doc; ships across PRs B/C/D) | DGX |
| [`coverage-capture-overbright-exposure`](completed/harness/coverage-capture-overbright-exposure.md) | P0 | shipped 2026-06-28 ([#117](https://github.com/zachoines/Sim2RealLab/pull/117), `8cd3bb4`) — over-bright d555 perception RGB (seed6/7 88%/75% clipped, 25/38 fully-white frames) fixed at the render layer: enable RTX histogram auto-exposure corpus-wide via `RenderCfg.carb_settings` (re-bake-free) + `coverage_capture --render-carb` probe + `measure_perception_exposure.py` gate. Validated on operator captures: clip → 1.5%/2.3%, 0 white frames, luma in band, seed2 PASS. Marginal crush/clip fine-tuning deferred to `perception-exposure-finetune`. | DGX |
| [`grounding-found-false-negatives`](active/harness/grounding-found-false-negatives.md) | P1 | active — unowned found-axis grounder negatives (target-absent frames); rides on coverage capture. Filed off the 2026-06-23 gap analysis. | DGX |
| [`perception-exposure-finetune`](parked/harness/perception-exposure-finetune.md) | P2 | parked (filed-on-trigger: next Infinigen corpus regen) — close the marginal crushed-black/clip gap left by #117 (ambient-fill sweep + band calibration); incl. the "drop inject_ceiling_light_emitters?" Kit check. | DGX |
| [`harness-throughput-measurement`](parked/harness/harness-throughput-measurement.md) | P2 | parked | DGX |
| [`scene-provider-floor-sampler-cli`](parked/harness/scene-provider-floor-sampler-cli.md) | P3 | parked (filed-on-trigger) | DGX |
| [`cosmos-replay-perturbation`](parked/harness/cosmos-replay-perturbation.md) | P3 | parked | DGX |
| [`depth-ffv1-video-column`](parked/harness/depth-ffv1-video-column.md) | P2 | parked (sequenced after the R1 `observation.detections.*` column; spike-gated) | DGX |
| [`harness-mode-modularization`](parked/harness/harness-mode-modularization.md) | P3 | parked (after Tier 2 bridge driver #88 lands) | DGX |
| [`grounding-negative-taxonomy`](parked/harness/grounding-negative-taxonomy.md) | P2 | parked (after `mission-generator` ships path-shape missions) | DGX |
| [`infinigen-label-taxonomy`](parked/harness/infinigen-label-taxonomy.md) | P3 | parked (filed-on-trigger) | DGX |
| [`distractor-asset-injection`](parked/harness/distractor-asset-injection.md) | P3 | parked (filed-on-trigger) | DGX |
| [`out-of-process-mission-text-llm`](parked/harness/out-of-process-mission-text-llm.md) | P3 | parked (filed-on-trigger) | DGX (cross-lane: autonomy) |
| [`capture-debug-overhead-cam`](parked/harness/capture-debug-overhead-cam.md) | P3 | parked (filed-on-trigger) | DGX |

### CLIP mid-mission validation

The learned components here share one frozen text-capable backbone — see [`context/perception-backbone-architecture.md`](context/perception-backbone-architecture.md) for the trunk + per-consumer-heads spine the briefs below sit on.

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`validator-evaluation`](active/clip-validation/validator-evaluation.md) | P1 | active | Either |
| [`backbone-bakeoff`](parked/clip-validation/backbone-bakeoff.md) | P2 | parked (runs *before* validator-evaluation v1 — picks the shared trunk) | DGX |
| [`cotrained-retrieval-augmented`](parked/clip-validation/cotrained-retrieval-augmented.md) | P3 | parked | DGX |
| [`implicit-memory-map`](parked/clip-validation/implicit-memory-map.md) | P3 | parked (planned warm-map consumer) | DGX |
| [`clip-multi-room-validator-remeasure`](parked/clip-validation/clip-multi-room-validator-remeasure.md) | P2 | parked | Either |
| [`vlm-grounding-finetune`](parked/clip-validation/vlm-grounding-finetune.md) | P2 | parked | DGX |

### Sim performance

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`isaac-sim-rt-2-default-renderer`](active/sim-performance/isaac-sim-rt-2-default-renderer.md) | P2 | active | DGX |
| [`bridge-throughput-toward-25hz`](active/sim-performance/bridge-throughput-toward-25hz.md) | P2 | active | DGX |
| [`mecanum-action-throughput`](active/sim-performance/mecanum-action-throughput.md) | P2 | active | DGX |
| [`bridge-publish-rate-decouple`](active/sim-performance/bridge-publish-rate-decouple.md) | P2 | active | DGX |
| [`gpu-solver-partitions-default`](active/sim-performance/gpu-solver-partitions-default.md) | P3 | active | DGX |
| [`bridge-scene-memory-budget-gb10`](active/sim-performance/bridge-scene-memory-budget-gb10.md) | P2 | active | DGX |

### Reliability (nav + executor + refactors)

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`nav2-mppi-motion-model-investigation`](active/reliability/nav2-mppi-motion-model-investigation.md) | P2 | active | Jetson |
| [`rtabmap-cold-start-determinism`](active/reliability/rtabmap-cold-start-determinism.md) | P2 | active | Jetson |
| [`executor-grounding-loss-mid-mission-recovery`](active/reliability/executor-grounding-loss-mid-mission-recovery.md) | P2 | active | Jetson |
| [`executor-replan-on-deviation`](parked/reliability/executor-replan-on-deviation.md) | P2 | parked | Either |
| [`executor-slam-tracking-precheck-mid-mission`](active/reliability/executor-slam-tracking-precheck-mid-mission.md) | P2 | active | Jetson |
| [`verify-arrival-occlusion-robustness`](active/reliability/verify-arrival-occlusion-robustness.md) | P2 | active | Jetson |
| [`planner-rotate-direction-prompt`](active/reliability/planner-rotate-direction-prompt.md) | P2 | active | DGX |
| [`rotate-in-place-large-angle-correctness`](active/reliability/rotate-in-place-large-angle-correctness.md) | P2 | active | Jetson |
| [`grounding-publisher-extraction`](active/reliability/grounding-publisher-extraction.md) | P2 | active | Jetson |
| [`nav-stall-multilayer-watchdog`](parked/reliability/nav-stall-multilayer-watchdog.md) | P3 | parked | Jetson |
| [`perception-side-bearing-service`](parked/reliability/perception-side-bearing-service.md) | P3 | parked | Jetson |
| [`d555-usb-dropout-framerate-collapse`](parked/reliability/d555-usb-dropout-framerate-collapse.md) | P2 | parked | Jetson |
| [`roboclaw-error-visibility-and-low-battery`](parked/reliability/roboclaw-error-visibility-and-low-battery.md) | P2 | parked | Jetson |
| [`imu-yaw-drift-no-magnetometer`](parked/reliability/imu-yaw-drift-no-magnetometer.md) | P2 | parked | Either |
| [`robot_localization-ekf-fused-odom`](parked/reliability/robot_localization-ekf-fused-odom.md) | P3 | parked | Jetson |
| [`executor-startup-health-check-contract`](active/reliability/executor-startup-health-check-contract.md) | P3 | active | Jetson |

### Tooling & ops

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`test-ci-workflow`](active/tooling/test-ci-workflow.md) | P3 | active | Either |
| [`isaac-lab-upgrade`](active/tooling/isaac-lab-upgrade.md) | P3 | active | DGX |
| [`script-tool-subsystem-grouping`](active/tooling/script-tool-subsystem-grouping.md) | P3 | active | DGX |
| [`windows-workstation-bringup`](active/tooling/windows-workstation-bringup.md) | P2 | active | DGX |
| [`jetson-test-gate-cross-lane-deps`](active/tooling/jetson-test-gate-cross-lane-deps.md) | P3 | active | Either |
| [`archive-interim-architecture-docs`](active/tooling/archive-interim-architecture-docs.md) | P3 | active | DGX |
| [`python-lint-format-baseline`](active/tooling/python-lint-format-baseline.md) | P3 | active | Either |
| [`tools-package-reorg`](parked/tooling/tools-package-reorg.md) | P3 | parked (land when no large `tools/`-touching PR is in flight — after the R1 detections column + `depth-ffv1-video-column` settle) | DGX |
| [`scene-contract-instance-discriminator`](parked/tooling/scene-contract-instance-discriminator.md) | P3 | parked (filed-on-trigger) | DGX |

### Experimental (long-horizon bets)

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`vla-v2-architecture`](parked/experimental/vla-v2-architecture.md) | P3 | parked | Either |
| [`vla-v2-map-conditioning`](parked/experimental/vla-v2-map-conditioning.md) | P3 | parked | DGX |
| [`mujoco-warp-physics-backend-spike`](parked/experimental/mujoco-warp-physics-backend-spike.md) | P3 | parked | DGX |

### Investigations (measurement / knowledge work)

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`next-integration-round`](active/investigations/next-integration-round.md) | P1 | active | Either |
| [`real-d555-depth-range-survey`](active/investigations/real-d555-depth-range-survey.md) | P2 | active | Jetson |
| [`training-throughput-profile-and-investigate`](active/investigations/training-throughput-profile-and-investigate.md) | P2 | active | DGX |
| [`defm-preprocess-antialias-audit`](active/investigations/defm-preprocess-antialias-audit.md) | P3 | active | DGX |
| [`collision-imu-signal-flaky`](active/investigations/collision-imu-signal-flaky.md) | P3 | active | DGX |
| [`sim-wall-service-latency-gap`](active/investigations/sim-wall-service-latency-gap.md) | P3 | active | Either |

---

## Ready to pick up

Grouped by priority tier, then by lane. Within each cell the rough
order is "smallest / least-blocking first," but pick what fits your
session. Parked briefs are not listed here — see **By epic** or
**Parked** below.

### P1 — high priority

| Brief | Owner | Estimate | Note |
|---|---|---|---|
| [`next-integration-round`](active/investigations/next-integration-round.md) | Either | M–L | Full end-to-end sim-in-the-loop run against `INTEGRATION_SIM_IN_THE_LOOP.md`; gating signal that bridge + autonomy + VLM/CLIP compose end-to-end |
| [`validator-evaluation`](active/clip-validation/validator-evaluation.md) | Either | L | Wire the orphaned `SemanticMapManager` + `BackgroundMapper` + `TransitMonitor` path into the production executor and measure pre-registered TPR/FPR/time-to-decision on harness output. Gating brief for `MISSION_VALIDATION_ARCHITECTURE.md` §4 staged plan. Filed off `mid-mission-validation-investigation` ship. |
| [`harness-architecture`](active/harness/harness-architecture.md) | DGX | XL (split across PRs B/C/D — see brief's Implementation tiers) | Architecture spec for the consolidated harness: one `source/strafer_lab/scripts/capture.py` entry point with `--driver` × `--mission-source` flags + LeRobot v3 canonical output. **Next pickable slice: Tier 1 (writer + teleop driver)** unblocks v1 measurement and v2 VLA training data without depending on bridge perf. Subsumes the retired teleop-driver / behavior-cloning-data-expansion / trajectory-first-captioning / oracle-driver / output-format-alignment briefs. |
| [`autonomy-stack`](active/multi-room/autonomy-stack.md) | Either | M | Lifts §1.10.1's multi-room deferral. Stored-map fallback in `scan_for_target` + planner transit-step emission + plan-compiler updates. Blocks on `observation-derived-room-state` and `frontier-exploration-primitive` (`planner-architecture-alignment` shipped in #36 as Option C). |
| [`domain-randomization-audit`](active/trained-policy/domain-randomization-audit.md) | DGX | M | Bench-measure real-chassis variability (mass, battery, D555 latency, Jetson jitter) and widen REAL_ROBOT_CONTRACT to match. Resume-train DEPTH baseline against the audited DR. Filed off the 2026-05-15 trained-policy audit. |

### P2 — medium priority

#### DGX lane

| Brief | Estimate | Note |
|---|---|---|
| [`isaac-sim-rt-2-default-renderer`](active/sim-performance/isaac-sim-rt-2-default-renderer.md) | S | Flip default renderer to Real-Time 2.0 + 4× FPS multiplier + Performance mode; re-measure bridge perf |
| [`bridge-scene-memory-budget-gb10`](active/sim-performance/bridge-scene-memory-budget-gb10.md) | M | Bridge/harness OOM on the GB10: `StraferNavCfg_BridgeAutonomy` loads the `sorted()`-first scene — currently a 29 GB `high_quality_dgx` room (1024-px tex / 5 rooms) — into the unified 121 GB pool → NVRM OOM-kill during render init. Add a deterministic scene-selection knob + a GB10 texture/room budget (or downscale-on-ingest); confirm the torch sm_121 build. `SCENE_USD=<singleroom>.usdc make sim-bridge` light-scene pin workaround exists (`--rooms living-room --quality low`). |
| [`planner-rotate-direction-prompt`](active/reliability/planner-rotate-direction-prompt.md) | S | Quick win — prompt edit |
| [`goal-noise-training`](active/trained-policy/goal-noise-training.md) | M | Targeted DEPTH-baseline training pass with goal-position noise; gates VLM-grounded mission quality for `strafer_direct` |
| [`policy-rate-shared-constants`](active/trained-policy/policy-rate-shared-constants.md) | S (~1 hr) | Delegate `_DEFAULT_NAV_SIM_DT` / `_DEFAULT_NAV_DECIMATION` in `strafer_env_cfg.py` to the new `strafer_shared.constants.POLICY_SIM_DT` / `POLICY_DECIMATION`, **plus** (added 2026-07-03) a shared `CMD_WATCHDOG_TIMEOUT_S` that `roboclaw_node.WATCHDOG_TIMEOUT_SEC` and `BridgeConfig.cmd_watchdog_sim_s` (PR #134) both default from — same stream-relative window, each side's own clock domain. Closes the duplications so neither the training rate nor the stop-on-silence window can silently desync sim from real |
| [`planner-far-target-staging`](active/multi-room/planner-far-target-staging.md) | M–L | World-state schema + planner prompt |
| [`windows-workstation-bringup`](active/tooling/windows-workstation-bringup.md) | L (~1 wk) | Investigation + port — run `make sim-bridge` on Windows (RTX 4080) against the Jetson stack. Isaac Lab 3 Windows support is experimental; phase the feasibility spike before committing to a full port |
| [`bridge-throughput-toward-25hz`](active/sim-performance/bridge-throughput-toward-25hz.md) | M | Follow-up to `async-camera-publishers`. Lift the bridge toward the predicted 25 Hz ceiling. |
| [`bridge-publish-rate-decouple`](active/sim-performance/bridge-publish-rate-decouple.md) | M | Spun out of `roller-contact-high-omega-bounce` (PR #76). Bridge runs `decimation 1` (untuned 120 Hz control) to keep the Jetson publish rate ~29 Hz; `decimation 4` fixes control fidelity but starves publish to ~8 Hz. Decouple publish cadence from control decimation so the bridge gets both. Coordinate with `bridge-throughput-toward-25hz`. |
| [`gpu-solver-partitions-default`](active/sim-performance/gpu-solver-partitions-default.md) | S | Spun out of `roller-contact-high-omega-bounce` (PR #76). ProcRoom pins `gpu_max_num_partitions=1` (default 8) from the high-env-count flip fix, not the bounce. Revert to default for solver parallelism and re-validate flips at production env count, or document why it must stay 1. |
| [`encoder-noise-shared-sample`](active/trained-policy/encoder-noise-shared-sample.md) | M | Filed off `observation-contract-cleanup` ship. Per-tick noised-ticks cache + policy/critic obs-function split so `wheel_encoder_velocities` and `body_velocity_xy` share a single encoder noise sample (matches real-robot signal chain). Closes the correlation gap that observation-contract-cleanup flagged as out of scope. |
| [`training-throughput-profile-and-investigate`](active/investigations/training-throughput-profile-and-investigate.md) | S–M | Phase profiler in the training loop; files follow-up briefs from results. |
| [`room-state-eval-harness`](active/multi-room/room-state-eval-harness.md) | M | v2 room-state — measurement harness for cluster purity / label precision / time-to-converge / connectivity P-R on a fixed multi-room scene set (incl. open-plan + multi-bedroom adversarials). Pure-eval brief; consumes LeRobot v3 datasets from [`harness-architecture`](active/harness/harness-architecture.md)'s scripted × coverage path. Blocks pickup on harness Tier 3 + `observation-derived-room-state` (shipped). |
| [`semantic-region-partition`](active/multi-room/semantic-region-partition.md) | M | **The v2 quality work** — feature+space HDBSCAN clustering + open-vocab labels, replacing v1's greedy-modularity + 7-class argmax. One `α` knob (SOTA-aligned, ConceptGraphs / HOV-SG shape); no training. Handles open-plan + multi-bedroom by construction. `RoomEntry` shape preserved (+`uncertainty`). |

#### Jetson lane

| Brief | Estimate | Note |
|---|---|---|
| [`grounding-publisher-extraction`](active/reliability/grounding-publisher-extraction.md) | S | Quick win — pure refactor follow-up to `vlm-bbox-overlay`; extracts the viz publishers out of `JetsonRosClient` |
| [`real-d555-depth-range-survey`](active/investigations/real-d555-depth-range-survey.md) | S–M | Investigation — bench measurement + write-up |
| [`rtabmap-cold-start-determinism`](active/reliability/rtabmap-cold-start-determinism.md) | M | Cold-start signature on populated DB: `Not found word N` burst + `Increment map id to 4`; triage bridge-teleport vs Mem/* config + ship the chosen disposition. After audit: A2 recommended — flip `localization:=true` default when populated DB exists. |
| [`nav2-mppi-motion-model-investigation`](active/reliability/nav2-mppi-motion-model-investigation.md) | M | Bench-measure Omni vs DiffDrive MPPI motion model on the same lap mission. Symptom: with Omni, MPPI's `/optimal_trajectory` correctly recovers toward `/plan` but the chassis goes tangent or reverses — gap between commanded velocity and executed motion. DiffDrive collapses the sampling space to (vx, wz), removing the vy/strafe sampling that Omni produces unstably. Cardinal-strafe path already lives at the executor layer. |
| [`rotate-in-place-large-angle-correctness`](active/reliability/rotate-in-place-large-angle-correctness.md) | S–M | `rotate_in_place` closes the loop on a single normalized target yaw: `rotate 360` no-ops (2π→0 target) and `>180°` takes the short way. Track accumulated traversal instead. Surfaced in PR #45 e2e (rotate 360 vs 180 at RTF≈0.085). |
| [`executor-grounding-loss-mid-mission-recovery`](active/reliability/executor-grounding-loss-mid-mission-recovery.md) | M | `_navigate_via_staging` re-grounding failure terminates immediately. Add mini-scan + semantic-map fallback with bounded recovery budget. Filed off the 2026-05-17 reliability audit. |
| [`executor-slam-tracking-precheck-mid-mission`](active/reliability/executor-slam-tracking-precheck-mid-mission.md) | S–M | Executor never queries `check_slam_tracking()`; silent failure when RTAB-Map loses tracking mid-mission. Add bounded precheck before each motion step. Filed off the 2026-05-17 reliability audit. |
| [`verify-arrival-occlusion-robustness`](active/reliability/verify-arrival-occlusion-robustness.md) | S–M | `_verify_arrival` false-negatives under partial occlusion. Add multi-frame voting + tilt-recovery + `arrival_occluded` soft-failure code. Filed off the 2026-05-17 reliability audit. |

#### Either lane

| Brief | Estimate | Note |
|---|---|---|
| [`strafer-direct-sim-validation`](active/trained-policy/strafer-direct-sim-validation.md) | M (1–2 days, rig-dependent) | Operator-driven sim validation extracted from the [`inference-package`](completed/inference-package.md) PR so it could merge with unit-testable acceptance closed. Three independent runs: rosbag parity (≤1e-5 NOCAM / ≤1e-3 depth), TRT-EP latency p95 < 10 ms, and the architectural-win mission (≥ 1.0 m/s sustained + obstacle avoidance). Last item gates on a deployable DEPTH checkpoint; the first two only need the sim-in-the-loop rig. |

### P3 — pickable, low priority

| Brief | Owner | Estimate | Note |
|---|---|---|---|
| [`test-ci-workflow`](active/tooling/test-ci-workflow.md) | Either | M | CI half of the now-shipped [`unify-test-targets-and-ci`](completed/unify-test-targets-and-ci.md) — a GitHub Actions matrix over `make test-*`. Land `autonomy` per-PR first (minimal safe gate); `vlm`/`ros` next; `lab` nightly + informative (needs a self-hosted DGX runner). Confirm the repo's Actions permissions first. |
| [`script-tool-subsystem-grouping`](active/tooling/script-tool-subsystem-grouping.md) | DGX | M | Sub-group `scripts/` by sub-system (policy/ infinigen/ diagnostics/ harness/) **and** amend the conventions placement rule (flat → sub-system) — the shared rule that also governs `tools/`. `tools/` itself is owned by `tools-package-reorg` (deduped). `scripts/harness/` sequenced behind `harness-architecture`. |
| [`python-lint-format-baseline`](active/tooling/python-lint-format-baseline.md) | Either | M | Get `make lint` (flake8) + `make format-check` (black) to green — never run before, expect a large backlog. Resolve the flake8(100)/black(120) line-length conflict + pin the toolchain (DGX base has neither) first; reformat formatting-only, ideally per-package to respect lane boundaries. Prerequisite for a CI lint gate (`test-ci-workflow`). |
| [`export-sidecar-training-preset`](active/trained-policy/export-sidecar-training-preset.md) | DGX | S | Sidecar `training_preset` records the configclass name instead of the rsl_rl preset variable; cosmetic but the field is operator-facing. Filed off [`export-onnx-depth`](completed/export-onnx-depth.md). |
| [`defm-preprocess-antialias-audit`](active/investigations/defm-preprocess-antialias-audit.md) | DGX | S–M | Measure projection-space delta between training-time DeFM antialiased preprocessing and the deployment ONNX-safe non-antialiased version, then decide alignment (leave / align deploy / align training). Filed off [`export-onnx-depth`](completed/export-onnx-depth.md). |
| [`collision-imu-signal-flaky`](active/investigations/collision-imu-signal-flaky.md) | DGX | S–M | `test_collision_imu_mean_differs_from_free` flakes (~50%, same command) — post-restitution-0 collisions no longer clear the IMU-vs-free significance bar. Strengthen the scenario, re-frame the assertion, or retire it. Surfaced + un-masked by [`strafer-lab-test-tree-unification`](completed/strafer-lab-test-tree-unification.md). |
| [`sim-wall-service-latency-gap`](active/investigations/sim-wall-service-latency-gap.md) | Either | S–M | Under `use_sim_time` at RTF ~0.1, wall-clock VLM/planner calls cost ~10× less mission time in sim than they will on the real robot. Measure per-endpoint latency, re-price the sim-validated deadlines that span service calls at RTF 1, file follow-ups. Filed off the 2026-07-02 sim-in-the-loop debugging. |
| [`isaac-lab-upgrade`](active/tooling/isaac-lab-upgrade.md) | DGX | M–L | Bump the pinned Isaac Lab (develop @ 2026-04-23, ~6 wks stale) + recreate `env_isaaclab3`; re-validate the sim stack via `make test-lab` + training/bridge smokes. Records the torch delta for the `.venv_vlm` consolidation question. |
| [`policy-export-deprecation-migration`](active/trained-policy/policy-export-deprecation-migration.md) | DGX | M–L | Move policy export off deprecated `torch.jit.*` / legacy `torch.onnx.export` (torch 2.9+ warnings) to a path the Jetson still loads, preserving determinism + the recurrent + cross-format-parity contracts. Gated by `isaac-lab-upgrade` (urgent once torch drops the legacy path). |
| [`semantic-graph-loop-closure`](active/multi-room/semantic-graph-loop-closure.md) | DGX | M | v2 room-state — detect duplicate-place nodes via CLIP-similarity + spatial proximity, annotate as `same_place` edges. Quiet long-horizon quality lift; required infrastructure for the parked `semantic-map-lifecycle-merge`. |

---

## Quick wins

Briefs estimated **S** that any agent can knock out in <1 day. Useful
for fresh-session pickup. Cross-cut — these also appear above.

- [`planner-rotate-direction-prompt`](active/reliability/planner-rotate-direction-prompt.md) (DGX, P2)
- [`grounding-publisher-extraction`](active/reliability/grounding-publisher-extraction.md) (Jetson, P2)
- [`isaac-sim-rt-2-default-renderer`](active/sim-performance/isaac-sim-rt-2-default-renderer.md) (DGX, P2)

---

## Parked

Filed-on-trigger or blocked-on-deps. These briefs live in
`docs/tasks/parked/<epic>/` and are not pickable until the trigger
fires or the dependency lands. Un-park by `git mv
parked/<epic>/<brief>.md active/<epic>/<brief>.md` in the PR that
picks them up.

| Brief | Trigger / blocks on | Why |
|---|---|---|
| [`depth-subgoal-hybrid-runtime`](parked/trained-policy/depth-subgoal-hybrid-runtime.md) | [`hybrid-mode`](completed/hybrid-mode.md) ✅ shipped **and** [`depth-subgoal-env`](active/trained-policy/depth-subgoal-env.md) shipped | Jetson-side runtime extension that composes the existing DEPTH observation pipeline with the hybrid-mode rolling-subgoal selection. Recommends a variant-agnostic refactor of the four hardcoded-DEPTH paths in PR #55's `strafer_inference` rather than a DEPTH_SUBGOAL-specific branch — the refactor becomes a durable substrate for any future variant the loader can produce. |
| [`batched-gpu-path-planner`](parked/trained-policy/batched-gpu-path-planner.md) | Trigger: profiling shows reset-time path planning gates training throughput (much larger env counts / shorter episodes than the ~256-env baseline). Filed off the `subgoal-env` PR #87 review; the numpy A* is correct and not a measured bottleneck today | Batched GPU wavefront / distance-field planner (extends `proc_room._gpu_bfs`) replacing the per-env numpy A* loop in `SubgoalCommand._resample_command` + vectorized `PathCursor.set_paths`. Same `nav_msgs/Path` output contract; the per-step `PathCursor.update` is already vectorized. Don't pick up without a profile justifying it. |
| [`rl-global-nav2-local`](parked/trained-policy/rl-global-nav2-local.md) | Trigger: first end-to-end deployment of `strafer_direct` (DEPTH MVP) or `hybrid_nav2_strafer` reveals that local-control RL is insufficient for VLM-grounded missions and the global-plan layer is the issue | Alternative architecture corner: RL as global waypoint planner, Nav2 as local controller. Filed off the 2026-05-15 trained-policy audit. Don't pick up preemptively — needs deployment evidence first |
| [`subgoal-corridor-clearance`](parked/trained-policy/subgoal-corridor-clearance.md) | Trigger: the recurrent (GRU) arm from [`nocam-subgoal-recurrent-runner`](completed/nocam-subgoal-recurrent-runner.md) still shows corner-cutting (`sustained_collision` vs `path_complete` fractions / corner cross-track error) at convergence | Corner clearance for the rolling-subgoal path. The planner-inflation margin is infeasible at this geometry (0.8 m min doorway + whole-cell inflation seals doors; ±0.12 m slack *is* the tracking spec). Lead candidate: planner-side medial-axis waypoint biasing that degrades to zero in doorways (also aligns training paths with Nav2's cost-decay deployment paths). Carries the off-path-bound decouple; widening doorways is rejected on principle. Don't pre-empt — recurrence may suffice. |
| [`cotrained-retrieval-augmented`](parked/clip-validation/cotrained-retrieval-augmented.md) | [`validator-evaluation`](active/clip-validation/validator-evaluation.md) shipped + [`harness-architecture`](active/harness/harness-architecture.md) Tier 3 shipped (provides the scripted × captioner speaker corpus) **and** [`backbone-bakeoff`](parked/clip-validation/backbone-bakeoff.md) shipped (selects the backbone) | Replaces the retired `learned-mid-mission-validator` as the cascade-improvement path. The co-training step and retrieval-augmented step compound; both compose with the existing cascade. |
| [`backbone-bakeoff`](parked/clip-validation/backbone-bakeoff.md) | **Reordered to run *before* `validator-evaluation` v1** (it picks the shared frozen trunk v1 ships on); now gated on harness Tier-1 episodes + the shared eval scaffolding, not on validator-evaluation shipping. See [`context/perception-backbone-architecture.md`](context/perception-backbone-architecture.md). | Narrowed to SigLIP-2-Base (lead) + MobileCLIP-2-S vs the OpenCLIP ViT-B/32 baseline (DINOv3-S demoted to a hybrid vision tower only if single-tower VPR fails); per-candidate eval widened with VPR Recall@K + region V-measure so one trunk serves all consumers. |
| [`vla-v2-architecture`](parked/experimental/vla-v2-architecture.md) | [`harness-architecture`](active/harness/harness-architecture.md) Tier 1 (teleop) **and** Tier 2 (bridge) shipped | Needs the action-labeled LeRobot v3 corpus (teleop primary, bridge supplement) before any VLA fine-tune is meaningful. Sim-first research arm; additive to v1. |
| [`vla-v2-map-conditioning`](parked/experimental/vla-v2-map-conditioning.md) | [`semantic-region-partition`](active/multi-room/semantic-region-partition.md) shipped **and** [`vla-v2-architecture`](parked/experimental/vla-v2-architecture.md) has a working training run | Audit-filed off the 2026-05-20 multi-room + clip-validation review. Decides how the v2 VLA consumes the map — picks one of {A: serialize the symbolic regions, B: consume the implicit memory map as consumer #2, C: no consumption} via a three-row ablation on cross-room mission success. |
| [`mujoco-warp-physics-backend-spike`](parked/experimental/mujoco-warp-physics-backend-spike.md) | Trigger: the PhysX-PGS roller fix shows limitations in a full-length training run / deployment, **or** Isaac Lab ships first-class Newton env integration (`isaaclab_newton` / `SimulationCfg` backend selector), **or** dedicated appetite to evaluate a physics-engine direction | Spun out of [`roller-contact-high-omega-bounce`](completed/roller-contact-high-omega-bounce.md). The env already ships Newton 1.0 + `mujoco_warp` (unintegrated). Time-boxed standalone spike: convert the Strafer USD via `mujoco_usd_converter`, run the max-yaw spin under `mujoco_warp`, compare roller-contact behavior to the PhysX reference, and file a go/no-go on a deeper backend migration. Don't pre-empt. |
| [`dynamic-region-granularity`](parked/multi-room/dynamic-region-granularity.md) | [`semantic-region-partition`](active/multi-room/semantic-region-partition.md) shipped **and** trigger: a real mission needs object/sub-region grounding v2's static regions can't express (see brief's "Trigger detail") | v3 escape valve — CLIO-style task-driven granularity ("Living room → Media area → remote" expands per mission). Filed off the PR #43 architecture review; replaces the rejected fixed `room → place → object` hierarchy. Don't pre-empt — v2 static regions may suffice. |
| [`implicit-memory-map`](parked/clip-validation/implicit-memory-map.md) | Trigger: [`cotrained-retrieval-augmented`](parked/clip-validation/cotrained-retrieval-augmented.md) reaches its Step B (cascade validator wants retrieval-augmented inference) **or** [`vla-v2-map-conditioning`](parked/experimental/vla-v2-map-conditioning.md) picks Option B. Now a **planned warm-map consumer** (single-home deployment); case-1 head prototypable without R1, case-2 head gated on R1. | The project's sub-symbolic implicit-mapping primitive (memory bank + cross-attention + RAG-aware training). Factored out of cotrained's Step B so it's built once and shared by both consumers. Filed off the PR #43 architecture review. |
| [`clip-multi-room-validator-remeasure`](parked/clip-validation/clip-multi-room-validator-remeasure.md) | [`autonomy-stack`](active/multi-room/autonomy-stack.md) shipped **and** [`validator-evaluation`](active/clip-validation/validator-evaluation.md) v1 shipped | Owns the per-leg / sub-goal **deviation contract** (scan=off, explore=disarmed, staging-hop=semantic-vs-hop-room, final=case-1+case-2) + re-runs the per-case ROC-AUC on multi-room data. Named in `MISSION_VALIDATION_ARCHITECTURE.md` §4.3. |
| [`vlm-grounding-finetune`](parked/clip-validation/vlm-grounding-finetune.md) | Trigger: the harness lands the first-class `observation.detections.*` column + `meta/detection_labels.json` (R1, Tier 2/3) | Adopts the orphaned Qwen2.5-VL grounding LoRA tooling onto the harness detections column. Serves multi-room `scan_for_target` / planner grounding **and** the validator's case-2 alternates. **Do not delete `bbox_extractor.py` — it is this brief's producer.** |
| [`harness-throughput-measurement`](parked/harness/harness-throughput-measurement.md) | Trigger: [`harness-architecture`](active/harness/harness-architecture.md) Tier 3 (scripted driver) about to commit to a `num_envs` target | Audit-filed: every parallel-env claim in the harness epic is asserted, not measured. Run this before the scripted driver picks a throughput acceptance bar. |
| [`cosmos-replay-perturbation`](parked/harness/cosmos-replay-perturbation.md) | Trigger: [`harness-architecture`](active/harness/harness-architecture.md) Tier 1 (teleop) shipped and teleop corpus ≥ 500 trajectories **and** NVIDIA Cosmos Predict / Transfer accessible on the DGX | Audit-filed: corpus multiplier via NVIDIA Cosmos world-model re-rendering (lighting / texture / weather variants per captured trajectory). Replaces / extends the design-doc's "replay-with-perturbation" item. |
| [`scene-provider-floor-sampler-cli`](parked/harness/scene-provider-floor-sampler-cli.md) | Trigger: first non-Infinigen scene source whose floor / structural prims aren't named `<room>_<i>_<j>_floor` wants `generate_scenes_metadata.py`'s auto floor-sampler instead of hand-authoring its manifest entry | Filed off PR #66 (`scene-provider-contract`) review. Parameterize `_FLOOR_NAME_RE` + `_ROOM_STRUCT_RE` behind `--floor-prim-pattern` / `--room-struct-pattern`, mirroring the `--ceiling-light-prim-pattern` work that shipped in #66. The contract's hand-author fallback (SCENE_PROVIDER_CONTRACT.md §c) means a second source isn't blocked; this removes the friction. |
| [`depth-ffv1-video-column`](parked/harness/depth-ffv1-video-column.md) | Sequenced after the R1 first-class `observation.detections.*` column lands (both edit `lerobot_writer.py` / `build_features`; two sequential PRs). Phase-1 spike (FFV1 `gray16le` bit-exact through PyAV on DGX **and** Jetson) gates the LeRobot integration. | Operator-committed depth-format migration: per-frame 16UC1 PNG sidecar → one lossless FFV1 `gray16le` stream per shard, registered as a first-class LeRobot video feature. Fixes the Tier-2/3 small-files read penalty **and** buys stock-loader ergonomics, accepting the LeRobot codec/feature-API coupling the PNG route avoided. Fallback if the spike fails: per-episode-packed sidecar (`(T,H,W)` uint16 per episode). |
| [`capture-debug-overhead-cam`](parked/harness/capture-debug-overhead-cam.md) | Trigger: a scripted/coverage capture run shows unexplained robot behavior (stuck / wedged / colliding with nothing visible / 0 usable episodes) whose cause needs eyes on the scene, not logs | Reuse `teleop_capture.py`'s overhead-viewport rig (`--hide-overhead` / `--overhead-regex`) behind a `--debug-overhead-cam` on the scripted/coverage capture entry, plus an optional periodic top-down video. Motivated by the `occupancy-interior-fidelity` skirting-board phantom-slab collider (invisible in RGB, obvious from overhead). Operator-only; zero corpus/throughput impact when off. |
| [`perception-exposure-finetune`](parked/harness/perception-exposure-finetune.md) | Trigger: the next Infinigen corpus regeneration lands (operator deferred to avoid tuning against soon-to-change scenes) | Closes the marginal crushed-black/clip gap left by the over-bright fix (#117, shipped): ceiling-on ambient-fill sweep (`rtx.sceneDb.ambientLightIntensity`) + seed5 bright-fixture clip decision + final band calibration. Also carries the "do Infinigen's `PointLampFactory`/`RectLight` emitters render well enough to drop `inject_ceiling_light_emitters`?" Kit check. Tools (`--render-carb`, `measure_perception_exposure.py`) already in place. |
| [`nav-stall-multilayer-watchdog`](parked/reliability/nav-stall-multilayer-watchdog.md) | Trigger: v1 stall watchdog from `progress-aware-nav-timeouts` produces real-world false-positives (cluttered-sim re-plans) or false-negatives (chassis wedge incident) | Filed-on-trigger sketch. Adds chassis-wedge + Nav2 recovery-rate signals on top of the v1 best-ever-distance watchdog. Don't pick up preemptively — v1 may be sufficient. |
| [`perception-side-bearing-service`](parked/reliability/perception-side-bearing-service.md) | Trigger: **the first** bearing-varying behavior (approach-from-angle, look-at-while-driving, face-target-while-translating, etc.) lands on the active roadmap | Filed-on-trigger refactor surfaced by `align-after-scan-grounding`'s shipped Option A. Start of the perception-owned geometry-primitives layer in `strafer_perception`. |
| [`d555-usb-dropout-framerate-collapse`](parked/reliability/d555-usb-dropout-framerate-collapse.md) | Trigger: real D555 connected to Jetson **and** either (a) multi-hour mission completed where dropouts could surface, or (b) the first tegra-xusb stall observed on real-robot bringup | Filed-on-trigger off the 2026-05-17 reliability audit. Adds `/perception/health` topic + framerate watchdog + executor gate. Not exercised in sim. |
| [`roboclaw-error-visibility-and-low-battery`](parked/reliability/roboclaw-error-visibility-and-low-battery.md) | Trigger: real-robot bringup begins (chassis powered and RoboClaws actually communicating over USB) | Filed-on-trigger off the 2026-05-17 reliability audit. Exposes CRC-error count + battery voltage + low-battery degraded mode. Not exercised in sim (`HARDWARE_PRESENT=false` bypasses driver). |
| [`imu-yaw-drift-no-magnetometer`](parked/reliability/imu-yaw-drift-no-magnetometer.md) | Trigger: real-robot multi-room mission shows yaw-drift between RTAB-Map closures > 2° p95 **or** RTAB-Map loop closure becomes unreliable on lab carpet | Filed-on-trigger investigation off the 2026-05-17 reliability audit. Decides between SLAM-anchored telemetry-only (option A) vs. external magnetometer add (option B) vs. VIO (option C). |
| [`executor-replan-on-deviation`](parked/reliability/executor-replan-on-deviation.md) | [`clip-multi-room-validator-remeasure`](parked/clip-validation/clip-multi-room-validator-remeasure.md) defines the deviation contract **and** [`validator-evaluation`](active/clip-validation/validator-evaluation.md) wires the validator | The "stop → re-plan" half of detect→stop→replan that `autonomy-stack` defers ("the mission fails — no automatic re-plan"). Routes a structured deviation reason to the compiler for a bounded re-plan (wrong_room → re-transit/explore from current pose; wrong_instance → bounded re-ground). Sibling of `executor-grounding-loss-mid-mission-recovery`. |
| [`robot_localization-ekf-fused-odom`](parked/reliability/robot_localization-ekf-fused-odom.md) | Trigger: DEPTH MVP deployment shows wheel-slip-driven failures (collision-recovery loops, low-friction wobble, sharp-turn over-rotation) that encoder-only `/strafer/odom` can't disambiguate. See brief's "Trigger detail." | Filed-on-trigger off the `observation-contract-cleanup` ship. EKF fusion of encoder + IMU + SLAM into `/odom_fused`; swaps `inference-package`'s `body_velocity_xy` source from `/strafer/odom` to `/odom_fused`. Don't pre-empt — encoder-only is the conventional choice and the slip-class failure may stay sub-threshold on real deployments. |
| [`llm-guided-frontier-gain`](parked/multi-room/llm-guided-frontier-gain.md) | [`observation-derived-room-state`](completed/observation-derived-room-state.md) shipped (no language-shaped frontier descriptions); v1 frontier primitive shipped in #40 | Extension to v1 frontier primitive — multiplies an LFG-style scalar LLM prior onto the geometric gain. `gain_weights.llm = 0.0` recovers v1 exactly. Cites LFG (arXiv:2310.10103) as the design precedent; CogNav-style state machine deferred to v3 ([`frontier-cognitive-fsm`](parked/multi-room/frontier-cognitive-fsm.md)). |
| [`frontier-cognitive-fsm`](parked/multi-room/frontier-cognitive-fsm.md) | [`llm-guided-frontier-gain`](parked/multi-room/llm-guided-frontier-gain.md) shipped **and** v2 shows a measurable plateau on long-horizon (≥ 3-room) missions per the brief's "Trigger detail" | v3 frontier-exploration upgrade — CogNav-style cognitive FSM (broad / contextual / identification / verification states) on top of v2's scalar prior. Filed-on-trigger: do not pick up preemptively. The detector, snapshot plumbing, and skill registration survive both v1→v2 and v2→v3; only the per-step loop rewrites. Cites CogNav, ICCV 2025. |
| [`room-state-runtime-ergonomics`](parked/multi-room/room-state-runtime-ergonomics.md) | Trigger: deployment regularly exceeds ~1K nodes (Finding A scaling) OR a real-robot CLIP late-bind failure surfaces (Finding B) | Two manager-internal ergonomic gaps surfaced by the `observation-derived-room-state` ship audit: `room_anchor`'s linear scan over all graph nodes (v1-acceptable; index follow-up at scale) and `RoomClassifier`'s sticky `enabled = False` latch (silent failure if CLIP late-binds). Both have clear filed-on-trigger conditions. |
| [`semantic-map-lifecycle-merge`](parked/multi-room/semantic-map-lifecycle-merge.md) | [`semantic-graph-loop-closure`](active/multi-room/semantic-graph-loop-closure.md) shipped (no merge candidates without it) | v2 room-state follow-up — replace v1's time-based `prune()` with hierarchical decay (recent / long-term layers + spatial pooling). Preserves static geometry across multi-day operation. Capacity-bounded by home area, not by time. |
| [`staging-hops-shadow-mode`](parked/multi-room/staging-hops-shadow-mode.md) | [`autonomy-stack`](active/multi-room/autonomy-stack.md) shipped (need the room-aware compiler to compare against) **and** [`planner-far-target-staging`](active/multi-room/planner-far-target-staging.md) shipped (need the far-target helper). | Migration step 2 from [§1.10.2](../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c) — populate `MissionIntent.staging_hops` from the LLM, log the agreement-rate against the compiler's plan, ship a weekly report. Compiler still ignores the field. The report is the trigger signal for `planner-scene-graph-expansion`. |
| [`planner-scene-graph-expansion`](parked/multi-room/planner-scene-graph-expansion.md) | [`staging-hops-shadow-mode`](parked/multi-room/staging-hops-shadow-mode.md) shipped AND its ≥ a week of shadow data shows the LLM's hops disagree with the compiler for reasons object poses / room inventories would fix (target disambiguation, intra-room landmark choice). See the "Trigger detail" section of the brief. | Extends `world_state` with `ObjectEntry` + per-room object inventories so the planner LLM has the spatial context to make `staging_hops` better than the Option C compiler. Prerequisite for promoting `staging_hops` from advisory to authoritative in [§1.10.2 step 3](../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c). |
| [`learned-spatial-encoder`](parked/multi-room/learned-spatial-encoder.md) | EITHER [`semantic-region-partition`](active/multi-room/semantic-region-partition.md)'s single `α` can't hold both open-plan and multi-bedroom splits / fails sim→real, OR [`semantic-graph-loop-closure`](active/multi-room/semantic-graph-loop-closure.md)'s raw-CLIP calibration can't meet the precision-recall floor / fails sim→real. See the brief's "Trigger detail." | v3 escape valve for BOTH v2 unsupervised mechanisms — one frozen DINOv2 trunk + place-recognition head (AnyLoc/SALAD; replaces raw-CLIP loop closure) + region head (learned partition; replaces HDBSCAN+`α`). The heads are functionally coupled (better place recognition → cleaner `same_place` edges → more aggressive region splits), which is why they're one brief. v2 mechanisms are the fallbacks. Collapsed `learned-region-head` + `learned-vpr-loop-closure` per the PR #43 architecture review. |
| [`tools-package-reorg`](parked/tooling/tools-package-reorg.md) | Land when no large `tools/`-touching PR is in flight — after `depth-ffv1-video-column` + `mission-generator`'s `build_mission_queue.py` settle | Split `strafer_lab/tools/` into purpose subpackages, **keeping the contract-side readers (`scene_metadata_reader`/`scene_connectivity`/`scene_paths`/`scene_labels`) separate from the Infinigen-specific parser (`infinigen_label_parser`, `scene_classes.room_struct_regex`)** so a future scene source has a clean home — mechanical move + import-path rewrite, no behavior change. Module table re-walked 2026-06-21 (`dataset_export.py` deleted). Conforms to the placement-rule amendment in `script-tool-subsystem-grouping`. |
| [`scene-contract-instance-discriminator`](parked/tooling/scene-contract-instance-discriminator.md) | Trigger: `mission-text-enrichment` picked up OR a second (non-Infinigen) scene source brought up | Add a source-agnostic `objects[].uid` so consumers stop reaching for the Infinigen `__spawn_asset_<N>_` prim token to tell same-label instances apart (`instance_id` is the factory-class id). Soft coupling today (the `(label, instance_id, prim_path)` fallback works); this removes it. Filed off the 2026-06-21 scene-source boundary audit. |
| [`infinigen-label-taxonomy`](parked/harness/infinigen-label-taxonomy.md) | Trigger: a detection-vocab / grounding-quality measurement shows `_infer_label`'s longest-tag fallback producing noisy / non-canonical labels | Comprehensive Infinigen factory-class → canonical-noun label map, replacing the hand-picked 8-category seed list + longest-tag fallback. Affects detection vocab + grounding quality. Filed off PR #96 (externalized `LABEL_CATEGORY_PRIORITY`). Composes with `mission-text-enrichment` (clean base label + descriptors). |
| [`distractor-asset-injection`](parked/harness/distractor-asset-injection.md) | Trigger: a real corpus run yields a goal-sparse scene (`< N` missions — the `objects=0 → 0 missions` incident is the filing motivation) OR a VLA/VLM measurement shows object-diversity saturation on the fixed base-scene set | Scatter labeled distractor/filler assets (curated pool) into Infinigen scenes via the Replicator object-based SDG path — navigability-aware placement reusing the occupancy seam, authored into `objects[]` + detections via the same `add_labels(...,"class")` path. Content-side (object-presence) DR sibling to `domain-randomization-audit`'s dynamics-side DR. **Diversity + sparse-goal fill, NOT disambiguation** (that's the generator REG + `mission-text-enrichment`). Showstopper = full Kit-bound per-variant re-process gated by the enforced occupancy-staleness guard → v1 scoped to the goal-sparse rescue. New scene-content source on the `SCENE_PROVIDER_CONTRACT`. |
| [`out-of-process-mission-text-llm`](parked/harness/out-of-process-mission-text-llm.md) | Trigger: LLM-quality mission text is wanted over oracle + templated, OR a grounded run needs `--use-planner-llm` / `--use-paraphrase-llm` without flipping the env's `cuda-bindings` pin | Non-blocking quality / deployability follow-up off `mission-generator`'s geometric-grounding pivot: that pivot took the *VL* model out of Kit, but the *text* LLM still loads `Qwen3-4B` in-process and re-introduces the torch-vs-Isaac-Sim `cuda-bindings` conflict. Move the waypoint + paraphrase passes behind the existing out-of-process `serve-planner` (extend it with text-generation endpoints; swap the in-process runners for HTTP-client runners). The v1 corpus ships fine on oracle + REG-templated text, so this is filed-on-trigger. Cross-lane: the new endpoints are `strafer_autonomy`-owned. |

---

## How to use this board

### As an agent (no specific brief assigned)

1. Scan **Ready to pick up** for your lane (or **Quick wins** if you
   have an hour).
2. Match priority + estimate to your session budget.
3. Read the brief end-to-end before starting; if its `Out of scope`
   or dependencies have shifted since the board was last updated,
   prefer the brief over the board.
4. If you want the feature-area view rather than the priority view,
   scan **By epic** for the relevant epic.

### As a contributor filing a new brief

1. Place the brief under the right `active/<epic>/` subdir (or
   `parked/<epic>/` if it's filed-on-trigger or blocked).
2. Add a row to **By epic** under the right epic.
3. Add a row to **Ready to pick up** under the right priority + lane
   (or **Parked** with the dependency named) in the same commit
   that files the brief.

### As a contributor picking up a brief

1. Move the row from **Ready to pick up** → **In flight** with the
   PR number, in the commit that opens the PR (or the first commit
   on the task branch if the PR comes later).
2. Un-parking? `git mv parked/<epic>/<brief>.md
   active/<epic>/<brief>.md` and update the brief's **By epic** row
   (state: parked → in flight) in the same commit.

### As a contributor shipping a brief

1. Move the brief into [`completed/`](completed/) (which stays
   flat) with the standard stamp (`**Status:** Shipped <date> in
   <commit> (<host>).` + `**PR:** <url>`).
2. Remove it from **In flight** and from its **By epic** row on
   this board.
3. If the brief's validation surfaced follow-ups, add them under
   **Ready to pick up** (or **Parked** if they have unshipped
   dependencies) in the same commit, in the appropriate
   `active/<epic>/` or `parked/<epic>/` subdir.
