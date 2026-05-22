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
| _None._ | | | |

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
| [`scene-connectivity-validation`](active/multi-room/scene-connectivity-validation.md) | P1 | active | DGX |
| [`room-state-eval-harness`](active/multi-room/room-state-eval-harness.md) | P2 | active | DGX |
| [`semantic-region-partition`](active/multi-room/semantic-region-partition.md) | P2 | active | DGX |
| [`query-room-by-text-v1`](active/multi-room/query-room-by-text-v1.md) | P2 | active | DGX |
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
| [`export-torchscript-depth`](active/trained-policy/export-torchscript-depth.md) | P2 | active | DGX |
| [`export-sidecar-training-preset`](active/trained-policy/export-sidecar-training-preset.md) | P3 | active | DGX |
| [`inference-package`](active/trained-policy/inference-package.md) | P1 | active | Jetson |
| [`recurrent-state-contract`](active/trained-policy/recurrent-state-contract.md) | P1 | active | Either |
| [`observation-contract-cleanup`](active/trained-policy/observation-contract-cleanup.md) | P1 | active | DGX |
| [`domain-randomization-audit`](active/trained-policy/domain-randomization-audit.md) | P1 | active | DGX |
| [`goal-noise-training`](active/trained-policy/goal-noise-training.md) | P2 | active | DGX |
| [`subgoal-env`](active/trained-policy/subgoal-env.md) | P2 | active | DGX |
| [`hybrid-mode`](parked/trained-policy/hybrid-mode.md) | P3 | parked | Jetson |
| [`rl-global-nav2-local`](parked/trained-policy/rl-global-nav2-local.md) | P3 | parked | Either |

### Harness & training data

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`teleop-driver`](active/harness/teleop-driver.md) | P1 | active | DGX |
| [`behavior-cloning-data-expansion`](active/harness/behavior-cloning-data-expansion.md) | P2 | active | DGX |
| [`mission-generator`](active/harness/mission-generator.md) | P2 | active | DGX |
| [`trajectory-first-captioning`](active/harness/trajectory-first-captioning.md) | P2 | active | DGX |
| [`oracle-driver`](parked/harness/oracle-driver.md) | P3 | parked | DGX |
| [`harness-throughput-measurement`](parked/harness/harness-throughput-measurement.md) | P2 | parked | DGX |
| [`output-format-alignment`](parked/harness/output-format-alignment.md) | P2 | parked | DGX |
| [`cosmos-replay-perturbation`](parked/harness/cosmos-replay-perturbation.md) | P3 | parked | DGX |

### CLIP mid-mission validation

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`validator-evaluation`](active/clip-validation/validator-evaluation.md) | P1 | active | Either |
| [`backbone-bakeoff`](parked/clip-validation/backbone-bakeoff.md) | P2 | parked | DGX |
| [`cotrained-retrieval-augmented`](parked/clip-validation/cotrained-retrieval-augmented.md) | P3 | parked | DGX |
| [`implicit-memory-map`](parked/clip-validation/implicit-memory-map.md) | P3 | parked | DGX |

### Sim performance

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`isaac-sim-rt-2-default-renderer`](active/sim-performance/isaac-sim-rt-2-default-renderer.md) | P2 | active | DGX |
| [`bridge-throughput-toward-25hz`](active/sim-performance/bridge-throughput-toward-25hz.md) | P2 | active | DGX |

### Reliability (nav + executor + refactors)

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`executor-cancel-mid-motion-cmd-vel-zero`](active/reliability/executor-cancel-mid-motion-cmd-vel-zero.md) | P1 | active | Jetson |
| [`executor-prefer-rotate-then-translate`](active/reliability/executor-prefer-rotate-then-translate.md) | P2 | active | Jetson |
| [`rtabmap-cold-start-determinism`](active/reliability/rtabmap-cold-start-determinism.md) | P2 | active | Jetson |
| [`executor-grounding-loss-mid-mission-recovery`](active/reliability/executor-grounding-loss-mid-mission-recovery.md) | P2 | active | Jetson |
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

### Tooling & ops

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`unify-test-targets-and-ci`](active/tooling/unify-test-targets-and-ci.md) | P3 | active | Either |
| [`windows-workstation-bringup`](active/tooling/windows-workstation-bringup.md) | P2 | active | DGX |

### Experimental (long-horizon bets)

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`vla-v2-architecture`](parked/experimental/vla-v2-architecture.md) | P3 | parked | Either |
| [`vla-v2-map-conditioning`](parked/experimental/vla-v2-map-conditioning.md) | P3 | parked | DGX |

### Investigations (measurement / knowledge work)

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`next-integration-round`](active/investigations/next-integration-round.md) | P1 | active | Either |
| [`real-d555-depth-range-survey`](active/investigations/real-d555-depth-range-survey.md) | P2 | active | Jetson |
| [`training-throughput-profile-and-investigate`](active/investigations/training-throughput-profile-and-investigate.md) | P2 | active | DGX |
| [`defm-preprocess-antialias-audit`](active/investigations/defm-preprocess-antialias-audit.md) | P3 | active | DGX |

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
| [`inference-package`](active/trained-policy/inference-package.md) | Jetson | L (~1.5 wk) | DEPTH MVP — `strafer_direct` mode with the trained ProcRoom-Depth policy. Phases 1–4 land without a deployable checkpoint; Phase 5 gates on DGX-side export+training. Architectural answer to MPPI's plateau |
| [`validator-evaluation`](active/clip-validation/validator-evaluation.md) | Either | L | Wire the orphaned `SemanticMapManager` + `BackgroundMapper` + `TransitMonitor` path into the production executor and measure pre-registered TPR/FPR/time-to-decision on harness output. Gating brief for `MISSION_VALIDATION_ARCHITECTURE.md` §4 staged plan. Filed off `mid-mission-validation-investigation` ship. |
| [`teleop-driver`](active/harness/teleop-driver.md) | DGX | M | Gamepad teleop entry point for in-process Isaac Lab data capture. Bypasses MPPI / Nav2 / planner; reuses `collect_demos.py` mapping; emits the canonical harness schema. Unblocks v1 measurement (CLIP cascade eval + co-trained validator) and v2 VLA training data without depending on bridge perf. |
| [`autonomy-stack`](active/multi-room/autonomy-stack.md) | Either | M | Lifts §1.10.1's multi-room deferral. Stored-map fallback in `scan_for_target` + planner transit-step emission + plan-compiler updates. Blocks on `observation-derived-room-state` and `frontier-exploration-primitive` (`planner-architecture-alignment` shipped in #36 as Option C). |
| [`scene-connectivity-validation`](active/multi-room/scene-connectivity-validation.md) | DGX | S | Verified-and-enriched `connectivity[]` block + door-open guarantee. Sim/harness-only; runtime equivalent is `observation-derived-room-state`. |
| [`recurrent-state-contract`](active/trained-policy/recurrent-state-contract.md) | Either | S–M | End-to-end spec for hidden-state shape, reset semantics, thread-safety across train/export/inference. Three existing recurrent briefs each describe their side; this brief pins the contract at the seams. Filed off the 2026-05-15 trained-policy audit. |
| [`observation-contract-cleanup`](active/trained-policy/observation-contract-cleanup.md) | DGX | S–M | Re-implement `body_velocity_xy` as encoder-derived FK so the sim signal chain matches what the real robot computes via `/strafer/odom`. Closes a silent sim-to-real bug before the DEPTH MVP ships. Filed off the 2026-05-15 trained-policy audit. |
| [`domain-randomization-audit`](active/trained-policy/domain-randomization-audit.md) | DGX | M | Bench-measure real-chassis variability (mass, battery, D555 latency, Jetson jitter) and widen REAL_ROBOT_CONTRACT to match. Resume-train DEPTH baseline against the audited DR. Filed off the 2026-05-15 trained-policy audit. |
| [`executor-cancel-mid-motion-cmd-vel-zero`](active/reliability/executor-cancel-mid-motion-cmd-vel-zero.md) | Jetson | S (~½ d) | Cancel-correctness bug — `rotate_in_place` doesn't observe a cancel event, so canceling a mid-rotation mission ignores the cancel until tolerance or timeout. Real-robot blocker. Filed off the 2026-05-17 reliability audit. |

### P2 — medium priority

#### DGX lane

| Brief | Estimate | Note |
|---|---|---|
| [`isaac-sim-rt-2-default-renderer`](active/sim-performance/isaac-sim-rt-2-default-renderer.md) | S | Flip default renderer to Real-Time 2.0 + 4× FPS multiplier + Performance mode; re-measure bridge perf |
| [`planner-rotate-direction-prompt`](active/reliability/planner-rotate-direction-prompt.md) | S | Quick win — prompt edit |
| [`goal-noise-training`](active/trained-policy/goal-noise-training.md) | M | Targeted DEPTH-baseline training pass with goal-position noise; gates VLM-grounded mission quality for `strafer_direct` |
| [`behavior-cloning-data-expansion`](active/harness/behavior-cloning-data-expansion.md) | M–L | Per-tick capture + depth + actions + time alignment + paraphrase + hard-negative injection. Driver-agnostic schema; bridge-driver upgrades ship in this brief. |
| [`planner-far-target-staging`](active/multi-room/planner-far-target-staging.md) | M–L | World-state schema + planner prompt |
| [`subgoal-env`](active/trained-policy/subgoal-env.md) | L (~1.5–2 wk) | New training env for `NOCAM_SUBGOAL` — sim-internal path planner + SubgoalCommand + path-tracking rewards + termination + training run. Unblocks hybrid mode |
| [`windows-workstation-bringup`](active/tooling/windows-workstation-bringup.md) | L (~1 wk) | Investigation + port — run `make sim-bridge` on Windows (RTX 4080) against the Jetson stack. Isaac Lab 3 Windows support is experimental; phase the feasibility spike before committing to a full port |
| [`bridge-throughput-toward-25hz`](active/sim-performance/bridge-throughput-toward-25hz.md) | M | Follow-up to `async-camera-publishers`. Lift the bridge toward the predicted 25 Hz ceiling. |
| [`mission-generator`](active/harness/mission-generator.md) | L | Free-text mission generator with LLM-emitted waypoints (multi-room default). Canonical mission queue source for teleop and oracle drivers. Blocks on `scene-connectivity-validation`. |
| [`trajectory-first-captioning`](active/harness/trajectory-first-captioning.md) | M–L | Speaker-model post-hoc captioning regime. Random-A→B drivers + Qwen2.5-VL-7B speaker → instructive-voice mission text + synthesized hard negatives. |
| [`training-throughput-profile-and-investigate`](active/investigations/training-throughput-profile-and-investigate.md) | S–M | Phase profiler in the training loop; files follow-up briefs from results. |
| [`export-torchscript-depth`](active/trained-policy/export-torchscript-depth.md) | S–M | DEPTH TorchScript export on real checkpoints — work around DeFM `BiFPN`'s un-scriptable `sum(generator)` via traced backbone. Sibling of [`export-onnx-depth`](completed/export-onnx-depth.md); ONNX already ships, this closes the redundant TorchScript path. |
| [`room-state-eval-harness`](active/multi-room/room-state-eval-harness.md) | M | v2 room-state — measurement harness for cluster purity / label precision / time-to-converge / connectivity P-R on a fixed multi-room scene set (incl. open-plan + multi-bedroom adversarials). Ships first; also the training corpus for the escape valves. Blocks pickup on `observation-derived-room-state` (shipped). |
| [`semantic-region-partition`](active/multi-room/semantic-region-partition.md) | M | **The v2 quality work** — feature+space HDBSCAN clustering + open-vocab labels, replacing v1's greedy-modularity + 7-class argmax. One `α` knob (SOTA-aligned, ConceptGraphs / HOV-SG shape); no training. Handles open-plan + multi-bedroom by construction. `RoomEntry` shape preserved (+`uncertainty`). |
| [`query-room-by-text-v1`](active/multi-room/query-room-by-text-v1.md) | S | Open-vocab room-state queries on the existing OpenCLIP ViT-B/32 backbone. Adds `SemanticMapManager.query_room_by_text(text)`; the query-side half of v2's open-vocab move, ships on the v1 backbone without waiting for `semantic-region-partition` or `backbone-bakeoff`. |

#### Jetson lane

| Brief | Estimate | Note |
|---|---|---|
| [`grounding-publisher-extraction`](active/reliability/grounding-publisher-extraction.md) | S | Quick win — pure refactor follow-up to `vlm-bbox-overlay`; extracts the viz publishers out of `JetsonRosClient` |
| [`real-d555-depth-range-survey`](active/investigations/real-d555-depth-range-survey.md) | S–M | Investigation — bench measurement + write-up |
| [`rtabmap-cold-start-determinism`](active/reliability/rtabmap-cold-start-determinism.md) | M | Cold-start signature on populated DB: `Not found word N` burst + `Increment map id to 4`; triage bridge-teleport vs Mem/* config + ship the chosen disposition. After audit: A2 recommended — flip `localization:=true` default when populated DB exists. |
| [`executor-prefer-rotate-then-translate`](active/reliability/executor-prefer-rotate-then-translate.md) | M | Decompose non-cardinal translations into rotate-to-face + forward translate at the executor layer; preserves cardinal strafe |
| [`rotate-in-place-large-angle-correctness`](active/reliability/rotate-in-place-large-angle-correctness.md) | S–M | `rotate_in_place` closes the loop on a single normalized target yaw: `rotate 360` no-ops (2π→0 target) and `>180°` takes the short way. Track accumulated traversal instead. Surfaced in PR #45 e2e (rotate 360 vs 180 at RTF≈0.085). |
| [`executor-grounding-loss-mid-mission-recovery`](active/reliability/executor-grounding-loss-mid-mission-recovery.md) | M | `_navigate_via_staging` re-grounding failure terminates immediately. Add mini-scan + semantic-map fallback with bounded recovery budget. Filed off the 2026-05-17 reliability audit. |
| [`executor-slam-tracking-precheck-mid-mission`](active/reliability/executor-slam-tracking-precheck-mid-mission.md) | S–M | Executor never queries `check_slam_tracking()`; silent failure when RTAB-Map loses tracking mid-mission. Add bounded precheck before each motion step. Filed off the 2026-05-17 reliability audit. |
| [`verify-arrival-occlusion-robustness`](active/reliability/verify-arrival-occlusion-robustness.md) | S–M | `_verify_arrival` false-negatives under partial occlusion. Add multi-frame voting + tilt-recovery + `arrival_occluded` soft-failure code. Filed off the 2026-05-17 reliability audit. |

#### Either lane

| Brief | Estimate | Note |
|---|---|---|
| _None currently._ | | |

### P3 — pickable, low priority

| Brief | Owner | Estimate | Note |
|---|---|---|---|
| [`unify-test-targets-and-ci`](active/tooling/unify-test-targets-and-ci.md) | Either | M | Makefile unification + stretch CI workflow. Doesn't block features; bumps to P2 once a second drift incident shows up. |
| [`export-sidecar-training-preset`](active/trained-policy/export-sidecar-training-preset.md) | DGX | S | Sidecar `training_preset` records the configclass name instead of the rsl_rl preset variable; cosmetic but the field is operator-facing. Filed off [`export-onnx-depth`](completed/export-onnx-depth.md). |
| [`defm-preprocess-antialias-audit`](active/investigations/defm-preprocess-antialias-audit.md) | DGX | S–M | Measure projection-space delta between training-time DeFM antialiased preprocessing and the deployment ONNX-safe non-antialiased version, then decide alignment (leave / align deploy / align training). Filed off [`export-onnx-depth`](completed/export-onnx-depth.md). |
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
| [`hybrid-mode`](parked/trained-policy/hybrid-mode.md) | [`inference-package`](active/trained-policy/inference-package.md) (Jetson) **and** [`subgoal-env`](active/trained-policy/subgoal-env.md) (DGX) both shipped | Hybrid backend extends the inference package's runtime AND consumes the `NOCAM_SUBGOAL` checkpoint produced by the env brief. The two prerequisites can run in parallel since they're cross-lane |
| [`rl-global-nav2-local`](parked/trained-policy/rl-global-nav2-local.md) | Trigger: first end-to-end deployment of `strafer_direct` (DEPTH MVP) or `hybrid_nav2_strafer` reveals that local-control RL is insufficient for VLM-grounded missions and the global-plan layer is the issue | Alternative architecture corner: RL as global waypoint planner, Nav2 as local controller. Filed off the 2026-05-15 trained-policy audit. Don't pick up preemptively — needs deployment evidence first |
| [`cotrained-retrieval-augmented`](parked/clip-validation/cotrained-retrieval-augmented.md) | [`validator-evaluation`](active/clip-validation/validator-evaluation.md) shipped + [`trajectory-first-captioning`](active/harness/trajectory-first-captioning.md) shipped (provides the speaker corpus) **and** [`backbone-bakeoff`](parked/clip-validation/backbone-bakeoff.md) shipped (selects the backbone) | Replaces the retired `learned-mid-mission-validator` as the cascade-improvement path. The co-training step and retrieval-augmented step compound; both compose with the existing cascade. |
| [`backbone-bakeoff`](parked/clip-validation/backbone-bakeoff.md) | [`validator-evaluation`](active/clip-validation/validator-evaluation.md) shipped | Audit-filed off the 2026-05-15 clip-validation review. Measures DINOv3-S / SigLIP-2-Base / MobileCLIP-2-S head-to-head against the v1 OpenCLIP ViT-B/32 baseline on the same eval set, so the cascade-improvement and v2 VLA briefs inherit a backbone with the alternative-considered trail in writing rather than defaulting to a 2021-era ViT-B/32. |
| [`vla-v2-architecture`](parked/experimental/vla-v2-architecture.md) | [`teleop-driver`](active/harness/teleop-driver.md) **and** [`behavior-cloning-data-expansion`](active/harness/behavior-cloning-data-expansion.md) shipped | Needs the action-labeled corpus (teleop primary, bridge supplement) before any VLA fine-tune is meaningful. Sim-first research arm; additive to v1. |
| [`vla-v2-map-conditioning`](parked/experimental/vla-v2-map-conditioning.md) | [`semantic-region-partition`](active/multi-room/semantic-region-partition.md) shipped **and** [`vla-v2-architecture`](parked/experimental/vla-v2-architecture.md) has a working training run | Audit-filed off the 2026-05-20 multi-room + clip-validation review. Decides how the v2 VLA consumes the map — picks one of {A: serialize the symbolic regions, B: consume the implicit memory map as consumer #2, C: no consumption} via a three-row ablation on cross-room mission success. |
| [`dynamic-region-granularity`](parked/multi-room/dynamic-region-granularity.md) | [`semantic-region-partition`](active/multi-room/semantic-region-partition.md) shipped **and** trigger: a real mission needs object/sub-region grounding v2's static regions can't express (see brief's "Trigger detail") | v3 escape valve — CLIO-style task-driven granularity ("Living room → Media area → remote" expands per mission). Filed off the PR #43 architecture review; replaces the rejected fixed `room → place → object` hierarchy. Don't pre-empt — v2 static regions may suffice. |
| [`implicit-memory-map`](parked/clip-validation/implicit-memory-map.md) | Trigger: [`cotrained-retrieval-augmented`](parked/clip-validation/cotrained-retrieval-augmented.md) reaches its Step B (cascade validator wants retrieval-augmented inference) **or** [`vla-v2-map-conditioning`](parked/experimental/vla-v2-map-conditioning.md) picks Option B | The project's sub-symbolic implicit-mapping primitive (memory bank + cross-attention + RAG-aware training). Factored out of cotrained's Step B so it's built once and shared by both consumers. Filed off the PR #43 architecture review. |
| [`oracle-driver`](parked/harness/oracle-driver.md) | [`subgoal-env`](active/trained-policy/subgoal-env.md) shipped (provides NoCam waypoint-follower checkpoint) **and** trigger: teleop throughput is the binding scale constraint for VLA training (see brief) | Filed-on-trigger sketch. Don't pick up preemptively. |
| [`harness-throughput-measurement`](parked/harness/harness-throughput-measurement.md) | Trigger: [`oracle-driver`](parked/harness/oracle-driver.md) or [`trajectory-first-captioning`](active/harness/trajectory-first-captioning.md) about to commit to a `num_envs` target | Audit-filed: every parallel-env claim in the harness epic is asserted, not measured. Run this before either scale-out brief picks a throughput acceptance bar. |
| [`output-format-alignment`](parked/harness/output-format-alignment.md) | Trigger: first downstream training brief about to consume the harness corpus, OR the harness about to ship the first 1k+ trajectories | Audit-filed: the current JSONL output schema doesn't match what GR00T / OpenVLA / π0 / Octo consume natively (LeRobot v2 / RLDS / robomimic HDF5). Pick the canonical format deliberately before the corpus grows. |
| [`cosmos-replay-perturbation`](parked/harness/cosmos-replay-perturbation.md) | Trigger: teleop corpus ≥ 500 trajectories **and** NVIDIA Cosmos Predict / Transfer accessible on the DGX | Audit-filed: corpus multiplier via NVIDIA Cosmos world-model re-rendering (lighting / texture / weather variants per captured trajectory). Replaces / extends the design-doc's "replay-with-perturbation" item. |
| [`nav-stall-multilayer-watchdog`](parked/reliability/nav-stall-multilayer-watchdog.md) | Trigger: v1 stall watchdog from `progress-aware-nav-timeouts` produces real-world false-positives (cluttered-sim re-plans) or false-negatives (chassis wedge incident) | Filed-on-trigger sketch. Adds chassis-wedge + Nav2 recovery-rate signals on top of the v1 best-ever-distance watchdog. Don't pick up preemptively — v1 may be sufficient. |
| [`perception-side-bearing-service`](parked/reliability/perception-side-bearing-service.md) | Trigger: **the first** bearing-varying behavior (approach-from-angle, look-at-while-driving, face-target-while-translating, etc.) lands on the active roadmap | Filed-on-trigger refactor surfaced by `align-after-scan-grounding`'s shipped Option A. Start of the perception-owned geometry-primitives layer in `strafer_perception`. |
| [`d555-usb-dropout-framerate-collapse`](parked/reliability/d555-usb-dropout-framerate-collapse.md) | Trigger: real D555 connected to Jetson **and** either (a) multi-hour mission completed where dropouts could surface, or (b) the first tegra-xusb stall observed on real-robot bringup | Filed-on-trigger off the 2026-05-17 reliability audit. Adds `/perception/health` topic + framerate watchdog + executor gate. Not exercised in sim. |
| [`roboclaw-error-visibility-and-low-battery`](parked/reliability/roboclaw-error-visibility-and-low-battery.md) | Trigger: real-robot bringup begins (chassis powered and RoboClaws actually communicating over USB) | Filed-on-trigger off the 2026-05-17 reliability audit. Exposes CRC-error count + battery voltage + low-battery degraded mode. Not exercised in sim (`HARDWARE_PRESENT=false` bypasses driver). |
| [`imu-yaw-drift-no-magnetometer`](parked/reliability/imu-yaw-drift-no-magnetometer.md) | Trigger: real-robot multi-room mission shows yaw-drift between RTAB-Map closures > 2° p95 **or** RTAB-Map loop closure becomes unreliable on lab carpet | Filed-on-trigger investigation off the 2026-05-17 reliability audit. Decides between SLAM-anchored telemetry-only (option A) vs. external magnetometer add (option B) vs. VIO (option C). |
| [`llm-guided-frontier-gain`](parked/multi-room/llm-guided-frontier-gain.md) | [`observation-derived-room-state`](completed/observation-derived-room-state.md) shipped (no language-shaped frontier descriptions); v1 frontier primitive shipped in #40 | Extension to v1 frontier primitive — multiplies an LFG-style scalar LLM prior onto the geometric gain. `gain_weights.llm = 0.0` recovers v1 exactly. Cites LFG (arXiv:2310.10103) as the design precedent; CogNav-style state machine deferred to v3 ([`frontier-cognitive-fsm`](parked/multi-room/frontier-cognitive-fsm.md)). |
| [`frontier-cognitive-fsm`](parked/multi-room/frontier-cognitive-fsm.md) | [`llm-guided-frontier-gain`](parked/multi-room/llm-guided-frontier-gain.md) shipped **and** v2 shows a measurable plateau on long-horizon (≥ 3-room) missions per the brief's "Trigger detail" | v3 frontier-exploration upgrade — CogNav-style cognitive FSM (broad / contextual / identification / verification states) on top of v2's scalar prior. Filed-on-trigger: do not pick up preemptively. The detector, snapshot plumbing, and skill registration survive both v1→v2 and v2→v3; only the per-step loop rewrites. Cites CogNav, ICCV 2025. |
| [`room-state-runtime-ergonomics`](parked/multi-room/room-state-runtime-ergonomics.md) | Trigger: deployment regularly exceeds ~1K nodes (Finding A scaling) OR a real-robot CLIP late-bind failure surfaces (Finding B) | Two manager-internal ergonomic gaps surfaced by the `observation-derived-room-state` ship audit: `room_anchor`'s linear scan over all graph nodes (v1-acceptable; index follow-up at scale) and `RoomClassifier`'s sticky `enabled = False` latch (silent failure if CLIP late-binds). Both have clear filed-on-trigger conditions. |
| [`semantic-map-lifecycle-merge`](parked/multi-room/semantic-map-lifecycle-merge.md) | [`semantic-graph-loop-closure`](active/multi-room/semantic-graph-loop-closure.md) shipped (no merge candidates without it) | v2 room-state follow-up — replace v1's time-based `prune()` with hierarchical decay (recent / long-term layers + spatial pooling). Preserves static geometry across multi-day operation. Capacity-bounded by home area, not by time. |
| [`staging-hops-shadow-mode`](parked/multi-room/staging-hops-shadow-mode.md) | [`autonomy-stack`](active/multi-room/autonomy-stack.md) shipped (need the room-aware compiler to compare against) **and** [`planner-far-target-staging`](active/multi-room/planner-far-target-staging.md) shipped (need the far-target helper). | Migration step 2 from [§1.10.2](../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c) — populate `MissionIntent.staging_hops` from the LLM, log the agreement-rate against the compiler's plan, ship a weekly report. Compiler still ignores the field. The report is the trigger signal for `planner-scene-graph-expansion`. |
| [`planner-scene-graph-expansion`](parked/multi-room/planner-scene-graph-expansion.md) | [`staging-hops-shadow-mode`](parked/multi-room/staging-hops-shadow-mode.md) shipped AND its ≥ a week of shadow data shows the LLM's hops disagree with the compiler for reasons object poses / room inventories would fix (target disambiguation, intra-room landmark choice). See the "Trigger detail" section of the brief. | Extends `world_state` with `ObjectEntry` + per-room object inventories so the planner LLM has the spatial context to make `staging_hops` better than the Option C compiler. Prerequisite for promoting `staging_hops` from advisory to authoritative in [§1.10.2 step 3](../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c). |
| [`learned-spatial-encoder`](parked/multi-room/learned-spatial-encoder.md) | EITHER [`semantic-region-partition`](active/multi-room/semantic-region-partition.md)'s single `α` can't hold both open-plan and multi-bedroom splits / fails sim→real, OR [`semantic-graph-loop-closure`](active/multi-room/semantic-graph-loop-closure.md)'s raw-CLIP calibration can't meet the precision-recall floor / fails sim→real. See the brief's "Trigger detail." | v3 escape valve for BOTH v2 unsupervised mechanisms — one frozen DINOv2 trunk + place-recognition head (AnyLoc/SALAD; replaces raw-CLIP loop closure) + region head (learned partition; replaces HDBSCAN+`α`). The heads are functionally coupled (better place recognition → cleaner `same_place` edges → more aggressive region splits), which is why they're one brief. v2 mechanisms are the fallbacks. Collapsed `learned-region-head` + `learned-vpr-loop-closure` per the PR #43 architecture review. |

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
