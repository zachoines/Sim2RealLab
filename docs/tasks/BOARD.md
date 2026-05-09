# Task board

Glanceable index of `docs/tasks/`. Pick from **Ready to pick up** by
matching your lane and session budget.

This file is updated **in the same commit that files / picks up /
ships a brief** — same maintenance contract as moving briefs to
[`completed/`](completed/). If the board and a brief disagree, the
brief wins; the board is a derived index, not the source of truth.

For brief format, authoring rules, and the agent-launcher template,
see [`README.md`](README.md). For lane definitions, see
[`context/ownership-boundaries.md`](context/ownership-boundaries.md).

---

## In flight

Open PRs — don't pick these up. Empty when no brief is being
executed (briefs are removed from this section in the same PR
that ships them; see "Shipping a brief: order of operations" in
[`README.md`](README.md)).

| Brief | Owner | PR | State |
|---|---|---|---|
| [`plan-compiler-skill-timeouts`](active/plan-compiler-skill-timeouts.md) | DGX | — | Branch `task/plan-compiler-skill-timeouts` pushed; follow-up brief queued; awaiting DGX implementation of the compiler-side change |

---

## Ready to pick up

Grouped by priority tier, then by lane. Within each cell the rough
order is "smallest / least-blocking first," but pick what fits your
session.

### P1 — high priority

| Brief | Owner | Estimate | Note |
|---|---|---|---|
| [`next-integration-round`](active/next-integration-round.md) | Either | M–L | Full end-to-end sim-in-the-loop run against `INTEGRATION_SIM_IN_THE_LOOP.md`; gating signal that bridge + autonomy + VLM/CLIP compose end-to-end |
| [`async-camera-publishers`](active/async-camera-publishers.md) | DGX | L (~3–5 d) | DGX bridge perf — closes the `OnPlaybackTick` gap (camera publish off Kit's main loop) |
| [`strafer-inference-package`](active/strafer-inference-package.md) | Jetson | L (~1.5 wk) | DEPTH MVP — `strafer_direct` mode with the trained ProcRoom-Depth policy. Phases 1–4 land without a deployable checkpoint; Phase 5 gates on DGX-side export+training. Architectural answer to MPPI's plateau |
| [`policy-export-onnx-depth`](active/policy-export-onnx-depth.md) | DGX | M (~1–2 d) | Implement `_OnnxDepthGRUModel` so DEPTH gets ONNX export. Unblocks the Jetson TRT-EP latency target. Filed off `policy-export-tooling` ship. |
| [`policy-loader-recurrent-state`](active/policy-loader-recurrent-state.md) | Either | S–M (~1 d) | Extend `strafer_shared.policy_interface.load_policy()` so recurrent artifacts expose `.reset()` and ONNX hidden state threads correctly. Prerequisite for stateful DEPTH inference on Jetson. |
| [`clip-mid-mission-validator-evaluation`](active/clip-mid-mission-validator-evaluation.md) | Either | L | Wire the orphaned `SemanticMapManager` + `BackgroundMapper` + `TransitMonitor` path into the production executor and measure pre-registered TPR/FPR/time-to-decision on harness output. Gating brief for `MISSION_VALIDATION_ARCHITECTURE.md` §4 staged plan. Filed off `mid-mission-validation-investigation` ship. |
| [`harness-teleop-driver`](active/harness-teleop-driver.md) | DGX | M | Gamepad teleop entry point for in-process Isaac Lab data capture. Bypasses MPPI / Nav2 / planner; reuses `collect_demos.py` mapping; emits the canonical harness schema. Unblocks v1 measurement (clip-eval, learned-validator) and v2 VLA training data without depending on bridge perf. |
| [`multi-room-autonomy-stack`](active/multi-room-autonomy-stack.md) | Either | M | Lifts §1.10.1's multi-room deferral. Stored-map fallback in `scan_for_target` + planner transit-step emission + plan-compiler updates. Required for multi-room as the MVP default. |
| [`multi-room-scene-connectivity-validation`](active/multi-room-scene-connectivity-validation.md) | DGX | S | Computes room connectivity at scene-gen time + verifies door-open default in `prep_room_usds.py`. Hard prerequisite for `multi-room-autonomy-stack` and `harness-mission-generator`. |

### P2 — medium priority

#### DGX lane

| Brief | Estimate | Note |
|---|---|---|
| [`d555-distortion-model-explicit`](active/d555-distortion-model-explicit.md) | S | Quick win |
| [`planner-rotate-direction-prompt`](active/planner-rotate-direction-prompt.md) | S | Quick win — prompt edit |
| [`policy-goal-noise-training`](active/policy-goal-noise-training.md) | M | Targeted DEPTH-baseline training pass with goal-position noise; gates VLM-grounded mission quality for `strafer_direct` |
| [`harness-behavior-cloning-data-expansion`](active/harness-behavior-cloning-data-expansion.md) | M–L | Per-tick capture + depth + actions + time alignment + paraphrase + hard-negative injection. Driver-agnostic schema; bridge-driver upgrades ship in this brief. |
| [`planner-far-target-staging`](active/planner-far-target-staging.md) | M–L | World-state schema + planner prompt |
| [`strafer-lab-subgoal-env`](active/strafer-lab-subgoal-env.md) | L (~1.5–2 wk) | New training env for `NOCAM_SUBGOAL` — sim-internal path planner + SubgoalCommand + path-tracking rewards + termination + training run. Unblocks hybrid mode |

#### Jetson lane

| Brief | Estimate | Note |
|---|---|---|
| [`rotate-in-place-sim-clock-deadline`](active/rotate-in-place-sim-clock-deadline.md) | S | Quick win |
| [`vlm-bbox-overlay`](active/vlm-bbox-overlay.md) | S | Quick win — Foxglove debug overlay |
| [`align-after-scan-grounding`](active/align-after-scan-grounding.md) | S–M | Executor + plan_compiler tweak |
| [`real-d555-depth-range-survey`](active/real-d555-depth-range-survey.md) | S–M | Investigation — bench measurement + write-up |
| [`nav2-startup-unknown-donut-path-noise`](active/nav2-startup-unknown-donut-path-noise.md) | M | Planner-side fix for the kinky-path-through-camera-donut symptom |

#### Either lane

| Brief | Estimate | Note |
|---|---|---|
| _None currently._ | | |

### P3 — has dependencies

| Brief | Owner | Estimate | Blocks on |
|---|---|---|---|
| [`strafer-inference-hybrid-mode`](active/strafer-inference-hybrid-mode.md) | Jetson | M (~3–4 d) | [`strafer-inference-package`](active/strafer-inference-package.md) shipped + [`strafer-lab-subgoal-env`](active/strafer-lab-subgoal-env.md) shipped (the latter produces the trained `NOCAM_SUBGOAL` checkpoint) |
| [`clip-cotrained-retrieval-augmented`](active/clip-cotrained-retrieval-augmented.md) | DGX | XL | Cascade-improvements research follow-up. A co-training step fine-tunes CLIP with the trajectory-first speaker model; a retrieval-augmented inference step adds cross-attention over the SemanticMapManager memory primitive. Filed-on-trigger after [`clip-mid-mission-validator-evaluation`](active/clip-mid-mission-validator-evaluation.md) ships. |
| [`strafer-vla-v2-architecture`](active/strafer-vla-v2-architecture.md) | Either | XL | [`harness-teleop-driver`](active/harness-teleop-driver.md) (primary data path) + [`harness-behavior-cloning-data-expansion`](active/harness-behavior-cloning-data-expansion.md) (schema) shipped. Sim-first research arm; additive to v1, not replacing it |
| [`harness-oracle-driver`](active/harness-oracle-driver.md) | DGX | L | Sketch — picked up only when [`strafer-lab-subgoal-env`](active/strafer-lab-subgoal-env.md) has shipped the NoCam waypoint-following checkpoint AND teleop throughput is the binding scale constraint for VLA training. See trigger condition in brief. |
| [`harness-mission-generator`](active/harness-mission-generator.md) | DGX | L | Free-text mission generator with LLM-emitted waypoints (multi-room default). Renamed + restructured from the former `harness-procedural-path-shape-generator` sketch. Canonical mission queue source for teleop and oracle drivers. Blocked on `multi-room-scene-connectivity-validation` for the connectivity graph. |
| [`harness-trajectory-first-captioning`](active/harness-trajectory-first-captioning.md) | DGX | M–L | Speaker-model post-hoc captioning regime (Speaker-Follower / hindsight-relabeling pattern). Random-A→B drivers + Qwen2.5-VL-7B speaker → instructive-voice mission text + synthesized hard negatives. FoV-honest by construction; complements teleop / bridge / oracle. |

---

## Quick wins

Briefs estimated **S** that any agent can knock out in <1 day. Useful
for fresh-session pickup. Cross-cut — these also appear above.

- [`d555-distortion-model-explicit`](active/d555-distortion-model-explicit.md) (DGX, P2)
- [`planner-rotate-direction-prompt`](active/planner-rotate-direction-prompt.md) (DGX, P2)
- [`rotate-in-place-sim-clock-deadline`](active/rotate-in-place-sim-clock-deadline.md) (Jetson, P2)
- [`vlm-bbox-overlay`](active/vlm-bbox-overlay.md) (Jetson, P2)

---

## Blocked

Briefs that aren't pickable until something else lands.

| Brief | Blocks on | Why |
|---|---|---|
| [`strafer-inference-hybrid-mode`](active/strafer-inference-hybrid-mode.md) | [`strafer-inference-package`](active/strafer-inference-package.md) (Jetson) **and** [`strafer-lab-subgoal-env`](active/strafer-lab-subgoal-env.md) (DGX) both shipped | Hybrid backend extends the inference package's runtime AND consumes the `NOCAM_SUBGOAL` checkpoint produced by the env brief. The two prerequisites can run in parallel since they're cross-lane |
| [`clip-cotrained-retrieval-augmented`](active/clip-cotrained-retrieval-augmented.md) | [`clip-mid-mission-validator-evaluation`](active/clip-mid-mission-validator-evaluation.md) shipped + [`harness-trajectory-first-captioning`](active/harness-trajectory-first-captioning.md) shipped (provides the speaker corpus) | Replaces the retired `learned-mid-mission-validator` as the cascade-improvement path. The co-training step and retrieval-augmented step compound; both compose with the existing cascade. |
| [`strafer-vla-v2-architecture`](active/strafer-vla-v2-architecture.md) | [`harness-teleop-driver`](active/harness-teleop-driver.md) **and** [`harness-behavior-cloning-data-expansion`](active/harness-behavior-cloning-data-expansion.md) shipped | Needs the action-labeled corpus (teleop primary, bridge supplement) before any VLA fine-tune is meaningful. Sim-first research arm; additive to v1. |
| [`harness-oracle-driver`](active/harness-oracle-driver.md) | [`strafer-lab-subgoal-env`](active/strafer-lab-subgoal-env.md) shipped (provides NoCam waypoint-follower checkpoint) **and** trigger condition: teleop throughput is the binding scale constraint for VLA training (see brief) | Filed-on-trigger sketch. Don't pick up preemptively. |
| [`harness-mission-generator`](active/harness-mission-generator.md) | [`multi-room-scene-connectivity-validation`](active/multi-room-scene-connectivity-validation.md) shipped (provides connectivity graph) | Promoted from the previous filed-on-trigger sketch. Now the canonical mission-queue source for teleop / oracle drivers. |
| [`progress-aware-nav-timeouts`](active/progress-aware-nav-timeouts.md) | [`plan-compiler-skill-timeouts`](active/plan-compiler-skill-timeouts.md) shipped (compiler must stop overriding `timeout_s` first; this brief replaces the env-knob backstop with per-step distance-derived budgets + Nav2 stall watchdog) | Jetson lane (executor + ros_client). Estimate M (~1–2 d). Composes with [`rotate-in-place-sim-clock-deadline`](active/rotate-in-place-sim-clock-deadline.md) which can land in either order. |

---

## How to use this board

### As an agent (no specific brief assigned)

1. Scan **Ready to pick up** for your lane (or **Quick wins** if you
   have an hour).
2. Match priority + estimate to your session budget.
3. Read the brief end-to-end before starting; if its `Out of scope`
   or dependencies have shifted since the board was last updated,
   prefer the brief over the board.

### As a contributor filing a new brief

1. Add a row to **Ready to pick up** under the right priority + lane,
   in the same commit that files the brief.
2. If the brief depends on another unshipped brief, file under
   **Blocked** with the dependency named explicitly.

### As a contributor picking up a brief

1. Move the row from **Ready to pick up** → **In flight** with the
   PR number, in the commit that opens the PR (or the first commit
   on the task branch if the PR comes later).

### As a contributor shipping a brief

1. Move the brief into [`completed/`](completed/) with the standard
   stamp (`**Status:** Shipped <date> in <commit> (<host>).` +
   `**PR:** <url>`).
2. Remove it from **In flight** on this board.
3. If the brief's validation surfaced follow-ups, add them under
   **Ready to pick up** (or **Blocked** if they have unshipped
   dependencies) in the same commit.
