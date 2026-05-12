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
| [`task-board-epic-structure`](active/task-board-epic-structure.md) | Either | _(this PR)_ | In review |

---

## By epic

Feature-area view. Active briefs are pickable; parked briefs are
filed-on-trigger or blocked-on-deps — see **Parked** below for the
explicit dependencies.

### Multi-room navigation

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`autonomy-stack`](active/multi-room/autonomy-stack.md) | P1 | active | Either |
| [`scene-connectivity-validation`](active/multi-room/scene-connectivity-validation.md) | P1 | active | DGX |
| [`planner-far-target-staging`](active/multi-room/planner-far-target-staging.md) | P2 | active | DGX |

### Trained-policy backend

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`export-onnx-depth`](active/trained-policy/export-onnx-depth.md) | P1 | active | DGX |
| [`loader-recurrent-state`](active/trained-policy/loader-recurrent-state.md) | P1 | active | Either |
| [`inference-package`](active/trained-policy/inference-package.md) | P1 | active | Jetson |
| [`goal-noise-training`](active/trained-policy/goal-noise-training.md) | P2 | active | DGX |
| [`subgoal-env`](active/trained-policy/subgoal-env.md) | P2 | active | DGX |
| [`hybrid-mode`](parked/trained-policy/hybrid-mode.md) | P3 | parked | Jetson |

### Harness & training data

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`teleop-driver`](active/harness/teleop-driver.md) | P1 | active | DGX |
| [`behavior-cloning-data-expansion`](active/harness/behavior-cloning-data-expansion.md) | P2 | active | DGX |
| [`mission-generator`](active/harness/mission-generator.md) | P2 | active | DGX |
| [`trajectory-first-captioning`](active/harness/trajectory-first-captioning.md) | P2 | active | DGX |
| [`oracle-driver`](parked/harness/oracle-driver.md) | P3 | parked | DGX |

### CLIP mid-mission validation

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`validator-evaluation`](active/clip-validation/validator-evaluation.md) | P1 | active | Either |
| [`cotrained-retrieval-augmented`](parked/clip-validation/cotrained-retrieval-augmented.md) | P3 | parked | DGX |

### Sim performance

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`isaac-sim-rt-2-default-renderer`](active/sim-performance/isaac-sim-rt-2-default-renderer.md) | P2 | active | DGX |
| [`bridge-throughput-toward-25hz`](active/sim-performance/bridge-throughput-toward-25hz.md) | P2 | active | DGX |

### Reliability (nav + executor + refactors)

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`nav-deadline-sim-time-audit`](active/reliability/nav-deadline-sim-time-audit.md) | P2 | active | Jetson |
| [`executor-prefer-rotate-then-translate`](active/reliability/executor-prefer-rotate-then-translate.md) | P2 | active | Jetson |
| [`rtabmap-cold-start-determinism`](active/reliability/rtabmap-cold-start-determinism.md) | P2 | active | Jetson |
| [`planner-rotate-direction-prompt`](active/reliability/planner-rotate-direction-prompt.md) | P2 | active | DGX |
| [`grounding-publisher-extraction`](active/reliability/grounding-publisher-extraction.md) | P2 | active | Jetson |
| [`nav-stall-multilayer-watchdog`](parked/reliability/nav-stall-multilayer-watchdog.md) | P3 | parked | Jetson |
| [`perception-side-bearing-service`](parked/reliability/perception-side-bearing-service.md) | P3 | parked | Jetson |

### Tooling & ops

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`task-board-epic-structure`](active/task-board-epic-structure.md) | P2 | in flight | Either |
| [`unify-test-targets-and-ci`](active/tooling/unify-test-targets-and-ci.md) | P3 | active | Either |
| [`windows-workstation-bringup`](active/tooling/windows-workstation-bringup.md) | P2 | active | DGX |

### Experimental (long-horizon bets)

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`vla-v2-architecture`](parked/experimental/vla-v2-architecture.md) | P3 | parked | Either |

### Investigations (measurement / knowledge work)

| Brief | Pri | State | Owner |
|---|---|---|---|
| [`next-integration-round`](active/investigations/next-integration-round.md) | P1 | active | Either |
| [`real-d555-depth-range-survey`](active/investigations/real-d555-depth-range-survey.md) | P2 | active | Jetson |
| [`training-throughput-profile-and-investigate`](active/investigations/training-throughput-profile-and-investigate.md) | P2 | active | DGX |

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
| [`export-onnx-depth`](active/trained-policy/export-onnx-depth.md) | DGX | M (~1–2 d) | Implement `_OnnxDepthGRUModel` so DEPTH gets ONNX export. Unblocks the Jetson TRT-EP latency target. Filed off `policy-export-tooling` ship. |
| [`loader-recurrent-state`](active/trained-policy/loader-recurrent-state.md) | Either | S–M (~1 d) | Extend `strafer_shared.policy_interface.load_policy()` so recurrent artifacts expose `.reset()` and ONNX hidden state threads correctly. Prerequisite for stateful DEPTH inference on Jetson. |
| [`validator-evaluation`](active/clip-validation/validator-evaluation.md) | Either | L | Wire the orphaned `SemanticMapManager` + `BackgroundMapper` + `TransitMonitor` path into the production executor and measure pre-registered TPR/FPR/time-to-decision on harness output. Gating brief for `MISSION_VALIDATION_ARCHITECTURE.md` §4 staged plan. Filed off `mid-mission-validation-investigation` ship. |
| [`teleop-driver`](active/harness/teleop-driver.md) | DGX | M | Gamepad teleop entry point for in-process Isaac Lab data capture. Bypasses MPPI / Nav2 / planner; reuses `collect_demos.py` mapping; emits the canonical harness schema. Unblocks v1 measurement (clip-eval, learned-validator) and v2 VLA training data without depending on bridge perf. |
| [`autonomy-stack`](active/multi-room/autonomy-stack.md) | Either | M | Lifts §1.10.1's multi-room deferral. Stored-map fallback in `scan_for_target` + planner transit-step emission + plan-compiler updates. Required for multi-room as the MVP default. |
| [`scene-connectivity-validation`](active/multi-room/scene-connectivity-validation.md) | DGX | S | Computes room connectivity at scene-gen time + verifies door-open default in `prep_room_usds.py`. Hard prerequisite for `autonomy-stack` and `mission-generator`. |

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

#### Jetson lane

| Brief | Estimate | Note |
|---|---|---|
| [`grounding-publisher-extraction`](active/reliability/grounding-publisher-extraction.md) | S | Quick win — pure refactor follow-up to `vlm-bbox-overlay`; extracts the viz publishers out of `JetsonRosClient` |
| [`real-d555-depth-range-survey`](active/investigations/real-d555-depth-range-survey.md) | S–M | Investigation — bench measurement + write-up |
| [`rtabmap-cold-start-determinism`](active/reliability/rtabmap-cold-start-determinism.md) | M | Cold-start signature on populated DB: `Not found word N` burst + `Increment map id to 4`; triage bridge-teleport vs Mem/* config + ship the chosen disposition |
| [`nav-deadline-sim-time-audit`](active/reliability/nav-deadline-sim-time-audit.md) | M | Audit executor + Nav2 wall-clock safety caps; replace absolute caps with sim-time-progress stall detectors so rotations don't abort partway at RTF ≤ 0.1 |
| [`executor-prefer-rotate-then-translate`](active/reliability/executor-prefer-rotate-then-translate.md) | M | Decompose non-cardinal translations into rotate-to-face + forward translate at the executor layer; preserves cardinal strafe |

#### Either lane

| Brief | Estimate | Note |
|---|---|---|
| _None currently._ | | |

### P3 — pickable, low priority

| Brief | Owner | Estimate | Note |
|---|---|---|---|
| [`unify-test-targets-and-ci`](active/tooling/unify-test-targets-and-ci.md) | Either | M | Makefile unification + stretch CI workflow. Doesn't block features; bumps to P2 once a second drift incident shows up. |

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
| [`cotrained-retrieval-augmented`](parked/clip-validation/cotrained-retrieval-augmented.md) | [`validator-evaluation`](active/clip-validation/validator-evaluation.md) shipped + [`trajectory-first-captioning`](active/harness/trajectory-first-captioning.md) shipped (provides the speaker corpus) | Replaces the retired `learned-mid-mission-validator` as the cascade-improvement path. The co-training step and retrieval-augmented step compound; both compose with the existing cascade. |
| [`vla-v2-architecture`](parked/experimental/vla-v2-architecture.md) | [`teleop-driver`](active/harness/teleop-driver.md) **and** [`behavior-cloning-data-expansion`](active/harness/behavior-cloning-data-expansion.md) shipped | Needs the action-labeled corpus (teleop primary, bridge supplement) before any VLA fine-tune is meaningful. Sim-first research arm; additive to v1. |
| [`oracle-driver`](parked/harness/oracle-driver.md) | [`subgoal-env`](active/trained-policy/subgoal-env.md) shipped (provides NoCam waypoint-follower checkpoint) **and** trigger: teleop throughput is the binding scale constraint for VLA training (see brief) | Filed-on-trigger sketch. Don't pick up preemptively. |
| [`nav-stall-multilayer-watchdog`](parked/reliability/nav-stall-multilayer-watchdog.md) | Trigger: v1 stall watchdog from `progress-aware-nav-timeouts` produces real-world false-positives (cluttered-sim re-plans) or false-negatives (chassis wedge incident) | Filed-on-trigger sketch. Adds chassis-wedge + Nav2 recovery-rate signals on top of the v1 best-ever-distance watchdog. Don't pick up preemptively — v1 may be sufficient. |
| [`perception-side-bearing-service`](parked/reliability/perception-side-bearing-service.md) | Trigger: at least one bearing-varying behavior (approach-from-angle, look-at-while-driving, face-target-while-translating, etc.) is on the active roadmap | Filed-on-trigger refactor surfaced by `align-after-scan-grounding`'s shipped Option A. Moves bearing math from autonomy executor into `strafer_perception` so future bearing-varying features don't have to thread quaternion math through the executor. |

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
