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
| _None._ | | | |

---

## Ready to pick up

Grouped by priority tier, then by lane. Within each cell the rough
order is "smallest / least-blocking first," but pick what fits your
session.

### P1 — high priority

| Brief | Owner | Estimate | Note |
|---|---|---|---|
| [`integration-prompts-refresh`](active/integration-prompts-refresh.md) | Either | M | Refreshes the DGX/Jetson/sim integration runbooks; blocks the next end-to-end integration round |
| [`async-camera-publishers`](active/async-camera-publishers.md) | DGX | L (~3–5 d) | DGX bridge perf — closes the `OnPlaybackTick` gap (camera publish off Kit's main loop) |
| [`strafer-inference-package`](active/strafer-inference-package.md) | Jetson | L (~1.5 wk) | DEPTH MVP — `strafer_direct` mode with the trained ProcRoom-Depth policy. Phases 1–4 land without a deployable checkpoint; Phase 5 gates on DGX-side export+training. Architectural answer to MPPI's plateau |
| [`policy-export-onnx-depth`](active/policy-export-onnx-depth.md) | DGX | M (~1–2 d) | Implement `_OnnxDepthGRUModel` so DEPTH gets ONNX export. Unblocks the Jetson TRT-EP latency target. Filed off `policy-export-tooling` ship. |
| [`policy-loader-recurrent-state`](active/policy-loader-recurrent-state.md) | Either | S–M (~1 d) | Extend `strafer_shared.policy_interface.load_policy()` so recurrent artifacts expose `.reset()` and ONNX hidden state threads correctly. Prerequisite for stateful DEPTH inference on Jetson. |

### P2 — medium priority

#### DGX lane

| Brief | Estimate | Note |
|---|---|---|
| [`d555-distortion-model-explicit`](active/d555-distortion-model-explicit.md) | S | Quick win |
| [`planner-rotate-direction-prompt`](active/planner-rotate-direction-prompt.md) | S | Quick win — prompt edit |
| [`policy-goal-noise-training`](active/policy-goal-noise-training.md) | M | Targeted DEPTH-baseline training pass with goal-position noise; gates VLM-grounded mission quality for `strafer_direct` |
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
| [`plan-compiler-skill-timeouts`](active/plan-compiler-skill-timeouts.md) | S | Quick win — drop hardcoded timeouts so `STRAFER_NAVIGATION_TIMEOUT_S` takes effect |

### P3 — has dependencies

| Brief | Owner | Estimate | Blocks on |
|---|---|---|---|
| [`strafer-inference-hybrid-mode`](active/strafer-inference-hybrid-mode.md) | Jetson | M (~3–4 d) | [`strafer-inference-package`](active/strafer-inference-package.md) shipped + [`strafer-lab-subgoal-env`](active/strafer-lab-subgoal-env.md) shipped (the latter produces the trained `NOCAM_SUBGOAL` checkpoint) |

---

## Quick wins

Briefs estimated **S** that any agent can knock out in <1 day. Useful
for fresh-session pickup. Cross-cut — these also appear above.

- [`d555-distortion-model-explicit`](active/d555-distortion-model-explicit.md) (DGX, P2)
- [`plan-compiler-skill-timeouts`](active/plan-compiler-skill-timeouts.md) (Either, P2)
- [`planner-rotate-direction-prompt`](active/planner-rotate-direction-prompt.md) (DGX, P2)
- [`rotate-in-place-sim-clock-deadline`](active/rotate-in-place-sim-clock-deadline.md) (Jetson, P2)
- [`vlm-bbox-overlay`](active/vlm-bbox-overlay.md) (Jetson, P2)

---

## Blocked

Briefs that aren't pickable until something else lands.

| Brief | Blocks on | Why |
|---|---|---|
| [`strafer-inference-hybrid-mode`](active/strafer-inference-hybrid-mode.md) | [`strafer-inference-package`](active/strafer-inference-package.md) (Jetson) **and** [`strafer-lab-subgoal-env`](active/strafer-lab-subgoal-env.md) (DGX) both shipped | Hybrid backend extends the inference package's runtime AND consumes the `NOCAM_SUBGOAL` checkpoint produced by the env brief. The two prerequisites can run in parallel since they're cross-lane |

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
