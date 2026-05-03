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

Open PRs — don't pick these up.

| Brief | Owner | PR | State |
|---|---|---|---|
| [`mppi-critic-tuning-for-sim-envelope`](mppi-critic-tuning-for-sim-envelope.md) | Jetson | [#15](https://github.com/zachoines/Sim2RealLab/pull/15) | review |

---

## Ready to pick up

Grouped by priority tier, then by lane. Within each cell the rough
order is "smallest / least-blocking first," but pick what fits your
session.

### P1 — high priority

| Brief | Owner | Estimate | Note |
|---|---|---|---|
| [`integration-prompts-refresh`](integration-prompts-refresh.md) | Either | M | Refreshes the DGX/Jetson/sim integration runbooks; blocks the next end-to-end integration round |
| [`policy-export-tooling`](policy-export-tooling.md) | DGX | M (~2–3 d) | `Scripts/export_policy.py` — hard dep for `strafer-inference-package` end-to-end |
| [`async-camera-publishers`](async-camera-publishers.md) | DGX | L (~3–5 d) | DGX bridge perf — closes the `OnPlaybackTick` gap (camera publish off Kit's main loop) |
| [`strafer-inference-package`](strafer-inference-package.md) | Jetson | L (~1 wk) | Phases 1–4 land without a deployable checkpoint; the architectural answer to MPPI's plateau |

### P2 — medium priority

#### DGX lane

| Brief | Estimate | Note |
|---|---|---|
| [`d555-distortion-model-explicit`](d555-distortion-model-explicit.md) | S | Quick win |
| [`planner-rotate-direction-prompt`](planner-rotate-direction-prompt.md) | S | Quick win — prompt edit |
| [`policy-goal-noise-training`](policy-goal-noise-training.md) | M | Targeted training pass; gates VLM-grounded mission quality |
| [`planner-far-target-staging`](planner-far-target-staging.md) | M–L | World-state schema + planner prompt |

#### Jetson lane

| Brief | Estimate | Note |
|---|---|---|
| [`rotate-in-place-sim-clock-deadline`](rotate-in-place-sim-clock-deadline.md) | S | Quick win |
| [`vlm-bbox-overlay`](vlm-bbox-overlay.md) | S | Quick win — Foxglove debug overlay |
| [`align-after-scan-grounding`](align-after-scan-grounding.md) | S–M | Executor + plan_compiler tweak |
| [`real-d555-depth-range-survey`](real-d555-depth-range-survey.md) | S–M | Investigation — bench measurement + write-up |
| [`nav2-startup-unknown-donut-path-noise`](nav2-startup-unknown-donut-path-noise.md) | M | Planner-side fix for the kinky-path-through-camera-donut symptom |

#### Either lane

| Brief | Estimate | Note |
|---|---|---|
| [`plan-compiler-skill-timeouts`](plan-compiler-skill-timeouts.md) | S | Quick win — drop hardcoded timeouts so `STRAFER_NAVIGATION_TIMEOUT_S` takes effect |

### P3 — has dependencies

| Brief | Owner | Estimate | Blocks on |
|---|---|---|---|
| [`strafer-inference-hybrid-mode`](strafer-inference-hybrid-mode.md) | Either | L | [`strafer-inference-package`](strafer-inference-package.md) shipped + `NOCAM_SUBGOAL` policy trained |

---

## Quick wins

Briefs estimated **S** that any agent can knock out in <1 day. Useful
for fresh-session pickup. Cross-cut — these also appear above.

- [`d555-distortion-model-explicit`](d555-distortion-model-explicit.md) (DGX, P2)
- [`plan-compiler-skill-timeouts`](plan-compiler-skill-timeouts.md) (Either, P2)
- [`planner-rotate-direction-prompt`](planner-rotate-direction-prompt.md) (DGX, P2)
- [`rotate-in-place-sim-clock-deadline`](rotate-in-place-sim-clock-deadline.md) (Jetson, P2)
- [`vlm-bbox-overlay`](vlm-bbox-overlay.md) (Jetson, P2)

---

## Blocked

Briefs that aren't pickable until something else lands.

| Brief | Blocks on | Why |
|---|---|---|
| [`strafer-inference-hybrid-mode`](strafer-inference-hybrid-mode.md) | [`strafer-inference-package`](strafer-inference-package.md) shipped + new `NOCAM_SUBGOAL` checkpoint trained | Hybrid backend extends the inference package's runtime; subgoal-following needs a new `PolicyVariant` and training run |

---

## Recently shipped — pending move to `completed/`

Empty when housekeeping is current. When non-empty, the next
brief-touching commit should move these.

| Brief | Shipped in | PR |
|---|---|---|
| _(none — `nav2-far-goal-staging` was moved in the same commit that introduced this board)_ | | |

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
