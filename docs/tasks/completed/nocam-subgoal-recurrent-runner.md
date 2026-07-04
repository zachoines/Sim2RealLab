# Recurrent (GRU) NOCAM_SUBGOAL runner arm for the corner-anticipation ablation

**Status:** Shipped 2026-07-04 in `98f073b` (DGX).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/136
**Follow-ups:** [`subgoal-corridor-clearance`](../parked/trained-policy/subgoal-corridor-clearance.md)
— the corridor-clearance half of the original brief, split out and parked
(infeasible as a planner-inflation margin at this geometry; see below).

**Type:** task / training-config ablation arm
**Owner:** DGX (`strafer_lab` lane — runner cfg + env registrations + unit tests)
**Priority:** P2 — unblocks the two-arm corner-cutting training comparison; the
env/obs contract is unchanged.
**Branch:** task/nocam-subgoal-recurrence-and-margin

## Story

As a **`strafer_lab` operator diagnosing persistent corner-cutting collisions in
the `NOCAM_SUBGOAL` policy**, I want **a recurrent (GRU) runner arm that trains on
the exact same env and obs as the shipped MLP arm**, so that **a two-arm
MLP-vs-GRU comparison can isolate whether the cutting is an *anticipation*
(memory) failure — a memoryless policy sees one subgoal bearing/distance sample
per step and cannot infer corridor curvature until the rolling subgoal has
already swept the corner.**

## What shipped

Two orthogonal additions in `source/strafer_lab/`, no change to the 19-dim obs
contract (recurrence changes the network, not the observation):

1. **`STRAFER_PPO_RECURRENT_RUNNER_CFG`** (`agents/rsl_rl_ppo_cfg.py`) — the NOCAM
   MLP runner with a recurrent **actor and critic** (stock `RslRlRNNModelCfg`,
   GRU / hidden 128 / 1 layer). Every PPO hyperparameter, rollout scalar, trunk
   width (`[256, 256, 128]`), obs group, and experiment name is byte-identical to
   `STRAFER_PPO_RUNNER_CFG`; the actor/critic model class + its three rnn fields
   are the only difference. A unit test pins this field-for-field so the ablation
   cannot silently drift a second variable.
   - **Both actor and critic are recurrent**, mirroring the DEPTH tier (whose
     critic is also recurrent). In this POMDP a memoryless critic would produce
     biased value estimates in exactly the history-dependent states that matter
     (mid-corner), degrading advantage estimation and muddying the ablation. The
     cfg infrastructure supports a recurrent critic (the DEPTH tier already ships
     one), so nothing forced an actor-only compromise.

2. **Two env registrations** (`tasks/navigation/__init__.py`) pairing the
   **existing** env-cfg classes (same env, same obs) with the new runner:
   - `Isaac-Strafer-Nav-RLNoCam-Subgoal-Robust-GRU-v0` → `StraferNavCfg_RLNoCamSubgoal_Robust`
   - `Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-GRU-v0` → `StraferNavCfg_RLNoCamSubgoal_Real_PLAY`
     (the play/eval + export-shape source for the GRU arm).

### Export path — no tooling change

The recurrent deployment substrate was already shipped and validated (loader
hidden-state handling, the recurrent-state contract, the sidecar `is_recurrent`
flag, inference-node mission-boundary resets). Verified by inspection that both
`scripts/export_policy.py` and `strafer_shared/policy_interface.py` key
recurrence off the **checkpoint/artifact structure** (`getattr(policy_model,
"is_recurrent")` → sidecar; loader dispatches on the sidecar flag, falling back
to TorchScript `.reset()` / ONNX `h_in` port) — **never off the variant name**.
Nothing hard-assumes `NOCAM_SUBGOAL ⇒ stateless`; obs-dim validation is
recurrence-agnostic (a 19-dim GRU passes unchanged). A GRU checkpoint therefore
exports with `is_recurrent: true` and loads correctly with zero tooling changes.
The GRU artifact exports with `--env Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-GRU-v0`.

## Why the corridor-margin half is not here (evidence chain)

The original brief paired this recurrence arm with a corridor **tracking
margin** (widen A\* obstacle inflation from robot-radius to robot-radius + a
0.10–0.15 m margin) to address the *other* named contributor to corner-cutting:
zero-clearance corridors, where a perfectly-tracked corner is still a graze
because inflation equals exactly the robot radius. That change **triggered the
brief's own doorway-passability STOP gate** and is infeasible at this geometry:

- The ProcRoom generator's only structured guaranteed opening is a doorway of
  **min width 0.8 m** (`door_width = rand()*0.4 + 0.8`), and inflation is
  quantized to whole 0.1 m cells via `math.ceil((ROBOT_HALF_WIDTH + margin) /
  GRID_RES)`. Today that is `ceil(0.28/0.1) = 3` cells (0.3 m radius).

  | margin (m) | inflation cells | radius (m) | free gap through 0.8 m door |
  |---|---|---|---|
  | 0.00 (today) | 3 | 0.30 | 0.20 m |
  | ≤ 0.019 | 3 | 0.30 | 0.20 m (**no-op** — grid identical to today) |
  | 0.02 – 0.11 | 4 | 0.40 | **0.00 m — sealed** |
  | 0.15 | 5 | 0.50 | **−0.20 m — sealed** |

- The margin is effectively binary: either *no change* (≤ 0.019 m) or *+0.10 m
  radius* (≥ 0.02 m), which seals the 0.8 m door. Even an idealized continuous
  planner leaves only `0.8 − 2·(0.28 + 0.10) = 0.04 m` of center-slack at margin
  0.10 — a **physical** incompatibility, not just a discretization artifact.
- The 0.8 m slack figure *is* the tracking-accuracy spec (±0.12 m) the blind
  policy must meet — there is no clearance to add; corridors cannot widen, so
  tracking must sharpen. That is precisely the recurrent arm's job, so the
  two-arm comparison **is** the corridor-clearance experiment.

Per the coordinator design consult, the clearance work is re-scoped and parked
as [`subgoal-corridor-clearance`](../parked/trained-policy/subgoal-corridor-clearance.md)
(lead candidate: planner-side medial-axis waypoint biasing that degrades to zero
in doorways — not an inflation bump). The off-path bound decoupling
(`_SUBGOAL_MAX_OFF_PATH_M`) moves there too: with no inflation change it is
numerically identical to today (0.3 m), so decoupling now would be churn with no
behavior change.

## Operator hand-off (not a merge gate)

Two-arm training on the SAME post-PR env, both with the shipped dwell-based
parking success:
- **Arm A** — `Isaac-Strafer-Nav-RLNoCam-Subgoal-Robust-v0` (MLP) → `strafer_nocam_subgoal_v1`.
- **Arm B** — `Isaac-Strafer-Nav-RLNoCam-Subgoal-Robust-GRU-v0` (GRU) → `v2` candidate.

Play gates on the respective `Real-Play` / `Real-Play-GRU` envs. Comparison
metrics: play-gate parking quality; corner cross-track error; `sustained_collision`
vs `path_complete` termination fractions at convergence. Those metrics are also
the trigger evaluation for the parked clearance brief, and the decision evidence
for the parked collider-noise question.

Note: the GRU runner keeps `experiment_name="strafer_navigation"` identical to
the MLP baseline (strict ablation cleanliness). rsl_rl separates runs by
timestamped subdir, so the two arms co-locate under one experiment tree without
collision; pass a distinct run name at launch if you want them foldered apart.

## Acceptance

Env/runner changes gate on **unit tests only** — the operator two-arm training
decision does **not** gate this PR.

- Pure half (`make test-lab-pure`): **704 passed, 1 skipped**, including 4 new
  `tests/navigation/test_recurrent_runner_cfg.py` checks (runner differs from the
  MLP baseline only in policy class + rnn fields; GRU/128/1 on actor and critic;
  both GRU ids register and resolve to the recurrent runner object; GRU arm
  reuses the existing subgoal env cfgs).
- Kit half (registration exact-set, runner-obs-profile mapping, and the
  `clip_actions` invariant extended to the new cfg): green.
