# Audit and tune sim_real_cfg.py domain randomization against state-of-the-art

**Type:** investigation / refactor
**Owner:** DGX (`strafer_lab` lane — env config + training run)
**Priority:** P1 — sim-to-real transfer quality is the entire premise
of the `strafer_direct` MVP in
[`inference-package`](inference-package.md). The current
`sim_real_cfg.py` was tuned in isolation against the brief's authors'
intuition; comparing the knob values to what peer teams (Isaac Lab
official envs, Wheeled Lab, ANYmal locomotion, GR00T sim-to-real
workflow) actually use reveals several gaps that ship as silent
deployment failures rather than as training-time errors.
**Estimate:** M (~2–3 days: bench measurement on the real chassis where
applicable, then a single REAL_ROBOT_CONTRACT update + targeted
training resume against a converged baseline). Pure config + training
work; no source-code architecture changes.
**Branch:** task/domain-randomization-audit

## Story

As a **DGX operator preparing a DEPTH checkpoint for real-robot
deployment**, I want **`sim_real_cfg.py`'s REAL_ROBOT_CONTRACT
randomization ranges to match the actual variability the real Strafer
chassis exhibits across runs (payload, battery state, mount tolerance,
control-loop jitter, perception latency)**, so that **the policy
trained against the contract is robust to deployment conditions instead
of being subtly over-fit to the narrow band the current config
exposes**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [inference-package.md](inference-package.md) — Phase 5's
  acceptance metric (1.0 m/s sustained vx with obstacle avoidance) is
  the criterion this brief's training resume must defend on the real
  robot, not just in sim.
- [goal-noise-training.md](goal-noise-training.md) — sister brief that
  addresses the goal-pose noise dimension specifically; this brief
  covers the other axes (dynamics, latency, sensor noise).

## Context

### What the current config does

[`sim_real_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py)
defines three presets:

- `IDEAL_SIM_CONTRACT` — no noise, no delays (debugging).
- `REAL_ROBOT_CONTRACT` — what every `*-Real-*` registered env
  consumes; the deployment-target.
- `ROBUST_TRAINING_CONTRACT` — aggressive randomization for the
  `*-Robust-*` envs.

The `*-Real-ProcRoom-Depth-v0` env (the
[`inference-package`](inference-package.md)
deployment target) trains against `REAL_ROBOT_CONTRACT`.

### What peer pipelines randomize

| Knob | REAL_ROBOT_CONTRACT today | Peer reference | Gap |
|---|---|---|---|
| Friction (μ) | `(0.6, 1.2)` | [Wheeled Lab](https://arxiv.org/html/2502.07380v2): `U(0.2, 0.8)` | OK in range but Strafer never sees low-friction surfaces (e.g. polished concrete, dust); widen lower bound. |
| Mass (multiplier) | `(0.95, 1.05)` | Wheeled Lab: "per-rollout, wider extents." Isaac Lab official envs commonly ±20–30%. | **Too tight.** Strafer's nominal mass ~4.5 kg; payload (D555 + cables + camera mount + occasional sensor pod) varies by ≥ ±15%. Real bench measurement needed. |
| Motor strength | `(0.92, 1.08)` | [ANYmal review](https://www.oaepublish.com/articles/ir.2022.20): "PD gains and stall torques" randomized. 4S LiPo voltage 14.0–16.8 V → ~±20% torque envelope. | **Too tight for battery dynamics.** ROBUST `(0.80, 1.20)` is closer. Promote ROBUST's range to REAL or split into a `motor_strength_battery_range`. |
| Motor time constant | `(0.03, 0.08)` s | GoBilda 5203 datasheet + bench measurement under varying load | Reasonable; verify with bench measurement. |
| Action latency | 0–2 steps (0–66 ms) | Wheeled Lab: "actuator delays randomized per roll-out" — typically 10–100 ms for serial/CAN bus. | OK for ROS-over-LAN; **too generous for real on-chassis serial**, which is closer to 5–15 ms. If sim-in-the-loop uses ROS but real chassis uses RoboClaw direct, these diverge. |
| Depth latency | 1 step (33 ms) | Intel D555 datasheet: stereo matching alone adds ~30–66 ms; add ROS transport. | **Too tight.** Real D555 publish-to-subscribe latency on Jetson is 60–120 ms measured. Widen to `(2, 4)` steps. |
| Control rate jitter | ±5% | ROS on Jetson under load: P99 jitter is 20–50% per [`rtabmap-cold-start-determinism`](../reliability/rtabmap-cold-start-determinism.md). | **Too tight.** Widen to ±15% for REAL, ±25% for ROBUST. |
| D555 mount angle | ±1° | Hand-mounted hardware, screw tolerances, chassis flex | Reasonable but probably understated; ±3° (ROBUST today) more realistic. |
| ProcRoom difficulty | `min_level=7, max_level=7` (fixed) | Curriculum literature: progressive difficulty during training. | **Fixed at one level.** Policy never sees easier or harder rooms. Doesn't generalize to deployment scene variance. |
| Goal-pose noise | ~0.15 m at reset only (`randomize_goal_noise` event, mode="reset") | [`goal-noise-training`](goal-noise-training.md) covers per-tick; this brief defers to it. | Covered separately — out of scope here, cross-ref only. |
| Encoder noise | velocity_noise_std=0.02 | Reasonable for GoBilda 5203 (quadrature) | Looks OK. |
| IMU noise | density-based (BMI055 datasheet-anchored) | Datasheet-correct | Looks OK. |

### Why "training-time" randomization, not just deployment "tolerance"

Adding randomization to training is more expensive than tolerating
deployment slop in a hand-coded controller. The trade is:

- Narrow randomization → faster convergence, brittle deployment.
- Wide randomization → slower convergence, robust deployment.

Peer pipelines that ship to real robots (Wheeled Lab, ANYmal,
Habitat-Sim2Real) all consistently widen randomization beyond what the
hardware spec sheet suggests, because deployment-time variability is
larger than spec sheets capture. The Strafer config is currently a
*spec-sheet-tight* contract; this brief widens it to *deployment-real*.

### Why now, not after the first deployment

If the DEPTH MVP ships at acceptance-quality on the spec-tight contract
and then degrades on the real robot, the failure mode is "policy works
in sim, plateaus / wobbles / clips on real" — exactly the kind of
silent-failure mode peer teams report when DR was under-specified.
Catching this before the first DEPTH ship is cheaper than re-training
from a partially-deployed checkpoint.

## Approach

Three phases. Phase 1 is measurement; Phase 2 is config edit; Phase 3
is a resume training run + comparative evaluation.

### Phase 1 — Bench measurement (1 day)

Measure on the real Strafer (Jetson, real chassis, D555) the four
knobs where peer references suggest the current config is mis-tuned:

1. **Payload mass variance.** Weigh the chassis in three configurations:
   bare, dev rig (current default), deployed-with-sensor-pod. Record
   median + range.
2. **Battery voltage swing under load.** Drive a 5-minute mission with
   the current `STRAFER_NAV_VEL_SCALE` setting; log battery voltage at
   start and after the run. Repeat for fresh-charged (16.8 V) and
   half-discharged (15.0 V) packs. Compute motor-strength range.
3. **D555 publish-to-subscribe latency.** Subscribe to
   `/d555/depth/image_rect_raw` at full rate and log
   `header.stamp - now()` for 60 s. Compute median + p95 latency in
   `physics_dt` units (1/120 s).
4. **Control-loop jitter on Jetson under load.** Run the existing
   inference node (once it ships) with a `time.perf_counter()` log on
   every tick; measure P50/P95/P99 inter-tick spacing under the same
   conditions as Phase 3. Compute jitter percentage vs.
   `_DEFAULT_NAV_DECIMATION * _DEFAULT_NAV_SIM_DT`.

Record measurements in the PR description as a single table.

### Phase 2 — Update REAL_ROBOT_CONTRACT

In [`sim_real_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py)
update `create_real_robot_contract()` based on Phase 1 measurements.
Each change must cite a row in the Phase 1 table — no speculative
widening.

Anticipated edits (subject to Phase 1 data):

```python
# Wider mass to cover sensor-pod variation
randomize_mass=EventTerm(
    func=mdp.randomize_mass, mode="reset",
    params={"mass_range": (0.85, 1.20)},  # was (0.95, 1.05)
)

# Wider motor strength to cover battery voltage swing
randomize_motor_strength=EventTerm(
    func=mdp.randomize_motor_strength, mode="reset",
    params={"strength_range": (0.85, 1.15)},  # was (0.92, 1.08)
)

# Higher depth latency to match measured D555 + ROS transport
TimingCfg(
    depth_latency_steps=2,  # was 1
    depth_latency_steps_range=(1, 4),  # was (0, 0); new field
)

# Wider control jitter to match Jetson under load
TimingCfg(
    control_frequency_jitter_pct=0.15,  # was 0.05
)
```

Also widen the ProcRoom difficulty range from `(7, 7)` fixed to a
graduated `(5, 9)` so the policy sees a span. Stay within the
solvable-room range — if `max_level=9` produces unsolvable layouts,
adjust per the proc_room solvability check.

The new defaults stay within `ROBUST_TRAINING_CONTRACT`'s envelope —
ROBUST stays the strict upper bound for stress-testing.

### Phase 3 — Resume training + comparative evaluation

Pick the converged DEPTH ProcRoom baseline checkpoint. Resume training
for 10–15% of the original training-iter budget against the updated
REAL_ROBOT_CONTRACT. This is the same "targeted final pass" pattern as
[`goal-noise-training`](goal-noise-training.md) Phase 3 — the policy
adapts to the wider distribution without forgetting the base
navigation.

Save as
`logs/rsl_rl/strafer_navigation/depth_dr_audit_v1/model_<step>.pt`.

Comparative evaluation, sweeping each new knob independently at
evaluation time:

| | baseline | DR-audit | Δ |
|---|---|---|---|
| Eval at original DR | success rate | success rate | should be ≈ same |
| Eval at +50% mass | success rate | success rate | DR-audit > baseline |
| Eval at +50% depth latency | success rate | success rate | DR-audit > baseline |
| Eval at +50% jitter | success rate | success rate | DR-audit > baseline |

Per-cell metrics: median final-distance-to-goal, success rate (reach
goal within episode), collision rate.

The DR-audit checkpoint should:
- Be within 5% of baseline at original-DR eval (no degradation under
  the train distribution — the wider DR shouldn't have damaged base
  policy).
- Substantially better at each stress-eval cell.

If the DR-audit checkpoint regresses at original-DR eval, either the
widening was too aggressive or the baseline wasn't actually converged.
Investigate before declaring done.

## Acceptance criteria

### Measurement

- [ ] PR description includes a Phase 1 measurement table with median
      + range for: payload mass, battery voltage range, D555 latency
      (median + p95), control-loop jitter (P50/P95/P99).

### Config

- [ ] `create_real_robot_contract()` updated in
      [`sim_real_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py)
      with each change citing a Phase 1 row.
- [ ] `_RANDOMIZE_PROC_ROOM_DIFFICULTY` extended from
      `(7, 7)` to a graduated range covering at least 3 levels.
- [ ] All unit tests under `source/strafer_lab/tests/` still pass —
      contract changes are config-only; no API change.

### Training + evaluation

- [ ] DR-audit checkpoint exists at
      `logs/rsl_rl/strafer_navigation/depth_dr_audit_v1/model_<step>.pt`
      with a sidecar JSON noting `baseline_checkpoint` provenance.
- [ ] PR description includes the comparative evaluation table:
      baseline vs DR-audit, evaluated at original DR and at +50% mass /
      depth-latency / jitter. DR-audit must show ≥ 10% success-rate
      improvement on stress cells and ≤ 5% degradation on the original
      DR cell.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py)
  `create_real_robot_contract()` (lines 425-484) — the edit site.
- [`source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py)
  `_RANDOMIZE_PROC_ROOM_DIFFICULTY` (around line 1649) — the proc-room
  level config.
- Wheeled Lab paper [arxiv:2502.07380](https://arxiv.org/html/2502.07380v2) —
  the closest peer pipeline (low-cost wheeled robots, Isaac Lab,
  rsl_rl). Section on visual navigation domain randomization.
- ANYmal locomotion review [iir.2022.20](https://www.oaepublish.com/articles/ir.2022.20)
  — the textbook reference for sim-to-real DR widening on robots that
  actually ship.
- GR00T sim-to-real workflow [blog](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow/)
  — current NVIDIA reference for what "sufficient" DR looks like for
  policies that deploy on real hardware.

## Out of scope

- **Goal-position noise.** That's
  [`goal-noise-training`](goal-noise-training.md). Don't double-tune.
- **Replacing the DR architecture.** `SimRealContractCfg`'s three-tier
  pattern is fine; this brief tunes the REAL tier's *values*, not the
  shape.
- **Sensor failure mode randomization.** `SensorFailureCfg` is
  configured separately (currently disabled in REAL). Re-enabling is
  a separate brief if a real deployment incident motivates it; this
  brief targets the common-case distribution, not failure modes.
- **NoCam policy retraining.** This brief targets the DEPTH MVP
  deployment target. NoCam has its own deployment lane (hybrid mode)
  filed under
  [`subgoal-env`](subgoal-env.md) +
  [`hybrid-mode`](../../parked/trained-policy/hybrid-mode.md); a
  parallel DR audit for NoCam_SUBGOAL can follow once that lane is
  active.
- **Cross-room / multi-room scene randomization.** That's the
  multi-room epic ([`multi-room/`](../../active/multi-room/)). This
  brief stays inside the single-room ProcRoom distribution.
