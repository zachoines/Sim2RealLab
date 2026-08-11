# Audit and tune sim_real_cfg.py domain randomization against state-of-the-art

**Type:** investigation / refactor
**Owner:** DGX (`strafer_lab` lane — env config + training run)
**Priority:** P1 — sim-to-real transfer quality is the entire premise
of the `strafer_direct` MVP in
[`inference-package`](../../completed/inference-package.md). The current
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
- [inference-package.md](../../completed/inference-package.md) — Phase 5's
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

`Isaac-Strafer-Nav-RLDepth-Real-v0` and its siblings train against
`REAL_ROBOT_CONTRACT`. **The shipped depth subgoal checkpoint does not** — it
trained on `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0`
(`StraferNavCfg_RLDepthSubgoalEnriched_Robust`, `RealismCfg(level="robust")`),
so every knob it saw came from `ROBUST_TRAINING_CONTRACT` plus
`EventsCfg_ProcRoom_Robust_Enriched`. Read the "today" column below as the REAL
tier; where the ROBUST value differs, the row says so.

### What the shipped depth checkpoint trained on, temporally

The rows below cover the physical knobs. The *temporal* texture is stated
separately because it was far narrower than the config surface suggested.
**This paragraph describes the checkpoint on disk, not the current contracts** —
the temporal axes below have since been randomized, and the next subsection
says how. A checkpoint trained before that lands saw:

- **Depth age: a fixed 2 control steps** (66.7 ms) on ROBUST, 1 on REAL. It is
  a `DelayBuffer` ring shift, not a hold — and the ring is zeroed on reset, so
  the policy sees an all-zero depth block for the first `latency_steps` ticks
  of every episode.
- **Frame drop: 1% i.i.d. Bernoulli per env per step** on ROBUST (0.1% on
  REAL). A drop re-emits the previously *emitted* frame, so repeats chain into
  geometric runs — but at p=0.01 the mean run is ~1.01 frames.
- **Tick rate: a fixed 30 Hz**, structural rather than configurable
  (`_DEFAULT_NAV_SIM_DT` 1/120 × `_DEFAULT_NAV_DECIMATION` 4).
  `control_frequency_hz` does **not** set it; it is read only to scale IMU
  noise density and the bias random-walk step.
- **Cross-modality skew: a constant**, never sampled — depth 2 steps, IMU 1,
  encoders 1, and the goal-shaped and velocity terms 0 (no noise model at
  all). The policy sees depth exactly two ticks older than its bearing, every
  tick, deterministically.

### What the contracts randomize temporally now

Three mechanisms, all per-environment and re-drawn at every reset, so one batch
spans a band of temporal textures rather than sitting at a point:

- **Depth stream holds.** The depth frame fails to advance for a run of steps,
  and the policy re-reads the previous *noisy* frame. Run lengths come from a
  mixture of two geometrics, so the stationary hold fraction and the mean run
  length are independent knobs. REAL bands the fraction over `(0.0, 0.35)` with
  runs of `(1.0, 1.6)` steps — centred on the arrival rate a healthy deploy
  stream sustains, roughly 23 Hz against a 30 Hz tick. ROBUST reaches `(0.0,
  0.60)` with runs of `(1.0, 2.0)` and a quarter of the runs drawn from a
  6-step burst component, which covers the worst arrival rate ever measured on
  the rig, about 12 Hz. The memoryless `frame_drop_probability` is unchanged
  and composes with the hold as a union: a repeat is either.
- **Depth age.** `depth_latency_steps_range` draws the ring shift per env at
  reset — `(0, 2)` on REAL and `(1, 3)` on ROBUST. Both bands are
  mean-preserving around the fixed value they replace, so this randomizes the
  age without moving its centre; where the centre belongs is Phase 1's bench
  measurement, still owed. Randomizing it also un-freezes the cross-modality
  skew, which was previously a constant.
- **Command holds.** A control step on which no new command arrives, so the
  chassis re-executes the last one. Same run-length law, `(0.0, 0.05)` on REAL
  and `(0.0, 0.25)` on ROBUST — deliberately far below the depth band, because
  this models the residual left after the deployed node's inference cadence
  rather than the cadence itself.

Two things are still **not** randomized, and neither is declared as if it were:

- **The tick period.** Still a structural 30 Hz. Randomizing it needs rollout
  surgery, not a config field, so `control_frequency_jitter_pct` was deleted
  rather than left declaring a randomization the env never performed. Same for
  `obs_latency_steps` and `obs_latency_steps_range`, whose wired replacement is
  the per-sensor `*_latency_steps` pair.
- **The recurrent stepping rate.** Training advances the hidden state on every
  tick; a depth-starved deploy node advances it more slowly. This is the
  residual the two mechanisms above approximate rather than reproduce, and it
  closes on its own if the node moves to timer-driven inference with bounded
  stale-depth reuse.

`scripts/eval_cadence_emulation.py` measures behaviour across this band and is
the instrument that scores whether the randomization closed the gap.

### What peer pipelines randomize

| Knob | REAL_ROBOT_CONTRACT today | Peer reference | Gap |
|---|---|---|---|
| Friction (μ) | `(0.6, 1.2)` | [Wheeled Lab](https://arxiv.org/html/2502.07380v2): `U(0.2, 0.8)` | OK in range but Strafer never sees low-friction surfaces (e.g. polished concrete, dust); widen lower bound. |
| Mass (multiplier) | `(0.95, 1.05)` | Wheeled Lab: "per-rollout, wider extents." Isaac Lab official envs commonly ±20–30%. | **Too tight.** Strafer's nominal mass ~4.5 kg; payload (D555 + cables + camera mount + occasional sensor pod) varies by ≥ ±15%. Real bench measurement needed. |
| Motor strength | `(0.92, 1.08)` | [ANYmal review](https://www.oaepublish.com/articles/ir.2022.20): "PD gains and stall torques" randomized. 4S LiPo voltage 14.0–16.8 V → ~±20% torque envelope. | **Too tight for battery dynamics.** ROBUST `(0.80, 1.20)` is closer. Promote ROBUST's range to REAL or split into a `motor_strength_battery_range`. |
| Motor time constant | `(0.03, 0.08)` s | GoBilda 5203 datasheet + bench measurement under varying load | Reasonable; verify with bench measurement. |
| Action latency | 1–3 control steps (33–100 ms); ROBUST 1–5 (33–167 ms) — `action_latency_steps` and `action_latency_steps_range` are **summed** in `get_action_config_params`, and the per-env lag is drawn once at action-term construction, never re-drawn on reset | Wheeled Lab: "actuator delays randomized per roll-out" — typically 10–100 ms for serial/CAN bus. | OK for ROS-over-LAN; **too generous for real on-chassis serial**, which is closer to 5–15 ms. If sim-in-the-loop uses ROS but real chassis uses RoboClaw direct, these diverge. |
| Depth latency | 1 step (33 ms) centre, drawn per env over `(0, 2)`; ROBUST 2 steps over `(1, 3)` | Intel D555 datasheet: stereo matching alone adds ~30–66 ms; add ROS transport. | **Randomized, centre still too tight.** Real D555 publish-to-subscribe latency on Jetson is 60–120 ms measured. The band is mean-preserving, so it added the age *distribution* the axis was missing and left the centre where it was; moving the centre to `(2, 4)` is still owed and still gated on Phase 1's measurement. |
| Depth frame holds | REAL `(0.0, 0.35)` hold fraction over runs of `(1.0, 1.6)` steps; ROBUST `(0.0, 0.60)` over `(1.0, 2.0)` with a quarter of runs from a 6-step burst. Per env, re-drawn at reset. | The deployed node's own arrival statistics: ~23 Hz against a 30 Hz tick when healthy, ~12 Hz with bursty stalls at its worst. | **Closed.** Was a 1% i.i.d. drop whose runs averaged ~1.01 frames — three orders of magnitude short of the run structure deploy produces. |
| Command holds | REAL `(0.0, 0.05)`, ROBUST `(0.0, 0.25)`, same run-length law | A control step on which the node publishes nothing, so the chassis re-executes its last command. | **Closed, deliberately small.** Kept far under the depth band: this is the residual after the node's inference cadence, and sizing it like the cadence would model the same stall twice. Its correct size follows whatever inference semantics the node ends up with. |
| Control rate jitter | **not implemented, and no longer declared.** `control_frequency_jitter_pct`, `obs_latency_steps` and `obs_latency_steps_range` were declared with zero consumers; all three are deleted. The env still ticks a fixed 30 Hz. | ROS on Jetson under load: P99 jitter is 20–50% per [`rtabmap-cold-start-determinism`](../reliability/rtabmap-cold-start-determinism.md). | **Whole axis not randomized.** Randomizing the tick period needs rollout surgery rather than a config field, so the fields were removed rather than left implying a randomization that never ran. Do not cite ±5% jitter or 0–2 steps of observation latency as part of any checkpoint's training distribution. What the deleted knobs were reaching for — a policy flat across the deploy stream's temporal texture — is partly served by the two hold rows above, but **not wholly**: a hold changes *whether* a tick delivers something new, while jitter changes *how much the world advanced between* two policy calls. Sim's `decimation` makes that interval exactly constant, so the second half is unmodelled and stays a known gap. **Deleting the field does not retire the gap — this row is the record of it**, and the trigger for building the mechanism is a measured tick-interval distribution from Phase 1 item 4 whose spread is wide enough to matter (a P99/P50 ratio materially above 1). Implementing it means a variable `decimation` per env per step, which is rollout-level surgery rather than a config field, and that is the reason it was not attempted here rather than a judgement that it does not matter. |
| **Referent-frame drift (map→odom displacement)** | Gain `(0.0, 0.5)` of a measured 0.166 m / 6.7° class on **both** REAL and ROBUST — the one term where robust does not reach past realistic — as a correlated SE(2) walk with τ = 2.0 s on the goal-shaped observations. Per env, re-drawn at reset; the privileged critic still reads the true referent. | The `map→odom` movement recorded on the rig, and the sensitivity arm built on it: 1× costs 24–28% of completion, monotone across 0.5×/1×/2×. The band's ceiling is set by the off-path corridor instead: termination reads TRUE cross-track against a fixed 0.30 m while the policy steers by the drifted referent, so gains approaching 1× end episodes on a displacement no observation carries. | **Closed, band re-anchored.** The goal-shaped terms carried no noise model at all, so the policy trained against a referent that was exact to the millimetre and deployed against one that wanders. The wander is landed. A training leg on a `(0.0, 1.25)` robust band scored **half** the drift-naive reference on its own distribution and never annealed its action distribution, which is what moved the ceiling to the corridor bound; **1× is now an evaluation probe against the trained band, not a point inside it.** τ remains an assumption pending Phase 1 item 6, and a Poisson jump component ships alongside with its band at zero (below). Both bands are re-derived once the rig measures steady-state σ and τ; if the measurement exceeds the trained band, the licensed response is scaling the corridor with the drawn gain, not widening the band. |
| **TF staleness (goal pose / base pose age)** | **not randomized** — sim re-reads the goal pose from the command term at every tick, so the policy sees a displaced reference frame (row above) but never an *old* one | Real Jetson reads goal pose in body frame via the chain `(map→odom)` ⊗ `(odom→base_link)` from a TF buffer that's only as fresh as the slowest publisher in the chain. RTAB-Map's `map→odom` updates at 1–10 Hz; under tracking loss or cold-start ([`rtabmap-cold-start-determinism`](../reliability/rtabmap-cold-start-determinism.md)) it can stall for 100 ms+ at a time. The policy's `body_frame_goal` reading then references a *stale* base pose, so the goal-in-body-frame drifts as the robot moves even though the goal hasn't. | **Age half still not randomized.** The displacement half of this axis closed with the drift row above; what remains is the *delay*, which is a different quantity — an offset that is wrong versus a reference that is old, and only the second one couples to the robot's own motion. Two-step approach unchanged: (1) measure per Phase 1 item 6 below; (2) age the body-frame goal observation by a sampled latency drawn from the measured distribution, sampled within an episode rather than once at reset. **The jump component now has a mechanism and no distribution:** a loop closure re-anchors `map→odom` discontinuously, which is a step rather than an Ornstein-Uhlenbeck path; it ships as a Poisson term inside the drift process with its rate and magnitude band at zero, so turning it on is a config change once Phase 1 item 6 records the closure-jump magnitude and rate alongside the steady-state staleness. |
| D555 mount angle | ±1° | Hand-mounted hardware, screw tolerances, chassis flex | Reasonable but probably understated; ±3° (ROBUST today) more realistic. |
| **D555 mount POSITION** | **not randomized** — fixed at `(CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z) = (0.20, 0.0, 0.25)` m | Hand-mounted bracket, screw-hole tolerance ~±2 mm, cable strain, operator unbolt/rebolt during dev | **Whole axis not randomized.** Every time the operator removes the D555 (e.g. for the IMU kernel fix from `docs/D555_IMU_KERNEL_FIX.md`, lens cleaning, or transport) and rebolts it, the position shifts by ~1–3 cm. The existing `randomize_d555_mount_offset` event in [`events.py:450`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py) handles orientation (`_d555_mount_quat`) and the IMU obs path rotates readings through it; nothing parallel exists for position. |
| ProcRoom difficulty | `min_level=7, max_level=7` on the vanilla generator; the **enriched** variants already un-pin to `U[4, 7]` via `_ENRICH_MIN_LEVEL` / `_ENRICH_MAX_LEVEL` | Curriculum literature: progressive difficulty during training. | **Partly closed.** The enriched retrain target already spans four levels; only the vanilla ProcRoom variants stay pinned. |
| Goal-pose noise | `randomize_goal_noise` (mode="reset") — 0.35 m on every ROBUST tier, 0.15 m on the flat-arena and Infinigen REAL tiers but **absent on the ProcRoom REAL tier** (the lane the depth checkpoints train in), and set to `None` for every **subgoal** variant regardless of tier; the subgoal tiers randomize the planner lookahead instead (`(0.9, 1.1)` real / `(0.7, 1.3)` robust) | [`goal-noise-training`](goal-noise-training.md) covers per-tick; this brief defers to it. | Covered separately — out of scope here, cross-ref only. Note for anyone reading a signed bearing offset off a subgoal policy: that referent is **unperturbed at reset**, so a signed offset is not goal jitter. |
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

Measure on the real Strafer (Jetson, real chassis, D555) the five
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
   `_DEFAULT_NAV_DECIMATION * _DEFAULT_NAV_SIM_DT`. The consumer of this
   measurement is the command-hold band, not a tick-period knob — a missed
   tick that publishes nothing is a hold, and that is what the contract
   models.
5. **D555 mount position vs. nominal.** With the camera mounted in its
   current deployed configuration, measure the actual `(x, y, z)`
   offset of the D555 lens optical center relative to
   `body_link` and compare against the constants
   `CAMERA_OFFSET_X = 0.20`, `CAMERA_OFFSET_Y = 0.0`,
   `CAMERA_OFFSET_Z = 0.25` m in
   [`strafer_shared.constants`](../../../../source/strafer_shared/strafer_shared/constants.py).
   Use a steel ruler / digital caliper at the chassis frame fiducials.
   Then **unbolt the D555, rebolt without intentional re-alignment, and
   re-measure** — the *delta* across that rebolt cycle is the
   distribution width the policy must be robust to. Record both the
   absolute offset (does `strafer_shared.constants` need to be
   updated?) and the rebolt delta (what's the variance the policy
   needs to handle).
6. **TF buffer staleness on the goal-pose body-frame transform.** On
   the running Jetson stack with RTAB-Map publishing `map→odom`,
   subscribe to the same goal topic the inference node consumes and at
   every tick log `(now - tf_buffer.lookup_transform(...).stamp)` for
   the `map→base_link` lookup the goal-pose body-frame projection
   uses. Run two regimes: (a) RTAB-Map nominally tracking; (b) the
   cold-start window of [`rtabmap-cold-start-determinism`](../reliability/rtabmap-cold-start-determinism.md)
   (immediately post DB-load, before first `localized` event). Report
   median + P95 + P99 staleness for both, in `physics_dt` units. The
   distribution from (a) feeds the steady-state randomization; the
   tail of (b) informs the upper bound of the ROBUST tier.
   **Also record the frame's own movement in the same log**, since it
   is the same sitting: sample `map→odom` at ~10 Hz alongside
   RTAB-Map's closure events and report (i) the autocorrelation time
   of the smooth wander — this pins the drift process's `tau_s`, which
   ships as a stated assumption of 2.0 s — and (ii) the magnitude and
   rate distributions of the closure jumps, which are what the shipped
   zero jump band is waiting on.

Record measurements in the PR description as a single table. Phase 1
items 5 and 6 are the inputs for Phase 2's new randomization configs
(camera position and TF staleness, respectively).

### Phase 2 — Update REAL_ROBOT_CONTRACT (and mirror into ROBUST)

In [`sim_real_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py)
update `create_real_robot_contract()` based on Phase 1 measurements.
Each change must cite a row in the Phase 1 table — no speculative
widening.

Any row the shipped depth checkpoint is expected to inherit must also land in
`create_robust_training_contract()` — the depth subgoal lane trains on the
ROBUST tier. Two of the anticipated edits below are already the shipped ROBUST
values (`depth_latency_steps=2`, `D555 mount angle ±3°`), so applying the block
verbatim moves REAL only.

Of the two temporal fields the block introduces, `depth_latency_steps_range`
has since **landed with its consumer** — the per-env sampling inside
`DelayBuffer` — at a mean-preserving band, so what this block still owes on
that row is the *centre*, which is Phase 1 item 3's measurement. The
TF-staleness pair remains a config field with no mechanism behind it; landing
it is still a two-part change, the field *and* the reader.

The frame's *displacement* has since landed separately as
`LocalizationDriftCfg`, so the block below no longer owes that half. What the
staleness pair still models is the reference's **age**, which the drift does
not: an offset that is wrong does not move when the robot does, and a reference
that is old does.

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

# Higher depth latency to match measured D555 + ROS transport.
# The range field and its per-env consumer have landed; what this edit
# still proposes is moving the centre, which Phase 1 item 3 measures.
TimingCfg(
    depth_latency_steps=2,  # was 1
    depth_latency_steps_range=(1, 4),  # widen from the mean-preserving (0, 2)
)

# NEW: TF staleness on the goal-pose body-frame projection.
# Models the gap between when the goal pose was last updated in
# map frame and when the policy reads the body-frame projection
# (RTAB-Map map→odom only refreshes at 1–10 Hz). Sampled per
# `mode="interval"` so the staleness drifts within an episode.
TimingCfg(
    goal_tf_staleness_steps=2,         # ~66 ms median (Phase 1 item 6)
    goal_tf_staleness_steps_range=(0, 6),  # ~0–200 ms span; widen
                                       # to (0, 12) in ROBUST tier
                                       # to cover cold-start tail
)
```

#### TF staleness implementation (new — addresses Phase 1 item 6)

The sim today projects the goal pose into body frame every tick via
the command term (fresh per tick by construction). Real consumes a
TF buffer, which can be 0–200 ms stale depending on RTAB-Map's
`map→odom` publish cadence. Two options for closing the gap:

1. **Sampled per-tick replay of a delayed base pose.** Cache a short
   ring of past base-poses (length matching the upper staleness step
   range); per env, per interval, sample a staleness step `k` and
   use base-pose-from-`k`-ticks-ago when projecting the goal pose
   into body frame for the policy observation. Cheap, additive, and
   matches the steady-state RTAB-Map cadence.
2. **Full TF buffer simulation.** Spin up a sim-side TF buffer
   mirror with publish-rate latency injected. Higher fidelity but
   significant scaffolding for a knob the policy primarily sees as a
   delay on its goal observation.

Pick Option 1 unless Phase 1 measurement shows the staleness
distribution is bimodal in a way the simple delay can't capture
(e.g. cold-start tail behaves qualitatively differently from
steady-state). Document the choice in the PR description.

#### D555 mount position randomization (new — addresses Phase 1 item 5)

The existing `randomize_d555_mount_offset` event handles orientation
only. Extend it to also sample a per-environment translation offset:

```python
# In events.py — extend randomize_d555_mount_offset
def randomize_d555_mount_offset(
    env, env_ids,
    max_angle_deg: float = 1.0,
    max_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0),  # new
) -> None:
    ...
    # Existing: roll/pitch/yaw quaternion stored on env._d555_mount_quat
    # Existing IMU obs path: ang_vel = quat_apply(env._d555_mount_quat, ang_vel)

    # NEW: per-env translation offset (meters, body frame)
    if not hasattr(env, "_d555_mount_translation"):
        env._d555_mount_translation = torch.zeros(env.num_envs, 3, device=device)
    tx = (torch.rand(num_resets, device=device) * 2.0 - 1.0) * max_translation_m[0]
    ty = (torch.rand(num_resets, device=device) * 2.0 - 1.0) * max_translation_m[1]
    tz = (torch.rand(num_resets, device=device) * 2.0 - 1.0) * max_translation_m[2]
    env._d555_mount_translation[env_ids] = torch.stack([tx, ty, tz], dim=-1)
```

And in the contracts:

```python
# REAL_ROBOT_CONTRACT — Phase 1 measurement informs ranges
randomize_d555_mount=EventTerm(
    func=mdp.randomize_d555_mount_offset, mode="reset",
    params={
        "max_angle_deg": 3.0,                     # was 1.0 — see angle row in gap table
        "max_translation_m": (0.02, 0.02, 0.01),  # NEW — ±2 cm xy, ±1 cm z
    },
)
# ROBUST_TRAINING_CONTRACT widens further: (0.03, 0.03, 0.015)
```

#### Where the position offset must propagate

Both IMU and depth-camera observation paths read from the camera
housing — both must reflect the position offset to avoid the policy
training against an inconsistent contract:

- **IMU lever-arm correction (cheap fix, do in this brief).** The
  IMU at offset `r` from the body center, under angular velocity `ω`
  and angular acceleration `α`, reads an additional
  `α × r + ω × (ω × r)` term beyond the body-frame acceleration. For
  Strafer's max rotation (~4 rad/s) at the nominal `r = (0.20, 0,
  0.25)` lever, the centripetal term is ~4 cm/s² — small but not
  zero, and crucially it *varies* with `r`. The randomized offset
  needs to flow into `imu_linear_acceleration` in
  [`observations.py:242`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
  — add the lever-arm contribution against `env._d555_mount_translation`.
- **Depth camera prim translation (harder, defer to follow-up if
  Isaac Sim runtime authoring is non-trivial).** Isaac Sim's
  `TiledCameraCfg.offset.pos` is set at scene-build time. Per-env
  runtime translation of the camera prim requires USD authoring at
  reset, which `mode="prestartup"` events can do for global edits
  but per-env requires a different pattern. **Two options:**
  1. **Per-env build-time sample.** Each parallel environment has
     its own camera prim under `{ENV_REGEX_NS}/Robot/...`. Sample
     translation once per env at scene build (extend
     `make_d555_camera_cfg` to accept an explicit offset and
     iterate over envs). The whole training run sees a *distribution*
     of camera positions across envs but each env is fixed for the
     run. Matches the real-robot semantics (fixed per mounting).
  2. **Skip depth-side position randomization; rely on IMU-side
     only.** If Option 1 turns out to be infeasible in a 1-day
     budget, drop the depth position randomization and accept the
     remaining gap — the IMU lever-arm correction is the higher-
     value piece (it shows up in every IMU tick), and the depth-
     camera 1-3 cm offset is partially absorbed by the depth
     encoder's spatial tolerance.
  Pick Option 1 if Phase 1 measurement shows >2 cm rebolt-delta on
  the real chassis; pick Option 2 otherwise. Document the choice in
  the PR description.

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
| Eval at the depth-hold band's upper edge | success rate | success rate | DR-audit > baseline |
| Eval at +1 cm D555 offset on each axis | success rate | success rate | DR-audit > baseline |

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
      (median + p95), control-loop jitter (P50/P95/P99), **D555
      mount position** (absolute `(x, y, z)` vs.
      `strafer_shared.constants` nominal, plus rebolt-cycle delta),
      and **TF staleness on the goal-pose body-frame transform**
      (median + P95 + P99, for both nominal-tracking and cold-start
      regimes).
- [ ] If the measured absolute D555 position differs from the
      `CAMERA_OFFSET_X/Y/Z` constants by more than the rebolt-delta
      itself, update `strafer_shared.constants` in the same commit
      (the nominal is the wrong center for the randomization
      distribution). This is the additive-only `strafer_shared`
      exception path; values cannot be removed or renamed.

### Config

- [ ] `create_real_robot_contract()` updated in
      [`sim_real_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py)
      with each change citing a Phase 1 row.
- [ ] `_RANDOMIZE_PROC_ROOM_DIFFICULTY` extended from
      `(7, 7)` to a graduated range covering at least 3 levels.
- [ ] `randomize_d555_mount_offset` extended in
      [`events.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py)
      to accept `max_translation_m: tuple[float, float, float]` and
      store the per-env offset on `env._d555_mount_translation`.
- [ ] `imu_linear_acceleration` in
      [`observations.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
      adds the lever-arm contribution
      (`α × r + ω × (ω × r)` against `env._d555_mount_translation`)
      so the IMU obs reflects the randomized mount position.
- [ ] Depth-camera position randomization implemented per Option 1
      (per-env build-time sample) OR Option 2 (skip; rely on IMU side
      only) per the Phase 2 decision rule; choice documented in the
      PR description.
- [ ] TF staleness randomization implemented per Option 1 (sampled
      per-tick replay of a delayed base pose) OR Option 2 (full TF
      buffer simulation) per the Phase 2 decision rule; choice
      documented in the PR description. Affects the policy
      observation path that consumes the goal-pose body-frame
      projection, NOT the depth or IMU obs paths. Scoped to the
      reference's **age**: its displacement landed separately as
      `LocalizationDriftCfg`, whose `tau_s` and zero jump band are the
      two numbers Phase 1 item 6 now also owes.
- [ ] All unit tests under `source/strafer_lab/tests/` still pass —
      contract changes are config-only; no API change.

### Training + evaluation

- [ ] DR-audit checkpoint exists at
      `logs/rsl_rl/strafer_navigation/depth_dr_audit_v1/model_<step>.pt`
      with a sidecar JSON noting `baseline_checkpoint` provenance.
- [ ] PR description includes the comparative evaluation table:
      baseline vs DR-audit, evaluated at original DR and at +50% mass /
      depth-latency / depth-hold. DR-audit must show ≥ 10% success-rate
      improvement on stress cells and ≤ 5% degradation on the original
      DR cell.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Capture-corpus implication of mount-DR widening (added 2026-06-23)

This brief widens `randomize_d555_mount_offset` (orientation 1°/2° → 3°, plus a new `max_translation_m` ±2 cm) for trained-policy robustness. That widening also has a **dataset** consequence. Split camera-affecting DR into two classes:

- **Appearance DR** (sensor/depth noise, exposure, blur) — moves the pixel output, not the geometry. No label impact; no logging needed.
- **Geometric DR** (this brief's mount extrinsic jitter) — moves the camera ray, making the camera-to-world transform episode-specific.

At the *current* ±2° rotation-only setting the camera effect is ~1–2 px (negligible — a captured corpus can derive pose-dependent labels from the nominal mount). **Once this brief raises the angle and adds translation, the realized mount diverges enough that any pose-derived label in a captured corpus** (region GT, "what's in view", depth back-projection, VPR pose-keying) **would be corrupted if derived from the nominal mount.** Therefore the bulk-capture corpus **logs the realized mount per episode** (`env._d555_mount_quat`, plus `env._d555_mount_translation` once it lands here) — implemented by [`coverage-capture-driver`](../../active/harness/coverage-capture-driver.md), per the 2026-06-23 data-requirements analysis (geometric camera DR is the consumer that the "no committed consumer" assessment assumed away). Keep the two briefs' mount-DR knobs in sync.

## Investigation pointers

- [`source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py)
  `create_real_robot_contract()` (lines 425-484) — the edit site.
- [`source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py)
  `_RANDOMIZE_PROC_ROOM_DIFFICULTY` (around line 1649) — the proc-room
  level config.
- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py:450`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py)
  — `randomize_d555_mount_offset` (orientation only today; extension
  site for position).
- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py:242`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
  — `imu_linear_acceleration` (the IMU obs path that consumes the
  current `_d555_mount_quat`; needs to also consume
  `_d555_mount_translation` for the lever-arm correction).
- [`source/strafer_shared/strafer_shared/constants.py`](../../../../source/strafer_shared/strafer_shared/constants.py)
  — `CAMERA_OFFSET_X/Y/Z` — the nominal position the Phase 1
  measurement compares against; the additive-only edit site if the
  nominal is wrong.
- [`source/strafer_lab/strafer_lab/tasks/navigation/d555_cfg.py:120`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/d555_cfg.py)
  — `make_d555_camera_cfg` — where the camera's `OffsetCfg.pos` is
  fixed at scene-build time (relevant for Option 1 of the depth-camera
  position randomization decision).
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
  [`hybrid-mode`](../../completed/hybrid-mode.md); a
  parallel DR audit for NoCam_SUBGOAL can follow once that lane is
  active.
- **Cross-room / multi-room scene randomization.** That's the
  multi-room epic ([`multi-room/`](../../active/multi-room/)). This
  brief stays inside the single-room ProcRoom distribution.
- **Applying the mount offset to the rendered camera.**
  [`procroom-depth-enrichment`](procroom-depth-enrichment.md) owns that
  (its F4, coordinator-routed 2026-07-25) and has shipped it on the
  enriched depth variants: the camera prim is pointed through the offset
  `randomize_d555_mount_offset` already samples, so the render and the IMU
  observation carry one misalignment instead of two. The **band** is still
  this brief's — F4 adds none of its own, so widening the ±1°/±3° row
  above now moves both sensors at once, and the mount-*translation*
  extension proposed here remains unimplemented on both.
