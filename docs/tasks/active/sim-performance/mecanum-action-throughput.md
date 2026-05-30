# Mecanum action throughput — slow rotation + sluggish diagonals

**Type:** investigation + tuning (sim-real action contract)
**Owner:** DGX agent
**Priority:** P2 — degrades teleop UX (sluggish, ergonomically frustrating) and quietly compresses the action-space distribution the harness captures into datasets. Not blocking any current acceptance bar.
**Estimate:** M (~one focused day: profile the per-stage cost of the action pipeline, A/B individual stages to isolate the dominant damper, then either retune or document the inherent cap).
**Branch:** `task/mecanum-action-throughput`

## Story

As a **teleop operator commanding the strafer through Isaac Sim** I want **commanded rotation rates and diagonal translations to feel responsive — within a small fraction of the documented mecanum dynamics** so that **demonstrations capture the action distribution a downstream policy will actually see, instead of a damped sub-band of it, and so the operator doesn't have to over-press the stick for several seconds to get the chassis to actually turn.**

## Motivation — observed in PR #63

After the world_arcade stick direction fix (`teleop_capture.py:_robot_pose` quaternion XYZW unpacking), the rotation command path was traceable end-to-end. Two related symptoms surfaced:

1. **Rotation is ~70× slower than the documented cap.** With `omega = +1.0` (normalized max) the env's MecanumWheelAction docstring + init log advertise `Max angular vel: 4.19 rad/s ≈ 240°/sec`. A controlled pure-rotation test (right-stick only) measured ~3.4°/sec sustained angular velocity. **Two orders of magnitude under spec.**

2. **Diagonal translation feels sluggish.** Operator did not quantify but reported this as the same family of complaint. Plausibly the L1 clamp in `MecanumWheelAction.process_actions` (`actions.py:299-311`) scales `(|vx| + |vy|)` jointly under a single linear-velocity cap, so any non-axis-aligned stick reduces effective per-axis throughput.

3. **Spinning-like-a-top while translating is imperfect.** Combining nonzero translation + nonzero rotation produces lower-quality curves than either alone. Likely the same L1 clamp damping each component when the L1 norm exceeds the cap.

## Suspected dampers (in order of likelihood)

The action passes through five stages between operator stick and PhysX wheel torque, each of which could account for some fraction of the observed damping (`source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py:282-354`):

1. **L1 clamp** (`l1_clamp_twist_batched`, lines 307-311) — caps `|vx| + |vy| + angular-equivalent` jointly so the chassis cannot exceed deployment-time per-wheel motor limits. For a max-stick pure-rotation command the clamp probably attenuates lightly, but for max-stick translation + rotation it scales each component down significantly. **This explains the diagonal + spin-while-translating sluggishness directly.** May or may not also damp pure rotation.
2. **Slew rate limit** (lines 336-340) — `max_delta_per_step` caps per-tick wheel-velocity change. From init: `max_accel=900 rad/s², range=(700.0, 1100.0)`. At 30 Hz env step that's 900/30 = 30 rad/s per step — comparable to the wheel's max angular vel (32.67 rad/s), so slew should ramp to max in one step. Unlikely to be the dominant damper, but A/B with this disabled is cheap.
3. **Motor dynamics** (first-order filter, lines 342-347) — `motor_alpha` controls a first-order LP with documented `tau=0.05s`. At 30 Hz step that's α ≈ 0.5 per step — so the wheel reaches ~63% of commanded in one step, ~95% in three. Adds time-to-reach-max but not steady-state attenuation. Unlikely the dominant damper.
4. **Command delay** (lines 331-333) — `_action_delay_buffer` with `steps=1-3`. Latency 33-100 ms, doesn't attenuate magnitude. Not a damper, but compounds with motor dynamics.
5. **Domain randomization on motor strength** (`randomize_motor_strength` event in `EventsCfg`) — at reset each env samples a motor-strength multiplier. If the random distribution sampled low for the operator's run, all wheel torques are scaled down → both translation and rotation are slow. **Could account for 70× damping if sampled at the low tail.** Per-episode and resets between tests, so reproducible from a fixed seed but variable across operator sessions.

## Acceptance

Ship one of (a)/(b) below:

- **(a) A retune that gets pure-rotation throughput within 2× of spec** (≥120°/sec at `omega=1.0` on a fresh reset). Tune one or more of: L1 clamp policy, slew limit, motor_alpha, or randomize_motor_strength's distribution. Preserve the sim-real contract (the same dynamics still apply to the trained policy's action stream) — document any contract change in the PR.
- **(b) A documented inherent cap** explaining why each stage in the pipeline is at its current value, why retuning would break the sim-real contract, and what the achievable upper bound is. In this case, file a follow-up `teleop-action-curve-shaping` brief proposing a stick-side response curve (e.g. apply a 2× gain on omega in teleop mode only, bypassing the L1 clamp on the teleop driver's side) so the OPERATOR's stick feels responsive without changing the policy-facing dynamics.

Acceptance checklist:

- [ ] Per-stage cost profile of `MecanumWheelAction.process_actions` measured with each stage individually disabled (5 runs minimum). Report sustained omega at `omega=1` with each stage on/off.
- [ ] Identification of the dominant damper(s) by absolute attenuation fraction.
- [ ] Per-stage cost profile repeated for diagonal translation (`vx=+1, vy=+1, omega=0`) to confirm the diagonal-sluggish + spin-while-translating claims share a root cause with the slow-rotation one.
- [ ] Either (a) a tuning PR closing the gap to 2× of spec, OR (b) a documented cap + follow-up brief filed.
- [ ] If (a): re-run any trained policy's strafer-direct-sim-validation to confirm the contract change didn't regress the policy's task success rate.

## Out of scope

- Re-architecting the mecanum action term itself. Keep `MecanumWheelAction.process_actions` API stable.
- Per-controller dynamics differences (e.g. spinning vs translating at different aggressiveness in the trained policy vs the teleop driver). The investigation surfaces the cap; the teleop-side response shaping is the proposed follow-up if (b) is chosen.
- Diagnosing the docstring's `4.19 rad/s` itself — assume the documented max is correct.
- The slow-omega issue's interaction with the `subgoal-env` or `goal-noise-training` briefs. They're orthogonal.

## Triggered by

PR #63 round-of-rounds operator testing on the harness teleop driver. The quaternion XYZW fix in `_robot_pose` unblocked accurate world→body rotation, which immediately surfaced this as the next bottleneck. Operator quote: *"moving at diagonals is a bit sluggish, and rotating like a top while moving in any direction is imperfect."*

## Hints for the implementer

- To reproduce the measurement: re-add a once-per-second `[diag]` print in `teleop_capture.py`'s env-step loop that logs `(yaw_deg, body_cmd, world_pos)`. Push right-stick only; the yaw delta between consecutive diag prints divided by the print interval gives sustained angular velocity.
- To A/B individual pipeline stages, the cleanest approach is to temporarily comment out one block at a time in `MecanumWheelAction.process_actions` lines 307-348 and re-measure. Don't edit those defaults in tree until the dominant damper is named.
- If `randomize_motor_strength` is the culprit, the distribution lives in the `EventsCfg` for the active env variant — `_BaseInfinigenPerceptionNavEnvCfg` for teleop.
- The `policy-rate-shared-constants` brief lives in trained-policy and proposes the policy's training-side dt as a shared constant — if (a) is chosen and decimation changes, cross-reference that brief.
