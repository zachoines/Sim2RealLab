# Roller contact instability at high body-rotation rate

**Type:** investigation (robot collision model)
**Owner:** DGX agent
**Priority:** P3 — teleop-cosmetic, not blocking. The chassis develops a
growing vertical bounce only at sustained near-max yaw rate (operator
spinning in place); normal driving and capture are unaffected. Filed so
the ruled-out causes are not re-investigated from scratch.
**Estimate:** M — a contact-model investigation + one or two USD-regen
experiments; no guaranteed fix (may resolve as a documented modeling
limit).
**Branch:** `task/roller-contact-high-omega-bounce`

## Story

As a **DGX operator teleoperating the Strafer** I want **the chassis to
stay planted when I spin it in place at max yaw rate** so that **a
spin-heavy maneuver doesn't pop the robot off the floor and the captured
trajectory stays physically plausible**.

## Symptom

At sustained near-maximum body-rotation rate (full right-stick, spinning
like a top), the chassis develops a periodic vertical jolt that **builds
in intensity** over a few seconds. At low and medium drive/rotation
speeds it is absent or negligible. It is a robot-internal contact
effect — independent of scene, present on the clean ground plane.

## What has been ruled out (measured, do not re-investigate)

This was investigated at length during `teleop-perf-architecture`. None
of the following changed the high-omega bounce:

- **Restitution.** The robot USD authored each roller's `RubberMaterial`
  with `restitution=0.12`; the ground plane is `restitution=0.0`. The
  asset was regenerated with **roller restitution = 0.0 and
  `restitutionCombineMode=min`** (verified on all 40 roller materials) —
  **no change** to the bounce. So it is not classic restitution.
- **Solver iteration count.** Tested at the shared 32/16 and down to
  `--phys-solver-pos-iters 4 --phys-solver-vel-iters 1` — **no change**.
  (8/4 is now the shared default for being cheaper, not for this bug.)
- **Depenetration velocity.** `--phys-max-depenetration-vel 0.5` (and
  1.0) — **no change**. Not a depenetration-ejection effect.
- **Contact stabilization.** `enable_stabilization=True` (now shared) +
  `--phys-stabilization-threshold` — helps general jitter at normal
  speed but **does not** remove the high-omega bounce.
- **Decimation / substep size.** Covered in `teleop-perf-architecture`:
  decimation 1 destabilizes the rollers generally (different failure);
  decimation 2/4 both show the high-omega bounce.

## Leading hypothesis

**Discrete-roller contact hand-off at high tangential speed.** Each
mecanum wheel is modeled as 10 separate roller bodies with gaps between
them, rather than a continuous contact surface. At high body-rotation
rate the ground-contact point transfers between adjacent discrete
rollers fast enough that the contact effectively skips the inter-roller
gaps, producing periodic vertical impulses that resonate and accumulate
— consistent with "builds in intensity" and with the negative results
above (it is a *geometry/contact-continuity* effect, not a
solver-convergence or energy-restitution one). A real robot does not
exhibit this because its rollers have continuous bearing contact.

This is a hypothesis, not a conclusion — the investigation is to confirm
or refute it.

## Findings (investigation outcome)

**Confirmed mechanism: temporal contact under-resolution at the shared
`sim.dt = 1/120`. Discrete rollers are the excitation source, but the
growing bounce is a time-integration instability of that excitation, not
a contact parameter and not the inter-roller gaps. Disposition: documented
modeling limit at high yaw rate; no shared-physics change landed.**

Measured with a scripted headless spin-in-place harness
(`Scripts/roller_bounce_probe.py`): it loads the shared `STRAFER_CFG` on a
clean ground plane, drives a pure yaw spin (left/right wheels opposed) at a
sweep of wheel speeds, and logs chassis z at the 120 Hz physics substep
rate — the 30 Hz policy loop aliases the effect away, so substep-rate
logging is required to see it. All runs use the shared nav physics
(`sim.dt = 1/120`, `enable_stabilization=True`); chassis yaw is held ≈ 0
so the vertical motion is purely drivetrain-induced.

### What the bounce is — wheel-speed sweep (5 s each)

| wheel speed | roller-pass freq | chassis-z late p2p | growth | dominant z freq |
|---|---|---:|---:|---|
| 25 % (8 rad/s)  | 12.6 Hz | 0.4 mm | — | 12.8 Hz (= roller-pass) |
| 50 % (16 rad/s) | 24.7 Hz | 1.9 mm | — | 25.6 Hz (= roller-pass) |
| 65 % (21 rad/s) | 31.9 Hz | 6.4 mm | 1.2× | 3.4 Hz (mode switch) |
| 75 % (23 rad/s) | 36.5 Hz | 7.9 mm | 1.6× | 3.8 Hz |
| 90 % (28 rad/s) | 43.8 Hz | 16.8 mm | — | low-freq |
| 100 % (31 rad/s)| 48.9 Hz | 16.7 mm | 2.2× | 4–7 Hz |

Two regimes:
- **Below ~60 % wheel speed** the only vertical motion is a sub-mm ripple
  **exactly at the roller-passing frequency** (10 × wheel-rev). This is the
  genuine discrete-roller contact undulation — the hypothesis's predicted
  fingerprint, and it is real and benign.
- **At/above ~65 % wheel speed (onset)** the response switches to a large,
  growing **low-frequency (~3–7 Hz) chassis bounce** that lifts the chassis
  ~18 mm off rest and settles into a sustained limit cycle (≈17 mm
  peak-to-peak, stable over a 10 s run). This is the reported symptom.

### Root cause — the decisive substep-rate test (100 % wheel speed)

| substep rate | chassis-z late p2p | growth | chassis lift | residual freq |
|---|---:|---:|---:|---|
| **120 Hz (shared)** | 16.7 mm | 2.2× | +18 mm | 4–7 Hz pumped mode |
| 240 Hz | 2.2 mm | 1.0× | ~0 | none growing |
| 480 Hz | 0.5 mm | 1.1× | ~0 | 52 Hz (pure roller-pass ripple) |

Halving the timestep removes the growth entirely; at 480 Hz only the benign
sub-mm 52 Hz roller-pass ripple remains. At 120 Hz the contact patch
advances ~12 mm per substep at full speed (≈ half the 25 mm inter-roller
spacing → only ~2 substeps per roller engagement); PhysX integrates the
fast-moving discrete-roller contact too coarsely and injects energy each
cycle, pumping a low-frequency chassis vertical mode. This unifies every
prior negative result: a sampling-rate problem cannot be fixed by any
material/solver **coefficient**, it worsens monotonically with speed, and a
bigger substep (decimation 1) made it worse.

### Candidate fixes tested and rejected

- **Inter-roller gap bridging (collider `contact_offset` / `rest_offset`).**
  `contact_offset 0.04` reduced p2p only ~13 % (growth 2.2→1.7×);
  `rest_offset 0.003` did not help (growth 2.2→2.8×). Inflating colliders to
  bridge the gaps does **not** remove the bounce — the literal "contact
  skips the gaps" framing is not the lever.
- **Roller passive-joint damping.** No effect at any value (0.1 / 0.5 / 2.0
  gave byte-identical results) because `roller_bearings` has
  `effort_limit_sim = 0.0`, which clamps the actuator damping torque to
  zero. The roller joints are truly passive (only contact friction acts);
  roller joint damping is a no-op, and the earlier 0.5 → 0.01 change was
  likewise inert.

### Disposition: documented modeling limit + recommended mitigation

The only effective fix is a smaller substep (≥ 240 Hz), but `sim.dt` is the
**shared physics contract**: halving it doubles physics cost on every mode,
undoing the per-substep cost win and adding 2× physics cost to RL training —
not justified for a P3 teleop-cosmetic effect that leaves normal driving and
capture untouched. No geometry change works; roller damping is inert. **No
change is landed to `STRAFER_CFG` or the robot USD; the RL composition
contract is unaffected (all 7 golden hashes verified green).**

Recommended mitigations, cheapest first, neither touching the shared physics
contract:
1. **Teleop yaw clamp (recommended, zero cost).** Onset is ~60–65 % of max
   wheel speed. Clamp the teleop yaw command to `|omega| ≤ ~0.6` in
   `_stick_to_body_action`
   ([`source/strafer_lab/scripts/teleop_capture.py`](../../../../source/strafer_lab/scripts/teleop_capture.py)) —
   the teleop driver, **not** the mecanum action term. Only full-stick
   top-spin is affected, and real missions rarely sustain it.
2. **Teleop-only 240 Hz substep (only if full-yaw fidelity is ever needed).**
   Fully suppresses the bounce at 2× physics cost; apply on a teleop
   variant's `sim.dt` only, never the RL/shared default.

### Reusable artifact

`Scripts/roller_bounce_probe.py` — the headless characterization harness;
re-run for an objective before/after chassis-z trace and bounce-frequency
readout at any wheel speed / substep rate / collider-offset / roller-damping
setting.

## Candidate directions (for the agent who picks this up)

- **Contact offset / rest offset** on the roller colliders, sized to
  bridge the inter-roller gap so the contact patch stays continuous as
  it hands off. Cheapest experiment; one USD regen
  (`Scripts/setup_physics.py`) + a teleop spin test.
- **Roller collider geometry** — convex roller shape, count, or a
  single swept/torus approximation of the contact band instead of 10
  discrete bodies. Heavier; changes the wheel's contact model.
- **Roller joint damping** — the passive roller bearings use
  `damping=0.5` (`assets/strafer.py`); whether the bounce couples to the
  passive-joint dynamics at high spin is worth a sweep.
- **Accept as a modeling limit** — if no cheap fix holds, document the
  high-omega regime as a known sim artifact and (optionally) clamp the
  teleop yaw command below the onset rate.

## Acceptance

- [x] Reproduce the bounce headlessly or via a scripted max-yaw spin so
      it is measurable without a human at the stick (e.g. contact-force
      or chassis-z trace over a fixed spin), giving an objective
      before/after signal. → `Scripts/roller_bounce_probe.py`.
- [x] Test the contact-offset experiment; record whether it removes or
      reduces the bounce. → only marginal (~13 %); does not remove it.
- [x] Either land a fix (USD regen + the offset/geometry change, with a
      teleop spin confirming the bounce is gone and a check that normal
      driving + the trained-policy contact behavior are unaffected) OR
      document the high-omega regime as a known modeling limit with the
      ruled-out causes above and a recommended yaw clamp. → documented as
      a modeling limit (see Findings); recommended teleop yaw clamp.

## Out of scope

- **The validated teleop-perf physics win** (solver iters 8/4 +
  `enable_stabilization`, now shared) — already landed in
  `teleop-perf-architecture`; this brief does not revisit it.
- **Diagonal-motion sluggishness** — a separate effect (zero-commanded
  wheels held by the velocity actuator's `damping`, an effective brake;
  real Strafer braking is off by default). Belongs with
  `mecanum-action-throughput` / the actuator model, not here.
- **Renderer / FPS perf** — `teleop-perf-architecture` and siblings.

## Triggered by

`teleop-perf-architecture`'s physics investigation: after the
solver-iters/stabilization win and the roller-restitution USD
regeneration, the high-omega bounce persisted unchanged, isolating it
from every contact-*parameter* lever and pointing at the roller contact
*geometry/model* — a distinct investigation filed here rather than
expanding the teleop-perf brief's scope.
