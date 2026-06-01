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

- [ ] Reproduce the bounce headlessly or via a scripted max-yaw spin so
      it is measurable without a human at the stick (e.g. contact-force
      or chassis-z trace over a fixed spin), giving an objective
      before/after signal.
- [ ] Test the contact-offset experiment; record whether it removes or
      reduces the bounce.
- [ ] Either land a fix (USD regen + the offset/geometry change, with a
      teleop spin confirming the bounce is gone and a check that normal
      driving + the trained-policy contact behavior are unaffected) OR
      document the high-omega regime as a known modeling limit with the
      ruled-out causes above and a recommended yaw clamp.

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
