# Spike a MuJoCo-Warp / Newton physics backend for the mecanum-roller contact model

**Type:** investigation / research
**Owner:** DGX agent
**Priority:** P3 (research bet; additive — the high-yaw roller bounce is
already fixed in PhysX via PGS, so this is not blocking and not a fix,
it is a feasibility probe of an alternative engine)
**Estimate:** M (time-boxed standalone spike — convert + load + one
scripted spin + write-up; deliberately bypasses the Isaac Lab env stack)
**Branch:** task/mujoco-warp-physics-backend-spike

## Story

As a **sim engineer responsible for the Strafer's contact fidelity**, I
want **a measured feasibility read on whether MuJoCo-Warp / Newton's
soft-contact model handles the 40-roller mecanum drivetrain better than
PhysX**, so that **a future decision about a physics-backend migration is
grounded in evidence rather than intuition, and the already-installed
Newton/MuJoCo libraries don't sit unexamined.**

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

This brief was spun out of the
[`roller-contact-high-omega-bounce`](../../active/sim-performance/roller-contact-high-omega-bounce.md)
investigation. That work found the high-yaw chassis bounce is the **TGS
solver injecting spurious velocity into the near-massless, free-spinning
mecanum rollers** (over-spinning them several-fold), and fixed it by
switching the shared nav `PhysxCfg` to **PGS** (`solver_type=0`) at the
native rate — contract-green, no perf penalty. PhysX exposes only PGS and
TGS (`solver_type` is `Literal[0, 1]`), so that lever is now exhausted.

While doing that, we found the env ships an entire **second engine** that
is installed but **not wired into Isaac Lab's env layer**:

- `newton` 1.0.0, `mujoco` 3.5.0, `mujoco_warp` 3.5.0.2,
  `newton_actuators`, `newton_usd_schemas`, `mujoco_usd_converter`
  (all importable from `env_isaaclab3`).
- There is **no** `isaaclab_newton` backend package, `SimulationCfg` has
  no backend selector (it is PhysX-bound: `physics_prim_path /
  physics_material / physics`), and nothing in `isaaclab` imports Newton
  as a sim backend. So Newton/MuJoCo here are standalone libraries +
  asset tooling, not a drop-in flag.

Why it is worth a look for *this* robot specifically: MuJoCo uses a
**compliant / soft-constraint contact model** with a different solver
family (`mjSOL_CG`, `mjSOL_NEWTON`, `mjSOL_PGS`), and is well regarded for
stable **contact-rich** simulation — many small simultaneous contacts,
exactly the mecanum-roller regime that tripped up PhysX TGS. The question
this spike answers: does MuJoCo's contact model represent the rollers more
naturally (no high-yaw bounce, physical roller speeds, smooth handoff)
without the per-body solver pathologies?

This is **exploratory and standalone**. It does not touch the shipped
PhysX/PGS fix and does not migrate the `strafer_lab` env. The deliverable
is a measurement + a go/no-go read on whether a deeper integration spike
is ever worth it.

## Acceptance criteria

- [ ] Convert the Strafer USD
      (`Assets/strafer/3209-0001-0006-physics.usd`, or the
      `-no-physics.usd` source) to a MuJoCo model via
      `mujoco_usd_converter`. Record what survives the conversion and what
      does not — the 4 drive joints, the 40 passive roller revolute
      joints, collision geometry (the SDF roller barrels vs MuJoCo's mesh
      collision), masses, friction/material — and what had to be
      reconstructed by hand.
- [ ] Load the converted model under `mujoco_warp` on a clean ground
      plane and run a scripted **max-yaw spin-in-place** (left/right wheels
      opposed at the 32.67 rad/s wheel limit), logging chassis-z at the
      solver substep rate — mirroring the metrics in
      `Scripts/roller_bounce_probe.py` (late peak-to-peak, growth ratio,
      dominant frequency, achieved wheel + roller speeds).
- [ ] Report the comparison vs the PhysX reference (PhysX TGS ≈ 16.7 mm
      growing bounce; PhysX PGS ≈ 1.1 mm flat at 120 Hz): does MuJoCo
      exhibit the high-yaw bounce? Are roller speeds physical? Is the
      contact handoff smooth? Note which MuJoCo solver (`CG` / `Newton` /
      `PGS`) and contact parameters were used.
- [ ] Land a written go/no-go verdict on whether a Newton/MuJoCo backend
      is worth a deeper integration spike — explicitly listing the feature
      gaps a real migration would have to close (sensors, camera/render,
      SDF mesh collision, domain randomization, the `ManagerBasedRLEnv`
      managers, training throughput). A negative result ("PhysX-PGS is
      sufficient; MuJoCo not worth the migration") is a valid, useful
      filed conclusion.
- [ ] If your work invalidates a fact in any referenced context module,
      package README, top-level `Readme.md`, or guide under `docs/`,
      update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`roller-contact-high-omega-bounce`](../../active/sim-performance/roller-contact-high-omega-bounce.md)
  — the parent investigation: the TGS-velocity-noise mechanism, the PGS
  fix, and the ruled-out levers.
- `Scripts/roller_bounce_probe.py` — the PhysX-side reference harness;
  reuse its spin pattern and metrics so the MuJoCo numbers are directly
  comparable. (Side map: wheel_1/wheel_3 = +y left, wheel_2/wheel_4 = −y
  right; rollers 10/wheel at 36°, ring radius 40 mm, wheel radius ~48 mm.)
- Installed libraries (from `env_isaaclab3` site-packages): `newton`,
  `mujoco`, `mujoco_warp`, `mujoco_usd_converter`, `newton_usd_schemas`,
  `newton_actuators`.
- MuJoCo solver options: `mujoco.mjtSolver` → `mjSOL_CG`, `mjSOL_NEWTON`,
  `mjSOL_PGS`.
- `Scripts/setup_physics.py` — how the rollers are authored in PhysX
  (revolute joints, SDF collision on the barrel covers, rubber material);
  useful for judging what the USD→MuJoCo conversion should preserve.

## Out of scope

- **Migrating the `strafer_lab` env stack to Newton/MuJoCo.** This spike
  produces a feasibility verdict, not an integration.
- **Changing the shipped PhysX/PGS fix.** The bounce is solved; do not
  revisit it here.
- **Sensor / camera / render parity, domain randomization, or training
  under MuJoCo.** Note the gaps as part of the verdict; do not close them.
- **Wiring MuJoCo into `ManagerBasedRLEnv` / `SimulationCfg`.** The spike
  runs standalone against `mujoco_warp` directly.

## Trigger (why this is parked, not active)

Filed-on-trigger — do not pick up pre-emptively. Pick up when one of:

- the PhysX-PGS roller fix shows limitations in a full-length training run
  or in deployment (contact fidelity becomes a sim-to-real blocker), **or**
- Isaac Lab ships first-class Newton env integration (an `isaaclab_newton`
  backend / a `SimulationCfg` backend selector), which would collapse most
  of the migration cost this spike exists to estimate, **or**
- there is dedicated appetite to evaluate a physics-engine direction for
  the broader contact-heavy robot work.
