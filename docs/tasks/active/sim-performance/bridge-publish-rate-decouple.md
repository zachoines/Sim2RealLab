# Decouple sim-in-the-loop sensor publish rate from physics/control decimation

**Type:** task / investigation + refactor (bridge mainloop)
**Owner:** DGX agent
**Priority:** P2 — fidelity of the sim-in-the-loop validation path. The
bridge currently has to choose between training-consistent control
dynamics and a realistic sensor publish rate to the Jetson; it cannot
have both. Not blocking (the bridge runs today), but every harness/bridge
validation run inherits whichever side was sacrificed.
**Estimate:** M (a measurement pass + a scoped bridge-mainloop refactor;
the candidate "hold the action across N substeps" approach may be small,
the general decoupling is larger)
**Branch:** task/bridge-publish-rate-decouple

> **Camera-cadence slice delivered separately.** The camera-publish-cadence
> half of this brief — making the *default* `--camera-frame-skip` correct for
> any decimation (derived to one publish per policy period, not a fixed 3) —
> shipped in
> [`bridge-camera-cadence-derive`](../../completed/bridge-camera-cadence-derive.md).
> What stays here is the deeper decouple: publishing *more often* than the
> control step (action-hold / sub-step pump) so `decimation 4` dynamics and a
> ~30 Hz publish rate coexist architecturally.

## Story

As a **sim-in-the-loop operator validating the Jetson autonomy stack**, I
want **the bridge to simulate the robot with the same control/motor
dynamics as training (decimation 4) while still publishing camera/odom to
the Jetson at ~25–30 Hz**, so that **a bridge/harness run reflects the
physics the deployed policy was trained against AND Nav2/SLAM on the Jetson
receive sensor rates close to the real D555 — instead of trading one for
the other.**

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md)
- [`context/conventions.md`](../../context/conventions.md)

## Context

Surfaced while making the nav physics shared/consistent in
[`roller-contact-high-omega-bounce`](../../completed/roller-contact-high-omega-bounce.md)
(PR #76): the PGS solver fix is now shared across training, teleop-capture,
and the bridge, but `run_sim_in_the_loop.py` still diverges on **control
rate**. It defaults `--decimation 1` (and `--render-interval 1`), overriding
the shared nav defaults (`decimation 4`, `render_interval 4`); `sim.dt`
(1/120) and the solver are inherited.

`decimation` couples two cadences that should be independent:

| `decimation` | control / action-update rate | motor dynamics | Jetson publish rate (wall-clock) |
|---|---|---|---|
| **1** (bridge default) | 120 Hz | **untuned** (training tunes for 30 Hz / 33 ms step) | **~29 Hz** (≈ real D555) |
| **4** (training / shared) | 30 Hz | training-consistent | **~8 Hz** (too low for Nav2/SLAM) |

So the current default buys a realistic publish rate at the cost of
running control/motor dynamics the trained policy and the real RoboClaw
never see; switching to `decimation 4` fixes the dynamics but starves the
Jetson at ~8 Hz. Real-time factor is ~unchanged either way (~0.25×), since
per-substep cost dominates (≈29 substeps/s vs ≈8×4=32 substeps/s) — this is
**not** a sim-speed tradeoff, it is a publish-cadence-vs-control-rate one.

`decimation=1` is presently a documented invariant
([`bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md),
the "Default decimation in bridge" note, perf fix `74979d6`) — this brief
revisits it, so that note must be updated by whoever lands the change.

### Why it is a real decoupling (the mechanism)

The camera/telemetry publishers run on an **OmniGraph** driven by
`OnPlaybackTick`, and the bridge mainloop pumps Kit (`simulation_app.update`)
**once per `env.step`** (the `kit-pump-redundancy-investigation` reduced it
to a single pump). So a publish fires once per control step → publish rate
is bolted to the env-step rate → bolted to `decimation`. `--camera-publish-
every` only *divides* that rate down (and defaults to match `render_interval`
so duplicate frames aren't pushed). There is no current path to publish
*more often* than the control step, so this is an architectural change, not
a flag.

## Approach (options to evaluate — pick per measurement)

- **Action-hold within a decimation-1 loop (likely cheapest).** Run the
  mainloop at `decimation 1` (Kit pump + publish stay ~29 Hz) but apply each
  incoming `cmd_vel` / action **held across 4 substeps** before accepting a
  new one — making the *effective* action-update rate 30 Hz (training-
  equivalent) while publish stays per-substep. Verify this reproduces
  decimation-4 motor dynamics closely enough.
- **Sub-step Kit pump at decimation 4.** Keep `decimation 4` but pump
  Kit / fire the publisher OmniGraph more than once per `env.step`. Must
  confirm frames aren't stale (render product only advances with physics)
  and quantify the added pump cost.
- **Explicit publish-cadence knob** decoupled from `decimation` entirely,
  so control rate and publish rate are set independently.

## Acceptance criteria

- [ ] The bridge runs **training-equivalent control/motor dynamics**
      (action-update / substep structure matching `decimation 4`) **and**
      publishes camera + odom to the Jetson at **≥ ~25 Hz** wall-clock.
      Show both with a measurement (control-rate + publish-rate trace).
- [ ] `make sim-bridge` and `make sim-harness` smoke-launch and the Jetson
      side receives the higher publish rate (no ~8 Hz starvation).
- [ ] No regression to the in-process harness drivers (teleop / scripted /
      bridge-driver) or to the LeRobot capture cadence.
- [ ] Update [`bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md)
      (the decimation note) and `run_sim_in_the_loop.py`'s `--decimation` /
      `--render-interval` help to reflect the new model.
- [ ] If your work invalidates a fact in any referenced context module,
      package README, top-level `Readme.md`, or guide under `docs/`, update
      those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).

## Investigation pointers

- `source/strafer_lab/scripts/run_sim_in_the_loop.py` — the bridge mainloop,
  `_apply_sim_in_the_loop_overrides`, and the `--decimation` /
  `--render-interval` / `--camera-publish-every` flags.
- `source/strafer_lab/strafer_lab/bridge/async_camera_publisher.py`,
  `bridge/async_publisher.py` — the OmniGraph publishers fired on
  `OnPlaybackTick`.
- [`completed/kit-pump-redundancy-investigation.md`](../../completed/kit-pump-redundancy-investigation.md)
  — the single-Kit-pump-per-loop state this builds on.
- [`bridge-throughput-toward-25hz`](bridge-throughput-toward-25hz.md) —
  **coordinate, do not overlap.** That brief makes each `env.step`
  *cheaper* (trims the `simulation_app.update` residual + manager loop) to
  raise overall loop rate; this brief decouples publish *cadence* from
  control *decimation*. They both touch the mainloop + Kit pump — land
  aware of each other.

## Out of scope

- **Per-step throughput trimming** — owned by `bridge-throughput-toward-25hz`.
- **The RL training cfg** — control rate / decimation there is unchanged.
- **The solver fix** — shared PGS landed in PR #76; not revisited here.
- **`strafer_ros` / the Jetson side** — this is the DGX bridge mainloop;
  the Jetson consumes whatever rate it is given.

## Triggered by

`roller-contact-high-omega-bounce` (PR #76): sharing the nav physics via PGS
made every mode use the same solver and `sim.dt`, which exposed that the
bridge still runs a different *control rate* (`decimation 1`) than training.
Closing that last consistency gap is its own change because — unlike the
solver — control rate is genuinely coupled to the Jetson publish rate and
needs the decoupling above rather than a shared-default flip.
