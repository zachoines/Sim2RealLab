# Give same-pose sim probes a captured scene instead of a seed

**Type:** tooling (measurement reproducibility)
**Owner:** DGX (`strafer_lab` probe tooling)
**Priority:** P2 — no rig time is blocked on it, but every future attribution
that needs "the same pose in sim" is, and the one that exists already lost its
pose once. It is cheap to fix and it silently invalidates measurements when it
is not fixed.
**Estimate:** S
**Branch:** `task/pose-stable-sim-probe`

## Story

As a **probe that has to put the robot back where the robot was**, I need **the
room to be the same room**, so that **a measurement taken at that pose still
means what it meant when the pose was chosen.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context — what happened

The same-pose probe behind
[`goal-a-attribution-2026-08-22`](../../../measurements/goal-a-attribution-2026-08-22/README.md)
reproduces a robot-side capture pose in pure sim. It does it in two steps: read
whatever pose the environment spawned the robot at, then teleport to that pose
plus the capture's map-frame offset. Both steps are sound. The assumption
underneath them is that the task ID plus seed 42 reproduces the room, and that
assumption is what failed.

Re-run unmodified on the canonical pair after the Isaac Lab pair flip
([`isaac-lab-upgrade`](../../completed/isaac-lab-upgrade.md)), the probe launched
and completed, every API it touches survived, and its mission-goal bearing
reproduced to 0.03°. But the spawn had moved, so the offset landed the robot
about 2 m away, facing open space. Its clean near-field share was **0.0000**
against the original run's **0.3786** — an empty denominator for the very metric
the probe exists to produce, with nothing in the output saying so. The
convention-fix record had to measure on a fixed captured frame instead
([`depth-convention-fix-2026-09-13`](../../../measurements/depth-convention-fix-2026-09-13/README.md) §5).

Seed determinism across a renderer or Isaac Lab version change is not a property
the repo can hold, and the upgrade record already documents depth array hashes
moving and seeded pose traces diverging chaotically. So the probe needs its scene
pinned by content, not by construction.

## Acceptance criteria

- [ ] A probe that needs a specific pose binds a **captured scene USD**, not a
      task ID plus a seed, and the repo has one helper for doing that rather than
      each probe rolling its own.
- [ ] The probe fails loudly when the scene it loads is not the scene the pose
      was chosen in — a recorded scene digest that is checked, not assumed.
- [ ] A probe that reaches its pose reports enough to tell that it did: the
      achieved pose against the intended one, and at least one content statistic
      of the frame (the near-field share is the one that caught this) so an empty
      denominator is visible in the output rather than inferred later.
- [ ] The 2026-08-22 probe's own pose is either recovered under this mechanism or
      recorded as unrecoverable, so the next attribution knows which.
- [ ] Scene capture and bind are covered by a test that does not need a Kit boot
      for the bind half.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Out of scope

- Reproducing the 2026-08-22 room on the retired pin. That is a separate
  question and the pin is retired.
- The attribution itself, and anything about the depth convention — settled in
  [`depth-nearfield-convention-mismatch`](../../completed/depth-nearfield-convention-mismatch.md).
- Scene-corpus generation. This is about binding one captured scene, not about
  making more of them.
