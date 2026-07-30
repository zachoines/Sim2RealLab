# Deploy-lane hardening — the six defects the v2 bridge validation found

**Status:** Shipped 2026-07-30 in `85cc34d` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/169
**Follow-ups:** [`nav2-lane-inflated-start-recovery`](../active/reliability/nav2-lane-inflated-start-recovery.md) — the nav2 lane still cannot escape the wedge; [`deploy-config-single-source`](../active/tooling/deploy-config-single-source.md) — partition rule for deploy config; [`ros-node-clock-deadlines`](../active/tooling/ros-node-clock-deadlines.md) — tick counters to node-clock deadlines.

**Type:** task / reliability (deploy lane)
**Owner:** Jetson
**Priority:** P1 (two defects nearly produced a silent false pass; one was a reproducible deadlock)
**Estimate:** L
**Branch:** `task/deploy-hardening`

> Filed retroactively. The dispatch was issued out-of-band
> (`DEPLOY_HARDENING_PR_PROMPT.md`, coordinator-side) rather than as an in-repo
> brief, so this records what was asked and what was held to, per the "the brief
> records what we set out to do" half of the lifecycle contract.

## Context bundle

- [context/conventions.md](../context/conventions.md)
- `V2_NX_VALIDATION_REPORT.md` §7 (RC-1…RC-7) and §13 (WP-3…WP-6) — the
  measurement session that found these. Lives coordinator-side, not in-repo.

## The problem (measured on the NX, 2026-07-28/29)

Six deploy defects, of which two nearly caused a silent false pass:

| | Defect |
|---|---|
| RC-1 | A service `environment:` key beats `env_file:`, so the canon-env model swap was inert on the sim-bridge lane — the run would have been v1 labelled as v2 |
| RC-2 | `STRAFER_MODELS_DIR` unset resolved `/models` to a nonexistent bind source |
| RC-3 | The deployed image lagged `main` by 9 days across a behaviour-significant fix, and nothing said so |
| RC-4 | rtabmap silently reloads a database recorded under a different procedurally-regenerated scene, corrupting `/plan` |
| RC-5 | The GPU image was not reproducible from source (the jp6 index stopped serving a pure-python dep) |
| RC-6 | Parked-in-inflation deadlock: a legitimate park near an obstacle makes the planner refuse the robot's own pose, starving the rolling subgoal until the policy zero-twists — and because the robot cannot move, the pose never changes |
| — | Separately, the obs-parity harness could not be armed on the compose lane at all, which blocked the epic's gating instrument |

## Acceptance

- [x] Obs-dump plumbed to the compose lane; empty/unset stays the default
- [x] The model swap is real on every lane, with an audit of every other
      `environment:` key
- [x] `restart` replaced by `up -d --force-recreate` wherever it would apply a
      config change, and a real default for the models directory
- [x] RC-6 fixed with the minimal evidenced option, favouring fail-loud, with
      the choice and its reasoning stated in the PR
- [x] Image provenance: build-commit label plus one startup log line, and the
      stale-build hazard documented
- [x] The SLAM database keyed to its scene, fail-loud on mismatch
- [x] Gates: both images build; `docker compose config` renders on every overlay
      combination; `make env-sync && make env-check` green; **RC-6 demonstrated
      against the reproduction on the rig**; obs-dump smoke armed and disabled;
      Jetson suites green; no control-path change outside the RC-6 fix

## What changed relative to the dispatch

Two premises in the dispatch were falsified and are recorded as corrections:

1. **"Nav2 GridBased/costmap options first — config before code" is not
   reachable.** Humble's `SmacPlanner2D` exposes no parameter that relaxes its
   start-cell check, and the `>=INSCRIBED` band is sized by the footprint's
   inscribed radius rather than `inflation_radius`. The same outcome was reached
   config-first by registering a *second* planner (`GridBasedRelaxed`, a
   `NavfnPlanner` whose `makePlan()` clears the robot's own cell) rather than
   relaxing the first.
2. **`deploy/.env` could not be committed** — `.env` is gitignored repo-wide with
   a stated "`.env.example` IS committed" convention, so the default was made to
   resolve to a real directory instead.

Three defects were found that the dispatch did not know about:

- `docker compose build` **skips services in an inactive profile**, so a bare
  build never rebuilt the GPU image and still exited 0 — very likely RC-3's real
  mechanism.
- The SLAM scene token never reached the container (RC-1's own shape, shipped by
  the change that fixes RC-1) — found only by running the stack against a live
  bridge.
- The subgoal generator was loading `inference.yaml`, because a
  `LaunchConfiguration` set by one include is visible to the next. Predates this
  work; invisible until a key existed whose YAML value differed from its code
  default.

**The standing lesson:** both of the last two passed `docker compose config`,
the full unit suites, and review. Static verification of a deploy change is not
evidence that a knob works.

## Out of scope

- The obs-parity capture itself (the harness is now armable; the workstation
  `--gym-dump` counterpart and its synchronisation remain open).
- The nav2 backend lane's half of the RC-6 wedge — see the follow-up brief.
- Any policy-side constant: arrival radius and dwell were explicitly untouched.

## Triggered by

The v2 policy NX bridge validation session (2026-07-28/29), whose pre-flight
found RC-1…RC-7 before any mission ran.
