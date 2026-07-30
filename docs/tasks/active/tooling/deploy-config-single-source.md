# Collapse the deploy config levels to one key, one home

**Type:** task / tooling (deploy config)
**Owner:** Either (the change is compose + `gen_env.py`; the rig only verifies it)
**Priority:** P2 (the shadowing hazard it removes has already caused one near-miss)
**Estimate:** M (~1 day: a canon lane file, generator + checker changes, overlay rewrite, rig verify)
**Branch:** `task/deploy-config-single-source`

## Context bundle

- [context/conventions.md](../../context/conventions.md)
- [`source/strafer_ros/deploy/README.md`](../../../../source/strafer_ros/deploy/README.md) — the "three levels of config" table this brief deletes
- Sibling: [`deploy-hardening`](https://github.com/zachoines/Sim2RealLab/pull/169), which added the
  `${VAR:-default}` indirection as a stopgap and the container-passthrough invariant

## The problem

A value can reach a container by four routes, with a precedence rule an operator
has to know by heart:

| Route | Interpolates `${}`? | Guarded by `env-check`? |
|---|---|---|
| canon `env_*.env` -> generated `compose/*.env` -> `env_file:` | no | yes |
| overlay `environment:` | yes | **no** |
| host env / `deploy/.env` -> `${}` | it is the input | n/a |
| a var a launch file reads from `os.environ` **inside** the container | needs an explicit `environment:` mapping | only since the deploy PR |

`environment:` beating `env_file:` is the shadowing hazard: a canon edit plus
`make env-sync` changed nothing on the sim-bridge lane, which nearly published a
run under the wrong artifact label. The deploy PR made the overlay values
`${VAR:-default}` so the host can at least drive them, but the level count is
unchanged and the sim-bridge lane's values are still completely unguarded.

## The design

Two compose behaviours were verified empirically before writing this, because
the design turns on them:

- **An overlay's `env_file` appends to the base list**, it does not replace it —
  so `[compose/autonomy.env]` + `[compose/sim_bridge.env]` compose cleanly.
- **The last file wins** for a duplicate key, so a lane file layers over canon.

A third behaviour was tested and **rejected**: a bare `environment: - VAR`
(no value) gives host override but *deletes* the `env_file` value when the host
does not set it — the key vanishes from the container entirely. So there is no
way for one key to be both env_file-defaulted and host-overridable.

That constraint gives the rule:

> **Every key lives in exactly one home.** Either a generated `env_file`
> (canon-owned, reviewed, not host-overridable), or an overlay `environment:`
> entry as `${VAR:-default}` (host-overridable, and absent from every env_file).
> Never both.

Concretely:

1. Add canon `strafer_bringup/config/env_sim_bridge.env` carrying the lane
   provisioning the overlay hardcodes today — `STRAFER_NAV_BACKEND`,
   `STRAFER_POLICY_VARIANT`, `STRAFER_USE_SIM_TIME`, `STRAFER_OBS_TIMEOUT_S`,
   `STRAFER_DEPTH_TIMEOUT_S` — and register it in `gen_env.py`'s `LANES` so it
   generates `compose/sim_bridge.env` and is byte-diffed like the others.
2. The sim-bridge overlay drops those five `environment:` keys and gains
   `env_file: [compose/sim_bridge.env]`.
3. What stays in `environment:` is only what genuinely needs a host lever, each
   `${VAR:-default}` and each absent from every env_file:
   `STRAFER_INFERENCE_MODEL_PATH` (per-run artifact swap),
   `STRAFER_SLAM_TASK_ID` / `STRAFER_SLAM_SCENE_TOKEN` (per-sim-run keying),
   `STRAFER_OBS_DUMP_PATH` (diagnostic, normally absent).
4. `check_env_sync` gains the partition invariant: **no key may appear in both a
   generated mirror and an overlay `environment:`.** That is what makes the
   shadowing structurally impossible rather than merely discouraged.

The REAL-HARDWARE GUARD needs no change — it asserts the widening keys are
absent from the *autonomy* lane, and the new lane is a separate file. Confirm
that in the implementation rather than assuming it.

## What this does not solve

The fourth route stays: a var a launch file reads from `os.environ` inside the
container still needs an explicit `environment:` mapping, because no `env_file`
entry can satisfy a read that happens in a different process namespace than
compose interpolation. The deploy PR's passthrough invariant covers it; this
brief should fold that check into the same partition test so there is one place
that answers "how does a value reach a node".

## Acceptance

- [ ] `compose/sim_bridge.env` is generated from a canonical
      `env_sim_bridge.env`; `make env-sync && make env-check` green, and editing
      canon + syncing demonstrably changes what the container loads.
- [ ] The sim-bridge overlay contains **no** literal node-config `environment:`
      key — only `${VAR:-default}` entries for the host levers listed above.
- [ ] `check_env_sync` fails when a key appears in both a mirror and an overlay
      `environment:`, mutation-tested by introducing one.
- [ ] The model swap still works from the host with no canon edit, verified the
      way the deploy PR verified it (`printenv` + `md5sum` of the loaded
      artifact, before and after `up -d --force-recreate`).
- [ ] `deploy/README.md`'s three-level table is replaced by the one-line rule.

## Out of scope

- Moving `STRAFER_MODELS_DIR` or `ZENOH_TAG` into canon — they are deploy-time
  host paths and tags, never node config, and belong in `deploy/.env`.
- The canonical `env_*.env` -> mirror generator design itself, which works.

## Triggered by

PR review of `deploy-hardening` (2026-07-29): "can we consolidate the three
levels of config cleanly into one source of truth, with a principled design?"
The answer is yes for node config, once the partition rule replaces the
precedence rule.
