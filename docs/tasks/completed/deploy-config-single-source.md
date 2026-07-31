# Collapse the deploy config levels to one key, one home

**Status:** Shipped 2026-07-30 in `81fb15d` (Either; verified on Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/172

**Type:** task / tooling (deploy config)
**Owner:** Either (the change is compose + `gen_env.py`; the rig only verifies it)
**Priority:** P2 (the shadowing hazard it removes has already caused one near-miss)
**Estimate:** M (~1 day: a canon lane file, generator + checker changes, overlay rewrite, rig verify)
**Branch:** `task/deploy-config-single-source`

## Context bundle

- [context/conventions.md](../context/conventions.md)
- [context/deploy-env-config.md](../context/deploy-env-config.md) — the context
  module this work added; the standing statement of the rule it establishes
- [`source/strafer_ros/deploy/README.md`](../../../source/strafer_ros/deploy/README.md) — the "three levels of config" table this brief deletes
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

- [x] `compose/sim_bridge.env` is generated from a canonical `env_sim_bridge.env`;
      `make env-sync && make env-check` green (`make test-jetson` GREEN, ROS suite
      601 passed). Canon reaches the container end-to-end, confirmed live: the
      inference container logs `obs_timeout_s overridden to 1.0 via
      STRAFER_OBS_TIMEOUT_S`, `depth_timeout_s overridden to 2.0`,
      `variant=DEPTH_SUBGOAL`, and the hybrid backend starts the subgoal
      generator — all from the new canon file through its mirror.
- [x] The sim-bridge overlay contains **no** literal node-config `environment:`
      key — only `${VAR:-default}` entries for `STRAFER_INFERENCE_MODEL_PATH` and
      `STRAFER_OBS_DUMP_PATH`.
- [x] `check_env_sync` fails when a key appears in both a mirror and an overlay
      `environment:`, mutation-tested in **both** directions (shadow added to the
      overlay; host lever added back to canon), plus two adjacent invariants
      (sim-only mirror loaded by an always-on lane; a dropped container-env
      passthrough). Baseline green after each revert.
- [x] The model swap still works from the host with no canon edit: unset →
      `/models/policy.onnx` / `md5sum 709bd26e…`; v2 set + `up -d
      --force-recreate` → `strafer_depth_subgoal_v2_998.onnx` / `0272270e…`;
      back again → `709bd26e…`. Rendered config, `printenv`, and in-container
      `md5sum` agree at every step.
- [x] `deploy/README.md`'s three-level table is replaced by the one-line rule.

### The live gate — the other two session-critical levers

- [x] `STRAFER_OBS_DUMP_PATH` — unset: empty `printenv`, no `obs dump ENABLED`
      line, no file. Armed: the node logs `Diagnostic obs dump ENABLED →
      /obs_dumps/node_obs.jsonl` and the dump **grows** 2.96 MB / 161 lines →
      5.00 MB / 272 lines.
- [x] `STRAFER_SLAM_SCENE_TOKEN` — a fresh token is accepted and keys the db
      (`scene_key=envcheckA`, sidecar `"key": "envcheckA"`); a second fresh token
      gets its own db; a **stale-db** launch (token `envcheckB` against the db
      claimed by `envcheckA`) is **refused** with exit 1. Control: the same fixed
      db with its matching token starts normally, so the refusal keys on the
      mismatch rather than on `database_path:=`.
- [x] `docker compose config` renders on all eight overlay combinations, and the
      DDS anchor keys stay literal on every one while appearing in no mirror —
      the partition did not swallow them.

## Confirmed rather than assumed

The REAL-HARDWARE GUARD needed no change: it asserts the widening keys are absent
from the *autonomy* lane, and the new lane is a separate file. A new invariant
closes the gap that observation leaves — the sim-only mirror must be loaded by no
always-on lane, so the widenings cannot reach the real robot by someone adding an
`env_file` entry.

## Out of scope

- Moving `STRAFER_MODELS_DIR` or `ZENOH_TAG` into canon — they are deploy-time
  host paths and tags, never node config, and belong in `deploy/.env`.
- The canonical `env_*.env` -> mirror generator design itself, which works.

## Triggered by

PR review of `deploy-hardening` (2026-07-29): "can we consolidate the three
levels of config cleanly into one source of truth, with a principled design?"
The answer is yes for node config, once the partition rule replaces the
precedence rule.
