# Deploy env config: one key, one home

How a value reaches a node in the containerized deploy. This module covers the
`strafer_ros/deploy` compose lanes only; the Isaac Lab navigation env-cfg
composition is a different subject, in
[`env-composition-contract.md`](env-composition-contract.md).

> **Every key lives in exactly one home.** Either a generated `env_file`
> (canon-owned, reviewed, not host-overridable), or a service `environment:`
> entry as `${VAR:-default}` (host-overridable, and absent from every
> `env_file`). Never both.

`make env-check` fails when a key appears in both, so there is no precedence to
memorise: to find where a value comes from, ask which home it is in.

## The two homes

| | Node config | Host lever |
|---|---|---|
| Lives in | canonical `strafer_bringup/config/env_*.env` | a compose service's `environment:`, as `${VAR:-default}` |
| Reaches the container via | a generated `deploy/compose/*.env` loaded as `env_file:` | compose interpolation at parse time |
| Changed by | editing canon, then `make env-sync` | a shell export or `deploy/.env` |
| Reviewed | yes — canon is committed and diffed | no — per-host, per-run |
| Examples | `STRAFER_NAV_BACKEND`, `STRAFER_POLICY_VARIANT`, the freshness windows | `STRAFER_INFERENCE_MODEL_PATH`, `STRAFER_SLAM_TASK_ID`, `STRAFER_SLAM_SCENE_TOKEN`, `STRAFER_OBS_DUMP_PATH` |

Applying either needs the container **recreated** —
`up -d --force-recreate <service>`. `docker compose restart` reuses the old
container environment and applies nothing.

## Lanes

One canon file per lane; [`gen_env.py`](../../../source/strafer_ros/deploy/tests/gen_env.py)'s
`LANES` table generates the mirrors and is the list of them.

| Canon | Generated mirror | Lane |
|---|---|---|
| [`env_autonomy.env`](../../../source/strafer_ros/strafer_bringup/config/env_autonomy.env) | `compose/autonomy.env` | real-robot autonomy + inference |
| [`env_sim_in_the_loop.env`](../../../source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env) | `compose/sim.env` | standalone sim lane (`docker-compose.sim.yml`) |
| [`env_sim_bridge.env`](../../../source/strafer_ros/strafer_bringup/config/env_sim_bridge.env) | `compose/sim_bridge.env` | sim-bridge lane (`docker-compose.override.sim-bridge.yml`) |

A lane file **layers over** a base one rather than replacing it: an overlay's
`env_file` appends to the base list, and the last file wins for a duplicate key.
The sim-bridge lane therefore loads `[autonomy.env, sim_bridge.env]` and takes
the sim-rate values on top of the canonical ones. The same key may appear in
several mirrors — that is the layering mechanism, not a partition violation.

The mirrors are **generated**; hand-editing one fails `make env-check`, which
regenerates in-memory and byte-diffs.

## Why the rule is a partition and not a precedence

Three compose behaviours, measured:

| Behaviour | Consequence |
|---|---|
| An overlay's `env_file` **appends** to the base list | lane mirrors compose |
| For a duplicate key the **last file wins** | a lane mirror layers over canon |
| A service `environment:` key **beats** `env_file:` | a literal there shadows canon silently |

A bare `environment: - VAR` with no value was tested and **rejected**: it gives
host override but *deletes* the `env_file` value when the host does not set it,
so the key vanishes from the container entirely. No key can be both
`env_file`-defaulted and host-overridable, so each one picks a side.

## Values that are in neither home

| Kind | Where | Why |
|---|---|---|
| DDS: `RMW_IMPLEMENTATION`, `CYCLONEDDS_URI`, `ROS_DOMAIN_ID` | the `x-dds-env` anchor in `docker-compose{,.sim}.yml`, as literals | the canonical URI is self-locating `$(...)` shell that `env_file` cannot express; the container path lives in exactly one place per lane |
| Deploy-time host paths and tags: `STRAFER_MODELS_DIR`, `ZENOH_TAG`, `STRAFER_GIT_REVISION` | `deploy/.env` | never node config — bind sources, image tags, build stamps |
| Deploy-only service URLs: `VLM_URL`, `PLANNER_URL` | appended to a mirror by a declared per-lane overlay in `gen_env.py` | no canonical counterpart |

## The route the partition does not cover

A var a launch file reads from `os.environ` **inside** its container still needs
an explicit `environment:` mapping, because compose interpolation happens in a
different process than the launch file. `STRAFER_SLAM_TASK_ID` /
`STRAFER_SLAM_SCENE_TOKEN` are the current instances, mapped on the `slam` and
`strafer-sim` services. Without the mapping the knob is silently inert — the
same failure shape as a shadow, which is why it is checked in the same place.

## What enforces this

[`check_env_sync.py`](../../../source/strafer_ros/deploy/tests/check_env_sync.py),
run by `make env-check` inside both host test umbrellas. Its docstring carries
the measured behaviours above so the invariant travels with its rationale.

| Invariant | Fails when |
|---|---|
| mirrors == `gen_env(canon)` | a mirror is hand-edited, or canon changed without `make env-sync` |
| partition | a key appears in both a mirror and a service `environment:` |
| container-env passthrough | a declared container-read key is unmapped by its service |
| DDS URI consistency | the anchors disagree with each other or with the bind-mount target |
| deploy-only keys | a lane's declared generator-overlay key is missing from its mirror, or present in canon |
| REAL-HARDWARE GUARD | a sim-only freshness widening reaches the autonomy lane |
| sim-only mirror scope | an always-on lane loads `compose/sim_bridge.env` |

Untracked per-machine overlays (`docker-compose.override.*.yml` that are not
committed) are outside the check by construction — it cannot see them. One
applied later in the `-f` chain can hard-pin a lever and shadow the host env.

## Changing a value

1. Node config → edit the lane's canon file, `make env-sync`, force-recreate.
2. Per-run or per-host → export it, or set it in `deploy/.env`, force-recreate.
3. New key that needs both → it cannot have both; pick the side that matches
   who owns the value, and `make env-check` will hold you to it.

Verify what a service actually got, rather than what was intended:

```bash
docker compose <same -f / --profile flags> config      # rendered
docker exec <container> printenv <KEY>                 # actual
```

Operator-facing procedure lives in
[`deploy/README.md`](../../../source/strafer_ros/deploy/README.md); the
sim-bridge lane's runbook is
[`docs/sim_bridge_autonomy_cheatsheet.md`](../../sim_bridge_autonomy_cheatsheet.md).
