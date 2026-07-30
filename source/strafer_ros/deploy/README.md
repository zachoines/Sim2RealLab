# strafer_ros — containerized deployment

DDS-native containerization of the Strafer stack for the Jetson Orin NX
(JetPack 6.2 / L4T r36.4.3). Two images decomposed over CycloneDDS.

## Images (decomposition by rebuild-frequency + GPU coupling)

| Image | Runs | Base | GPU |
|---|---|---|---|
| `strafer-cpu` | base · perception · slam · navigation · autonomy (separate containers, same image) | `ros:humble-ros-base` | no |
| `strafer-gpu` | inference (policy) only | `nvcr.io/nvidia/l4t-jetpack:r36.4.0` | yes |

DDS: **CycloneDDS**, `ROS_DOMAIN_ID=42`, the canonical tuned `cyclonedds.xml`
(bind-mounted straight from `strafer_bringup/config`, not copied). Every service
is `network_mode: host` (Cyclone uses UDP loopback between same-host containers —
no shared `/dev/shm`).

## Layout
```
deploy/
├── docker/{Dockerfile.cpu, Dockerfile.gpu}   # entrypoint inlined in each
├── docker-compose.sim.yml     # sim-in-the-loop (nav2 backend, no GPU/hardware)
├── docker-compose.yml         # full deploy (5 services + policy/remote/viewer/sim-bridge profiles)
├── docker-compose.dev.yml     # live ./source bind-mount overlay
├── docker-compose.override.sim-bridge.yml    # sim-bridge provisioning (explicit -f only)
├── .env.example               # per-host compose interpolation — `cp .env.example .env`
├── models/                    # default /models bind source (ships empty; artifacts not committed)
├── compose/{sim.env, autonomy.env}           # env_file mirrors — GENERATED from canon; do not hand-edit
├── host-setup/install-host-prereqs.sh        # rmem sysctl, nvidia runtime, netfilter, compose, udev
└── tests/{gen_env.py, check_env_sync.py}     # gen_env writes the mirrors from canon; check_env_sync (make env-check) fails on drift
```

The sim-bridge lane's operator runbook is
[`docs/sim_bridge_autonomy_cheatsheet.md`](../../../docs/sim_bridge_autonomy_cheatsheet.md).

## Applying a config change — `restart` does NOT do it

> **`docker compose restart` reuses the OLD container environment.** It restarts
> the process inside the existing container; it does not re-render the service
> config. Measured: an overlay edited to roll a policy back to v1, then
> `restart inference`, silently **reloaded v2** while the config on disk said v1.
> Nothing warned.

Any change to an env var, an `env_file`, a compose file, `deploy/.env`, or a
canonical `env_*.env` + `make env-sync` needs the container **recreated**:

```bash
docker compose <the same -f / --profile flags you brought it up with> \
  up -d --force-recreate <service>
```

Then verify what the container actually got, rather than what you intended:

```bash
docker compose <same flags> config | sed -n '/inference:/,/^  [a-z]/p'   # rendered config
docker exec strafer_inference printenv STRAFER_INFERENCE_MODEL_PATH      # what it really has
```

`restart` is still the right tool for one thing: reloading **bind-mounted source**
under `docker-compose.dev.yml`, where the container env is unchanged and only the
files behind it moved.

### Three levels of "config", and which one wins

| Level | Where | Beats |
|---|---|---|
| host env / `deploy/.env` | `${VAR:-default}` expansions, at compose **parse** time | fills in the overlay defaults |
| service `environment:` | compose files / overlays | **beats `env_file:`** |
| `env_file:` | `compose/*.env`, generated from canon | lowest |

That middle row is the subtle one: a hard-pinned `environment:` key silently
shadows the generated mirror, so editing canonical `env_autonomy.env` +
`make env-sync` changes **nothing** on a lane that applies such an overlay — and
a policy swap done that way runs the old artifact under the new label. Every
canon-backed key in the deploy overlays is therefore `${VAR:-<lane default>}`, so
the lane default still stands on its own while the host (or canon, if you source
it) can drive it. Set the model with `STRAFER_INFERENCE_MODEL_PATH` in your shell
or `deploy/.env` — **not** by editing a tracked compose file.

The DDS keys (`RMW_IMPLEMENTATION` / `CYCLONEDDS_URI` / `ROS_DOMAIN_ID`) are
deliberately NOT indirected: they come from the `x-dds-env` anchor by design, and
`make env-check` pins them to literals.

## Image provenance — is the running stack the code you think it is?

Both images stamp the commit they were built from into
`org.opencontainers.image.revision`, and every container announces it on the
first line of its log:

```bash
make images                                        # stamps `git rev-parse` (+ `-dirty`)
docker logs strafer_inference 2>&1 | head -1       # [strafer] image=strafer-gpu revision=69014c6f1a2b
docker inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' strafer-gpu:humble
```

`revision=unknown` means the image was built by a bare `docker compose build`
with no stamp — treat its contents as unverified.

> **Stale-build hazards — two of them, both silent.**
>
> 1. **A bare `docker compose build` skips `strafer-gpu` entirely.** Compose does
>    not build services whose profile is inactive, and `inference` lives in the
>    `policy` profile — so the build exits **0** having rebuilt only the CPU
>    image. Measured: `strafer-gpu:humble` kept an empty revision label
>    straight through a "successful" build, which is the most likely way a
>    deployed GPU image comes to lag `main` across a behaviour-significant fix.
>    Use `make images` (it passes `--profile policy`) or add the flag yourself.
> 2. **A *failed* build commits no layer**, so the tag stays resolvable and
>    byte-identical to the previous image — the build errors out but the stack
>    still comes up, on the old code.
>
> Both fail the same way: a build you believe succeeded and a tag that resolves
> are two different claims. **Check the revision labels after every rebuild** —
> `make images` prints them for you.

### Config — single source of truth
Runtime env for both lanes lives in the canonical `strafer_bringup/config/env_*.env`
(shell, hand-edited, with the rationale comments). The compose `env_file` mirrors
under `compose/` are **generated** from them by `tests/gen_env.py` — edit canon,
then `make env-sync`. DDS vars (RMW / CYCLONEDDS_URI / ROS_DOMAIN_ID) come from the
compose `x-dds-env` anchor, not the mirror, so the self-locating `$(...)` URI never
enters a container. Deploy-only keys with no canonical home (VLM_URL / PLANNER_URL)
are a declared overlay in the generator. `make env-check` (run inside `make test`)
regenerates + byte-diffs and fails on any drift — including the CYCLONEDDS_URI the
old overlap-diff skipped.

## Build / deploy / policy / remote
```bash
cd Sim2RealLab/source/strafer_ros/deploy
sudo bash host-setup/install-host-prereqs.sh    # once per host
make images                                     # builds BOTH images + stamps the build commit
docker compose up                               # base perception slam navigation autonomy  (base/perception need hardware)
docker compose --profile policy up              # + GPU inference  (see "Policy backend")
docker compose --profile remote up              # + Zenoh bridge   (see "Zenoh / remote")

# live-edit iteration — bind-mounted SOURCE only, where the container env is
# unchanged. A config/env change still needs `up -d --force-recreate` (see
# "Applying a config change" below):
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Viewer (Foxglove)
`foxglove_bridge` on `:8765` for live inspection of the domain-42 graph (TF,
costmaps, depth, `cmd_vel`, subgoals, the policy's `navigate_to_pose` feedback).
Opt-in and agnostic to which stack is up — use it for the real-robot deploy or
the sim-bridge e2e (it launches no robot nodes, just attaches to DDS):
```bash
docker compose --profile viewer up viewer      # or: make viewer
```
Connect Foxglove Studio to `ws://<robot-ip>:8765` (the bridge binds `0.0.0.0`),
or tunnel it: `ssh -L 8765:localhost:8765 <user>@<robot-ip>`. Bare-metal:
`ros2 launch strafer_bringup viewer.launch.py` (args: `viewer_port`, `use_sim_time`).

### Policy backend (GPU inference)
The `inference` service runs `inference_policy.launch.py`, which reproduces the
canonical backend coupling and **fails loud** rather than silently degrading:
1. set `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` (the depth policy's backend) or
   `strafer_direct` in canonical `strafer_bringup/config/env_autonomy.env`, then `make env-sync`;
2. put the exported policy under `deploy/models/` (or point `STRAFER_MODELS_DIR`
   elsewhere — see `deploy/models/README.md`) and set
   `STRAFER_INFERENCE_MODEL_PATH=/models/<model>.onnx`;
3. `docker compose --profile policy up`.

An empty/missing model under a policy backend — or a non-policy backend — aborts
the inference container at launch (no silent nav2 fallback). `hybrid_nav2_strafer`
also auto-starts the rolling-subgoal generator.

> **On a lane that applies an overlay** (e.g. the sim-bridge lane), step 1's
> canon edit only reaches the container because the overlay values are
> `${VAR:-default}` — see "Three levels of config" below. Confirm with
> `docker compose ... config` before trusting a swap, and recreate rather than
> restart.

### Diagnostics: train↔deploy obs parity

The inference node can dump each assembled observation as JSONL for
`scripts/obs_parity.py` (contract: `strafer_inference/scripts/PARITY_SCHEMA.md`).
It reads `obs_dump_path` **once at init**, so it can only be armed at launch —
never `ros2 param set`. On the compose lane:

```bash
# in docker-compose.override.sim-bridge.yml, uncomment BOTH the
# STRAFER_OBS_DUMP_PATH line and the ./obs_dumps bind mount, then:
docker compose <same flags> up -d --force-recreate inference
docker logs strafer_inference 2>&1 | grep 'obs dump ENABLED'
```

> **Not for normal missions.** A `DEPTH`/`DEPTH_SUBGOAL` variant writes a full
> depth vector per tick at 30 Hz (~2–3 MB/s). Unset it and force-recreate when
> the capture is done. Empty/unset is the default and costs nothing per tick.

There is no separate dump-variant knob: the node stamps each record with its
loaded `policy_variant`, so a dump cannot disagree with the artifact running.

### SLAM: the map is keyed to the scene

RTAB-Map persists its database and **reloads it on start**. On the sim lane the
scene is procedurally regenerated, so a restarted sim is a new layout — and
reloading the previous run's map silently corrupts `/rtabmap/map` and every Nav2
`/plan` measured against it. Set a token per sim run:

```bash
STRAFER_SLAM_SCENE_TOKEN=run4        # -> ~/.ros/rtabmap_run4.db
```

Whatever path is used, a `<db>.scene.json` sidecar records the key it belongs to,
and a launch whose key **disagrees** aborts with the ways out rather than loading
a foreign map. An empty key reproduces the historical behaviour exactly — the
real robot maps one persistent scene and should reload `~/.ros/rtabmap.db`.
`database_path:=<path>` still overrides everything, and `rtabmap_args:=-d` wipes
and remaps.

### Zenoh / remote workstation
`--profile remote` starts a Zenoh bridge exposing the ROS graph on
`tcp/0.0.0.0:7447`; the workstation runs a **version-matched** bridge:
`zenoh-bridge-ros2dds -d 42 -e tcp/<robot-ip>:7447` (set `ZENOH_TAG` to match).

> **Security:** the bridge is **unauthenticated** and exposes the whole ROS graph
> on `0.0.0.0:7447`. It uses `restart: "no"` deliberately so it does not
> auto-resurrect that exposure across reboots — start it only when you need
> remote access. **Zenoh is a remote-workstation convenience, NOT the
> DGX↔robot sim transport**: the A/B experiments run wired-LAN CycloneDDS; nobody
> bridges 921 KB depth over WiFi TCP mid-experiment.

## Sim-in-the-loop (no hardware, no GPU)
```bash
docker compose -f docker-compose.sim.yml up
```
Runs `bringup_sim_in_the_loop.launch.py` (description → timestamp_fixer → SLAM →
Nav2 → goal_projection → executor → foxglove) in one `strafer-cpu` container,
consuming the Isaac Sim bridge over CycloneDDS. Backend is `nav2`, so no GPU.
The sim lane's `sim.env` names the GPU image only as a pointer — there is **no
sim GPU service** (a DEPTH policy on the CPU image would run ~84 ms vs the 33 ms
budget); the DGX↔robot A/B sessions don't need it.

## Notes
- `nvidia-ctk --set-as-default` changes the host's **default** docker runtime
  (fine on a dedicated robot host; the host-setup script only does it when the
  runtime isn't already nvidia, and only then restarts docker).
- `restart: unless-stopped` on the robot stack auto-resurrects crashed nodes on
  reboot — the node watchdogs mitigate, but be aware of the semantics.
- **New-Jetson provisioning:** flash JP6.2 → `sudo apt install docker.io` →
  `sudo bash host-setup/install-host-prereqs.sh` → `docker compose build` → mount
  the model. Everything except the host kernel/udev/sysctl and the per-device
  TensorRT engine cache lives in images.
