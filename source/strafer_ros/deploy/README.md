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
├── docker-compose.yml         # full deploy (5 services + policy/remote profiles)
├── docker-compose.dev.yml     # live ./source bind-mount overlay
├── compose/{sim.env, autonomy.env}           # env_file mirrors — GENERATED from canon; do not hand-edit
├── host-setup/install-host-prereqs.sh        # rmem sysctl, nvidia runtime, netfilter, compose, udev
└── tests/{gen_env.py, check_env_sync.py}     # gen_env writes the mirrors from canon; check_env_sync (make env-check) fails on drift
```

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
docker compose build                            # builds strafer-cpu AND strafer-gpu
docker compose up                               # base perception slam navigation autonomy  (base/perception need hardware)
docker compose --profile policy up              # + GPU inference  (see "Policy backend")
docker compose --profile remote up              # + Zenoh bridge   (see "Zenoh / remote")

# live-edit iteration (python/launch/yaml reflect on `restart <svc>`):
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Policy backend (GPU inference)
The `inference` service runs `inference_policy.launch.py`, which reproduces the
canonical backend coupling and **fails loud** rather than silently degrading:
1. set `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` (the depth policy's backend) or
   `strafer_direct` in canonical `strafer_bringup/config/env_autonomy.env`, then `make env-sync`;
2. put the exported policy under `./models` (or set `STRAFER_MODELS_DIR`) and set
   `STRAFER_INFERENCE_MODEL_PATH=/models/<model>.onnx`;
3. `docker compose --profile policy up`.

An empty/missing model under a policy backend — or a non-policy backend — aborts
the inference container at launch (no silent nav2 fallback). `hybrid_nav2_strafer`
also auto-starts the rolling-subgoal generator.

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
