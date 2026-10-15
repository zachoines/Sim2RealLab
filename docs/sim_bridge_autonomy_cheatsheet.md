# Sim-bridge autonomy stack — spin-up + natural-language testing (strafer-nx)

This file holds commands that paste as they are, and descriptions or gotchas of at most three
lines, each linking its long form once. Mechanism, history and measurements live in those docs.

## Prereqs — on the DGX (`dgx-spark`, 192.168.50.196), 3 terminals

1. `cd ~/Workspace/Sim2RealLab && source env_setup.sh && make serve-vlm` — VLM on `:8100`
2. `cd ~/Workspace/Sim2RealLab && source env_setup.sh && make serve-planner` — planner on `:8200`
3. the **raw** sim-bridge (ProcRoom), at the script's default cadence flags:
   ```bash
   cd ~/Workspace/Sim2RealLab
   export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp PYTHONUNBUFFERED=1
   source env_setup.sh && source $CONDA_ROOT/etc/profile.d/conda.sh && conda activate env_isaaclab3
   tools/kit_boot_watchdog.sh --label sim-bridge -- \
       $ISAACLAB -p source/strafer_lab/scripts/run_sim_in_the_loop.py \
       --mode bridge --headless --enable_cameras \
       --task Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0
   ```
   [`kit_boot_watchdog.sh`](../tools/kit_boot_watchdog.sh) relaunches a Kit boot that stops early
   and stays stopped (Isaac Sim 6.0.1), printing one accounting line per attempt, and never touches
   a bridge once its boot is under way. `make sim-bridge` is wrapped too but passes no `--task`.

   Confirm the cadence print: `frame_skip=3 (derived, derived 3)` / `publish 30.00 Hz sim`.

   **Do not add `--decimation 4 --render-interval 4`:** depth starves to ~0.3 Hz sim; see
   [`enriched-lane-rig-stability`](tasks/active/reliability/enriched-lane-rig-stability.md) mode 4.

   **Match the task to the policy's training scene.** The depth-subgoal work was validated on
   `ProcRoom-Enriched-v0` (enclosed rooms, tall furniture, mid-room columns); plain `ProcRoom-v0`
   is a different distribution and its map is not interchangeable.

## On the NX (`strafer-nx`)

```bash
cd ~/Sim2RealLab/source/strafer_ros/deploy
export STRAFER_MODELS_DIR=/home/zachoines/strafer_models       # policy.onnx + .json live here

# 1) Tear down. `docker compose down` IGNORES profiled services unless their profile is active,
#    so mirror the up's -f + --profile flags, or inference/sim-perception/viewer keep running
#    (and donut_warmup, one-shot at sim-perception startup, never re-runs):
docker compose \
  -f docker-compose.yml \
  -f docker-compose.override.sim-bridge.yml \
  -f docker-compose.override.autonomy-local.yml \
  --profile policy --profile sim-bridge --profile viewer \
  down --remove-orphans
#    ...or remove every container in the 'strafer' project, whatever its profiles:
#    docker rm -f $(docker ps -aq --filter label=com.docker.compose.project=strafer)

# 1b) FRESH MAP on every sim restart. ProcRoom regenerates its layout, but rtabmap reloads its db
#     from the named volume, so a recreate alone comes back on the old map (WM=NNN), which
#     silently corrupts /plan. Bump the scene token (non-destructive; every earlier map is kept):
#       export STRAFER_SLAM_SCENE_TOKEN=sim$(date +%Y%m%d-%H%M)   # -> ~/.ros/rtabmap_<token>.db
#     The destructive alternative empties the volume:
#       docker volume rm strafer_strafer_ros_home

# 2) Spin up the full sim-bridge autonomy stack
#    (policy inference + SLAM + nav + sim support nodes + Foxglove + executor)
docker compose \
  -f docker-compose.yml \
  -f docker-compose.override.sim-bridge.yml \
  -f docker-compose.override.autonomy-local.yml \
  --profile policy --profile sim-bridge --profile viewer \
  up -d inference slam navigation sim-perception viewer autonomy

# 3) Wait ~60-90s: donut_warmup spins the robot to seed rtabmap's map. Sanity-check:
docker ps --format '{{.Names}}\t{{.Status}}' | grep strafer_
docker logs strafer_slam 2>&1 | grep -iE 'FATAL|rtabmap \(' | tail -3   # want "rtabmap (N): ..." iterating, no FATAL
docker logs strafer_slam 2>&1 | grep 'loadDataFromDb().*repair'   # want none: a hit = truncated db
#    On FATAL Memory.cpp:3852::addLink() (a db-load abort, not a /clock hitch), restart once:
#      docker restart strafer_slam        # wait ~30s for map->odom to re-establish
#    If it aborts again within ~1 s of "rtabmap subscribed to", the db is unloadable: bump the
#    token (1b; the old db is kept for forensics) and re-seed by re-running steps 1-3.
```

Details: [`enriched-lane-rig-stability`](tasks/active/reliability/enriched-lane-rig-stability.md)
mode 1 (the `addLink()` abort, the slam `stop_grace_period`); scene keying in the deploy
[README](../source/strafer_ros/deploy/README.md#slam-the-map-is-keyed-to-the-scene).

## Send a natural-language command

The object must be **visible from where the robot stands**: the go-to plan grounds against
`/d555/color`, turning through six headings if the first view misses it.
```bash
cd ~/Sim2RealLab
make submit-deploy CMD="go to the chair"
```
- **Watch it:** `docker logs -f strafer_autonomy` (VLM grounding, goal projection, navigation)
- **View it:** Foxglove Studio → `ws://localhost:8765`
  (tunnel from a workstation: `ssh -L 8765:localhost:8765 zachoines@192.168.50.161`;
  [viewer](../source/strafer_ros/deploy/README.md#viewer-foxglove))

## Good to know

- **Applying any config change: never `docker compose restart`.** It reuses the old container
  env and applies nothing. Recreate with `up -d --force-recreate <service>` and the stack's
  `-f`/`--profile` flags, then verify ([deploy README](../source/strafer_ros/deploy/README.md)):
  ```bash
  docker exec strafer_inference printenv STRAFER_INFERENCE_MODEL_PATH
  docker exec strafer_inference sh -c 'md5sum "$STRAFER_INFERENCE_MODEL_PATH"'
  ```
- **Pre-flight: confirm BOTH hosts are wired before spending rig time.** On WiFi the sim
  host's uplink cannot carry the camera streams and the policy gets no usable depth
  ([link budget](tasks/active/reliability/sim-bridge-link-transport-capacity.md)).
  ```bash
  ssh <sim-host> 'ip -brief addr | grep -v "^lo "'   # a wired NIC must be UP, not just present
  # then, once the stack is up, the number that actually matters:
  docker logs strafer_inference 2>&1 | grep 'cadence:' | tail -1
  # `depth rx` must climb ~1 per `ticks timer`; if it stays flat, depth is not arriving,
  # whatever the topic list says. Until a goal is active, inferences=0 and
  # skips watchdog≈ticks are normal: the watchdog counts the idle `goal` source.
  ```
- **Swapping the policy or the anchoring — use the tool:**
  ```bash
  source/strafer_ros/deploy/tools/configure_inference.sh <model> <mission|rolling>
  # model: a filename under /models, or an absolute container path.
  # Run with no arguments to list what is mounted.
  ```
  It force-recreates `inference` on the stack's own `-f` chain, then exits non-zero unless the
  container shows the requested model and anchoring and a loaded policy. Its
  [header](../source/strafer_ros/deploy/tools/configure_inference.sh) lists the checks.
- **Model swap by hand:** set `STRAFER_INFERENCE_MODEL_PATH` in the shell or `deploy/.env`,
  not a tracked compose file, and force-recreate `inference`. An untracked overlay later in the
  `-f` chain can shadow it ([README](../source/strafer_ros/deploy/README.md#one-key-one-home)).
- **Once the tool has run, recreating `inference` by hand silently reverts the anchoring.** The
  documented `-f` chain lacks the tool's `docker-compose.override.anchor.yml`, so the image's
  baked config returns, and the model comes from the shell or `deploy/.env` again. Re-run the tool.
- **Check what is running:** `docker logs strafer_inference 2>&1 | head -1` prints the image's
  `revision=<commit>`. `revision=unknown` means no build stamp, and a failed build silently keeps
  the old code under the tag ([deploy README](../source/strafer_ros/deploy/README.md)).
- **Where the policy config comes from:** canon `strafer_bringup/config/env_sim_bridge.env`, which
  reaches `inference` only; edit it, `make env-sync`, force-recreate. Not `compose/sim.env`: that is
  the standalone sim lane's. See [`deploy-env-config`](tasks/context/deploy-env-config.md#lanes).
- The executor's hybrid backend and [sim time](tasks/context/bridge-runtime-invariants.md) come only
  from the **untracked** `docker-compose.override.autonomy-local.yml` in the deploy dir; without sim
  time its policy-goal budget runs on wall time. Expect `hybrid_nav2_strafer`, `true`:
  ```bash
  docker exec strafer_autonomy printenv STRAFER_NAV_BACKEND STRAFER_USE_SIM_TIME
  ```
- **If the VLM cannot find the named object**, the go-to plan's
  [`scan_for_target`](../source/strafer_autonomy/README.md) checks six headings 60° apart, then
  fails with `target_not_found_after_scan`. Pick something clearly visible.
- **How a policy goal ends** (`error_code` from `make submit-deploy`): `navigation_failed`, the node
  aborted (first grep: [60 s sim bound](tasks/context/bridge-runtime-invariants.md), ~8 min wall);
  `navigation_timeout`, the executor quit first, on wall time or a frozen `/clock` (second grep).
  ```bash
  docker logs strafer_inference 2>&1 | grep 'mission timed out'
  docker logs strafer_autonomy 2>&1 | grep 'Nav2 wait: /clock stalled'
  ```
- **If the robot parks in the costmap inflation halo**, `GridBased` refuses its own pose as a
  planning start. Each lane then retries on `GridBasedRelaxed`, a degraded mode that logs.
- **Hybrid lane** ([guards](../source/strafer_ros/strafer_inference/config/subgoal_generator.yaml)):
  the subgoal generator switches planner itself. In `docker logs strafer_inference` the usual
  sequence is just these two:
  ```
  GridBased refused 2 consecutive replans (status 6); ... Switching to 'GridBasedRelaxed' ...
  Robot moved 0.25 m since the planner started refusing; returning to planner 'GridBased'.
  ```
- `Planner starvation:` appears only if `GridBasedRelaxed` fails too. It starts the hold, which
  republishes the last valid subgoal for `starvation_hold_s` of policy time so the policy can
  drive clear. `Starvation hold exhausted` means it stayed wedged: strafe it free (below).
- **Nav2 lane** ([design](tasks/completed/nav2-lane-inflated-start-recovery.md)):
  `start_cell_planner_selector` lets the BT's plan step retry on `GridBasedRelaxed` only while
  the robot's own cell is inscribed. `docker logs strafer_navigation` shows both:
  ```
  Robot's own global-costmap cell is 253 (>= inscribed 253); 'GridBased' cannot plan from it. Admitting 'GridBasedRelaxed' ...
  'GridBasedRelaxed' produced this goal's path: the nav2 lane is planning from a start the primary planner refuses.
  ```
- **Neither escape plans from deep in the halo:** `GridBasedRelaxed` clears only the robot's own
  cell and needs a free neighbour
  ([measured limit](tasks/completed/nav2-lane-inflated-start-recovery.md#found-while-verifying)).
- **Probe plannability** before trusting a measurement run. `SUCCEEDED` is fine; `ABORTED` /
  *"Starting point in lethal space"* is still wedged: free it with a manual holonomic strafe on
  `/cmd_vel` (Nav2's `/backup` refuses for the same reason).
  ```bash
  docker compose -f docker-compose.yml exec navigation bash -lc \
    'source /opt/ros/humble/setup.bash && source /ws/install/setup.bash; \
     ros2 action send_goal /compute_path_to_pose nav2_msgs/action/ComputePathToPose \
     "{goal: {header: {frame_id: map}, pose: {position: {x: 0.1, y: 0.1}, orientation: {w: 1.0}}}, planner_id: GridBased}"'
  ```
- **Direct (no-VLM) goal** for quick policy checks. Pick a **reachable** goal: run the
  plannability probe at its x/y first, since goals behind the robot or outside the seeded map
  return "no valid path".
  ```bash
  docker compose -f docker-compose.yml exec navigation bash -lc \
    'source /opt/ros/humble/setup.bash && source /ws/install/setup.bash; \
     ros2 action send_goal /strafer_inference/navigate_to_pose nav2_msgs/action/NavigateToPose \
     "{pose: {header: {frame_id: map}, pose: {position: {x: -1.0, y: 0.3}, orientation: {w: 1.0}}}}"'
  ```
