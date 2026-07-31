# Sim-bridge autonomy stack — spin-up + natural-language testing (strafer-nx)

## Prereqs — on the DGX (`dgx-spark`, 192.168.50.196), 3 terminals

1. `serve-vlm`    — VLM on `:8100`
2. `serve-planner` — planner on `:8200`
3. the **raw** sim-bridge (ProcRoom, **skip 0** — `make sim-bridge` uses the wrong flags):
   ```bash
   cd ~/Workspace/Sim2RealLab
   export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
   source env_setup.sh && source $CONDA_ROOT/etc/profile.d/conda.sh && conda activate env_isaaclab3
   $ISAACLAB -p source/strafer_lab/scripts/run_sim_in_the_loop.py \
       --mode bridge --headless --enable_cameras \
       --task Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0 --decimation 4 --render-interval 4
   ```
   Confirm the cadence print: `frame_skip=0 (derived, derived 0)` / `publish 30.00 Hz sim`.

   **Match the task to the policy's training scene.** The depth-subgoal work was
   validated on `ProcRoom-Enriched-v0` (enclosed rooms, tall furniture, mid-room
   columns); plain `ProcRoom-v0` is a different distribution and its map is not
   interchangeable.

## On the NX (`strafer-nx`)

```bash
cd ~/Sim2RealLab/source/strafer_ros/deploy
export STRAFER_MODELS_DIR=/home/zachoines/strafer_models       # policy.onnx + .json live here

# 1) Tear down. IMPORTANT: `docker compose down` IGNORES profiled services
#    unless their profile is active -- so mirror the up's -f + --profile flags,
#    or inference/sim-perception/viewer keep running (and donut_warmup, which is
#    one-shot at sim-perception startup, never re-runs):
docker compose \
  -f docker-compose.yml \
  -f docker-compose.override.sim-bridge.yml \
  -f docker-compose.override.autonomy-local.yml \
  --profile policy --profile sim-bridge --profile viewer \
  down --remove-orphans
#    ...or nuke the whole 'strafer' project regardless of profiles (bulletproof):
#    docker rm -f $(docker ps -aq --filter label=com.docker.compose.project=strafer)

# 1b) FRESH MAP. rtabmap persists its db in a named volume and RELOADS it, so a
#     recreate alone comes back on the old map (WM=NNN). Prefer bumping the
#     scene token -- non-destructive, keeps every previous run's map:
#       export STRAFER_SLAM_SCENE_TOKEN=run5     # -> ~/.ros/rtabmap_run5.db
#     The destructive option still works if you want the volume empty:
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
#    If rtabmap died on a /clock hitch (FATAL Memory.cpp addLink), just restart it:
#      docker restart strafer_slam        # wait ~30s for map->odom to re-establish
```

## Send a natural-language command

Object must be **visible in the current camera view** (the VLM grounds against `/d555/color`):
```bash
cd ~/Sim2RealLab
make submit-deploy CMD="go to the chair"
```
- **Watch it:** `docker logs -f strafer_autonomy` (VLM grounding, goal projection, dispatch)
- **View it:** Foxglove Studio → `ws://localhost:8765`
  (tunnel from your Mac: `ssh -L 8765:localhost:8765 zachoines@192.168.50.24`)

## Good to know

- **Applying ANY config change: never `docker compose restart`.** It reuses the
  old container environment and applies nothing — a policy rollback to v1 will
  silently keep running v2 while the config on disk says v1. Use
  `up -d --force-recreate <service>` with the same `-f`/`--profile` flags, then
  verify: `docker exec strafer_inference printenv STRAFER_INFERENCE_MODEL_PATH`
  and `md5sum` the artifact. (`docker restart strafer_slam` below is fine —
  that's crash recovery, not a config change.)
- **Swapping the policy:** set `STRAFER_INFERENCE_MODEL_PATH` in your shell or
  `deploy/.env` (`cp .env.example .env`), then force-recreate `inference`. The
  overlay's value is `${STRAFER_INFERENCE_MODEL_PATH:-/models/policy.onnx}`, so
  the host wins and the v1 artifact stays the default. Do **not** edit a tracked
  compose file for this, and note a local `docker-compose.override.*.yml` applied
  later in the `-f` chain hard-pins the value and will shadow the env again.
- **Bump the SLAM scene token on every sim restart:** `STRAFER_SLAM_SCENE_TOKEN=run4`
  → `~/.ros/rtabmap_run4.db`. ProcRoom is procedurally regenerated, so a
  restarted sim is a new layout; reloading the old map silently corrupts `/plan`.
  A db recorded under a different token is now refused at launch rather than
  loaded.
- **Check what you're actually running:** `docker logs strafer_inference 2>&1 | head -1`
  prints `[strafer] image=strafer-gpu revision=<commit>`. `revision=unknown`
  means the image carries no build stamp; a *failed* `docker compose build`
  leaves the old tag resolvable, so a stack can silently run stale code.
- **Where the policy config comes from:** this lane's node config is canon —
  `strafer_bringup/config/env_sim_bridge.env` (`STRAFER_NAV_BACKEND=hybrid_nav2_strafer`,
  `STRAFER_POLICY_VARIANT=DEPTH_SUBGOAL`, `STRAFER_USE_SIM_TIME=true`, and the two
  sim-rate timeout widenings) — generated into `deploy/compose/sim_bridge.env`, which
  `docker-compose.override.sim-bridge.yml` loads as a second `env_file` on top of
  `autonomy.env`. Edit canon, `make env-sync`, force-recreate. **Do not** look for these
  in `deploy/compose/sim.env` — that file (nav2 / `DEPTH`) belongs to the *standalone*
  `docker-compose.sim.yml` lane, which this stack does not use. The overlay's
  `environment:` carries only the host levers (`STRAFER_INFERENCE_MODEL_PATH`,
  `STRAFER_OBS_DUMP_PATH`). Every key has exactly one home and `make env-check` fails
  on a key in both, so nothing here shadows the canon path — see
  [`context/deploy-env-config.md`](tasks/context/deploy-env-config.md).
- The executor is set to the **hybrid (policy) backend** via `docker-compose.override.autonomy-local.yml`, so semantic goals drive the DEPTH_SUBGOAL policy (not plain nav2). That override is **untracked** (host-specific URLs) — it lives in the deploy dir.
- If the VLM can't find the named object, the mission fails at grounding ("target not found"). Pick something clearly in frame.
- **Known open items:** the policy parks near the goal but doesn't trip `NavigateToPose`'s success radius, and the mission-runner's hybrid-nav step has a **7 s** timeout that's short for the sim's RTF — so a mission may report a nav timeout even when the robot arrived. Drive/loop are correct; it's a completion-signal/tuning gap.
- **If the robot parks within ~0.4 m of an obstacle** it lands in the costmap
  inflation halo, where `GridBased` refuses its own pose as a planning start.
  The subgoal generator now escapes that itself — it retries on
  `GridBasedRelaxed` and, failing that, republishes the last subgoal for a
  bounded window — but both are DEGRADED modes and log at WARN/ERROR. In
  `docker logs strafer_inference` the usual sequence is just the first two:
  ```
  GridBased refused 2 consecutive replans (status 6); ... Switching to 'GridBasedRelaxed' ...
  Robot moved 0.25 m since the planner started refusing; returning to planner 'GridBased'.
  ```
  `Planner starvation:` only appears if the fallback planner ALSO fails, and
  `Starvation hold exhausted` means it stayed wedged through the whole hold —
  that one needs manual intervention. Probe plannability before trusting a
  measurement run:
  ```bash
  docker compose -f docker-compose.yml exec navigation bash -lc \
    'source /opt/ros/humble/setup.bash && source /ws/install/setup.bash; \
     ros2 action send_goal /compute_path_to_pose nav2_msgs/action/ComputePathToPose \
     "{goal: {header: {frame_id: map}, pose: {position: {x: 0.1, y: 0.1}, orientation: {w: 1.0}}}, planner_id: GridBased}"'
  ```
  On the **nav2** lane the escape is in the navigate-to-pose BT instead: the
  `start_cell_planner_selector` node reads the robot's own global-costmap cell
  and only then does the BT's plan step retry on `GridBasedRelaxed`. It is a
  degraded mode too, and `docker logs strafer_navigation` names it both ways:
  ```
  Robot's own global-costmap cell is 253 (>= inscribed 253); 'GridBased' cannot plan from it. Admitting 'GridBasedRelaxed' ...
  'GridBasedRelaxed' produced this goal's path: the nav2 lane is planning from a start the primary planner refuses.
  ```
  Neither lane's escape helps from *deep* inside the halo: `GridBasedRelaxed`
  clears only the robot's own cell, so it needs a free neighbour to propagate
  into. Measured: it plans from ~0.20 m off lethal and refuses from ~0.15 m.

  `SUCCEEDED` = fine. `ABORTED` / *"Starting point in lethal space"* = still
  wedged; free it with a manual holonomic strafe on `/cmd_vel` (Nav2's `/backup`
  refuses for the same reason).
- Direct (no-VLM) goal for quick policy checks:
  ```bash
  docker compose -f docker-compose.yml exec navigation bash -lc \
    'source /opt/ros/humble/setup.bash && source /ws/install/setup.bash; \
     ros2 action send_goal /strafer_inference/navigate_to_pose nav2_msgs/action/NavigateToPose \
     "{pose: {header: {frame_id: map}, pose: {position: {x: -1.0, y: 0.3}, orientation: {w: 1.0}}}}"'
  ```
  (pick a **reachable** goal — probe first with nav2 `ComputePathToPose`; goals behind the robot / outside the seeded map return "no valid path".)
