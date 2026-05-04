# Sim-in-the-loop integration runbook

End-to-end test of the DGX Spark ↔ Jetson Orin Nano pipeline using
Isaac Sim in place of the real robot. Success criterion: an operator
submits a natural-language mission from an SSH session on the Jetson,
Nav2 plans a path against the simulated D555 streams the DGX
publishes, the Jetson publishes `/cmd_vel` in response, and the
simulated robot visibly moves in Isaac Sim to carry out the command.

**No VLM or CLIP training is needed for the bridge mission stages** —
the VLM service is used for ad-hoc grounding during missions, and CLIP
is used only for `verify_arrival` checks (optional). Stage 6 covers
the separate VLM/CLIP **data-collection** path through the same
bridge.

The runbook is designed to localise failures by building the pipeline
up in stages. Each stage has a clean go/no-go check; if a stage fails,
the troubleshooting tree at its end tells you what to inspect before
moving on. **Do not skip stages** — the later stages assume the
earlier ones work.

---

## Related docs

- [`docs/example_commands_cheatsheet.md`](example_commands_cheatsheet.md)
  is the canonical place for the operator one-liners. Where this
  runbook references a command, prefer the cheatsheet's exact form.
- [`docs/PERF_INVESTIGATION_SIM_IN_THE_LOOP.md`](PERF_INVESTIGATION_SIM_IN_THE_LOOP.md)
  Findings 8-10 + Recommendations carry the perf attribution that
  picked `make sim-bridge` (headless) as the default.
- [`docs/SYSTEM_FLOW_DIAGRAMS.md`](SYSTEM_FLOW_DIAGRAMS.md) Flow 5
  (sim-in-the-loop) and Flow 6 (real-robot execution) — the bridge
  exercises Flow 5's outer shell; the manual-mission stage exercises
  Flow 6's control path with simulated sensors.
- [`docs/tasks/context/bridge-runtime-invariants.md`](tasks/context/bridge-runtime-invariants.md)
  — cmd_vel normalization, telemetry/camera split, headless vs
  `--viz kit`, scene-side prerequisites. **Read this before
  diagnosing bridge issues.**
- DGX prerequisite: Isaac Sim must boot under `AppLauncher` and a
  smoke-test env (e.g.
  `Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0`) must reset
  cleanly before starting this runbook. If `AppLauncher` doesn't
  boot, nothing else here will work.
- Agent prompts (when handing this to two parallel assistants):
  - [`docs/INTEGRATION_PROMPT_DGX.md`](INTEGRATION_PROMPT_DGX.md)
  - [`docs/INTEGRATION_PROMPT_JETSON.md`](INTEGRATION_PROMPT_JETSON.md)

---

## Scope and non-goals

**In scope**
- Cross-host DDS discovery verification.
- DGX-side ROS 2 bridge publishing simulated D555 + odometry + TF
  topics that the Jetson's autonomy stack normally consumes.
- Jetson-side bringup (`bringup_sim_in_the_loop.launch.py`) —
  perception re-stamping, `/scan` synthesis, RTAB-Map, Nav2, goal
  projection, executor.
- Manual mission submission via `strafer-autonomy-cli` and visible
  robot motion in Isaac Sim, including the cmd_vel-normalization
  cross-check.
- Autonomous mission sweep via `run_sim_in_the_loop.py --mode harness`
  against a scene's `scene_metadata.json`.
- VLM/CLIP data-collection sweep through the same bridge (Stage 6).

**Out of scope (explicitly deferred)**
- Real-robot execution on hardware — see
  [`docs/SYSTEM_FLOW_DIAGRAMS.md`](SYSTEM_FLOW_DIAGRAMS.md) Flow 6.
  Real-robot bringup uses `autonomy.launch.py` / `base.launch.py`,
  which this runbook does not exercise.
- IMU-driven SLAM — the Isaac Sim ROS 2 bridge has no
  `ROS2PublishImu` node, so RTAB-Map runs visual-only. See
  [`docs/tasks/DEFERRED_WORK.md`](tasks/DEFERRED_WORK.md).

---

## Host topology

See [`docs/tasks/context/repo-topology.md`](tasks/context/repo-topology.md)
for hostnames, IPs, repo paths, conda envs, ROS distro, and
DDS/domain settings. The runbook assumes:

- DGX Spark at `192.168.50.196`, conda env `env_isaaclab3` active.
- Jetson Orin Nano at `192.168.50.24`, colcon `install/setup.bash`
  sourced.
- Both on the same LAN, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`,
  `ROS_DOMAIN_ID=42`, no multicast-blocking firewall between them.

---

## Prerequisites checklist

Before starting Stage 1, **all** of the following must be true. If
anything here is red, fix it first — do not paper over it later.

**Both hosts**
- [ ] On the same LAN, can ping each other by IP.
- [ ] `cyclonedds` / `rmw_cyclonedds_cpp` available to ROS 2.
- [ ] `ROS_DOMAIN_ID=42` exported in the shell you will use.
- [ ] `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` exported in the shell
      you will use.

**DGX only**
- [ ] `source env_setup.sh` runs clean from `~/Workspace/Sim2RealLab`
      (prints the configured paths and exports `LD_PRELOAD`,
      `STRAFER_BLENDER_BIN`, `ISAACSIM_PATH`, `ROS_DOMAIN_ID`,
      `RMW_IMPLEMENTATION`).
- [ ] `conda activate env_isaaclab3` succeeds.
- [ ] Isaac Sim boots under `AppLauncher` and a smoke-test env (e.g.
      `make sim-bridge` or
      `$ISAACLAB -p Scripts/test_strafer_env.py
       --env Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0
       --num_envs 1 --duration 5 --headless`) reaches the env-step
      loop without errors.
- [ ] `strafer_msgs` is importable from `env_isaaclab3` — the
      sim-in-the-loop harness imports
      `strafer_msgs.action.ExecuteMission` and
      `strafer_msgs.srv.GetMissionStatus`. If that fails,
      colcon-build `strafer_msgs` and add its `install/setup.bash`
      to the shell.
- [ ] VLM service reachable from the Jetson at
      `http://192.168.50.196:8100/health` — expected
      `{"status":"ok","model_loaded":true}`. Start with
      `make serve-vlm` if it isn't up.

**Jetson only**
- [ ] `colcon build` of `strafer_ros` is current;
      `install/setup.bash` sourced.
- [ ] `source source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env`
      sets `RMW_IMPLEMENTATION`, `ROS_DOMAIN_ID=42`,
      `HARDWARE_PRESENT=false`, and bumps
      `STRAFER_NAVIGATION_TIMEOUT_S` for slow-RTF sim runs.
- [ ] `strafer-executor` and `strafer-autonomy-cli` on `PATH`
      (installed via `pip install -e source/strafer_autonomy`).

---

## Stage 1 — DDS discovery across the LAN

**Goal**: the two hosts see each other on ROS 2. No topics exist yet,
but `ros2 topic list` on one host will surface topics the other host
publishes.

**Setup**

On **both hosts**, in a fresh shell:

```bash
# DGX
cd ~/Workspace/Sim2RealLab
source env_setup.sh
conda activate env_isaaclab3
```

```bash
# Jetson
cd ~/workspaces/Sim2RealLab
source source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env
source install/setup.bash
```

**Test**

On the Jetson, run a dummy talker:

```bash
# Jetson
ros2 run demo_nodes_cpp talker
```

On the DGX, list topics:

```bash
# DGX
ros2 topic list
```

**Go/no-go**: `/chatter` appears in the DGX's `ros2 topic list`
output. If it does, reverse the roles (DGX talker, Jetson lister) and
confirm the same in the other direction. Kill both.

**Troubleshooting**

| Symptom | Most likely cause | What to check |
|---|---|---|
| `/chatter` missing from DGX | DDS not routing | `echo $ROS_DOMAIN_ID` + `echo $RMW_IMPLEMENTATION` on both hosts. Both values must match exactly. |
| Discovery fails after a reboot | Multicast blocked on the LAN | `export CYCLONEDDS_URI='<CycloneDDS><Domain><General><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'` on both hosts. |
| One host sees its own topics but not the other's | Firewall, hosts on different subnets | `ping` the other host; `ip route` on both; `sudo ufw status`. |
| ROS commands hang with no output | `ros2 daemon` stuck with stale settings from a prior shell | `ros2 daemon stop && ros2 daemon start` on both hosts. |

---

## Stage 2 — DGX bridge alone, manually driven

**Goal**: the DGX publishes simulated D555 + odometry + TF topics; an
operator can drive the simulated robot by manually publishing
`/cmd_vel`. The Jetson plays no role yet.

**Setup**

On the **DGX**, headless is the default daily-driver:

```bash
cd ~/Workspace/Sim2RealLab
source env_setup.sh
make sim-bridge
```

`make sim-bridge` is `--mode bridge --headless --enable_cameras`.
Use `make sim-bridge-gui` (adds `--viz kit`, drops `--headless`)
**only** when you need to watch the editor viewport for visual
debugging — the editor viewport adds ~85 ms / loop on a
DISPLAY-attached host (see
[`docs/PERF_INVESTIGATION_SIM_IN_THE_LOOP.md`](PERF_INVESTIGATION_SIM_IN_THE_LOOP.md)
Findings 8-10). For visual debugging while running headless, use the
Jetson-side Foxglove viewer in Stage 3.5 instead of `--viz kit`.

Expected console output includes, in order:

```
[INFO] ... isaac-sim ... starting
[sim_in_the_loop] bridge graph built at /World/ROS2Bridge
[sim_in_the_loop] chassis_prim=/World/envs/env_0/Robot/strafer/body_link
[sim_in_the_loop] color camera prim=/World/envs/env_0/Robot/strafer/body_link/d555_camera_perception
[sim_in_the_loop] action tensor shape = (1, 3)
```

**Test A — bridge topics are publishing**

Open a second shell on the DGX:

```bash
# DGX shell 2
source env_setup.sh
ros2 topic list | sort
```

Expected (extras allowed; these are the must-haves):

```
/cmd_vel
/d555/color/camera_info
/d555/color/image_raw
/d555/depth/camera_info
/d555/depth/image_rect_raw
/strafer/odom
/tf
```

Verify the camera topic is active:

```bash
# DGX shell 2
ros2 topic hz /d555/color/image_raw
```

Expected: ~30 Hz after a few seconds.

**Test B — robot moves when you publish /cmd_vel**

Publish a forward-velocity Twist:

```bash
# DGX shell 2
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.2}}'
```

Watch `/strafer/odom` in a third shell:

```bash
# DGX shell 3
ros2 topic echo /strafer/odom --once
```

**Go/no-go**: `pose.pose.position.x` in the odometry message advances
over time when `/cmd_vel` is active and stays put when it stops. If
running with `make sim-bridge-gui`, the robot should visibly translate
forward at ~0.2 m/s in the Kit viewport.

**Troubleshooting**

| Symptom | Most likely cause | What to check |
|---|---|---|
| `run_sim_in_the_loop.py` crashes on `enable_extension("isaacsim.ros2.bridge")` | Extension not bundled with the Isaac Sim build | `ls $ISAACSIM_PATH/exts/isaacsim.ros2.bridge/` — directory must exist. |
| `build_bridge_graph` crashes on a missing camera prim | env not fully spawned when graph is built | Inspect the `chassis_prim=...` and `color camera prim=...` lines — both must point at prims that exist in the stage. |
| `/d555/color/image_raw` not in `ros2 topic list` | Bridge extension enabled but graph not evaluated | Check `[sim_in_the_loop] bridge graph built at ...` appeared. If not, graph build silently failed — re-run with full Kit logs. |
| `ros2 topic hz` returns 0 Hz | Isaac Sim is stuck at iteration 0 | The camera render is not executing. Try `make sim-bridge-gui` once to confirm a frame renders in the viewport. |
| `/strafer/odom` pose doesn't advance under `/cmd_vel` | Bridge's `SubscribeTwist` node not wired or env action layout mismatch | Read the action-tensor shape line — must be `(1, 3)`. If it's `(1, 4)` the env config has changed and the launcher's injection map in `_run_bridge_mode` needs updating. |

For perf attribution (when bridge throughput regresses):

```bash
# DGX
$ISAACLAB -p source/strafer_lab/scripts/run_sim_in_the_loop.py \
    --mode bridge --headless --enable_cameras \
    --profile --profile-interval 10 --profile-window 200
```

`--profile` rolls p50/p99 across `cmd_vel read`, `env.step (total)`,
`publish_state`, `simulation_app.update`, plus per-tick `sim.step`
and `sim.render`. Compare to the baselines in
[`docs/PERF_INVESTIGATION_SIM_IN_THE_LOOP.md`](PERF_INVESTIGATION_SIM_IN_THE_LOOP.md)
Findings 8-10 to see which phase regressed.

---

## Stage 3 — Jetson consumes the bridge's sensor topics

**Goal**: the Jetson brings up its perception / SLAM / Nav2 stack
against the DGX-published topics, produces the `*_sync` re-stamped
variants and `/scan`, and runs RTAB-Map without the real hardware
drivers.

**Setup**

Keep the DGX bridge running from Stage 2 (`make sim-bridge`).

On the **Jetson**:

```bash
cd ~/workspaces/Sim2RealLab
source source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env
source install/setup.bash
ros2 launch strafer_bringup bringup_sim_in_the_loop.launch.py \
    vlm_url:=http://192.168.50.196:8100 \
    planner_url:=http://192.168.50.196:8200
```

Expected: executor registers the `execute_mission` action, RTAB-Map
logs `Received first image` once the DGX sensor streams arrive.

**Test A — bridge topics are visible on the Jetson**

In another Jetson shell:

```bash
# Jetson shell 2
source /opt/ros/humble/setup.bash
source ~/workspaces/Sim2RealLab/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42

ros2 topic list | sort
```

Expected to include the bridge-published raw topics plus the
Jetson-synthesised derived topics:

```
/cmd_vel
/d555/aligned_depth_to_color/image_sync        ← timestamp_fixer
/d555/color/camera_info
/d555/color/image_raw
/d555/color/image_sync                          ← timestamp_fixer
/d555/depth/camera_info
/d555/depth/image_rect_raw
/rtabmap/map
/scan                                           ← depthimage_to_laserscan
/strafer/odom
/tf
```

Verify the TF tree is complete:

```bash
# Jetson shell 2
ros2 run tf2_tools view_frames
```

Expected chain: `map` → `odom` (from RTAB-Map) → `base_link` (from
the bridge) → `d555_link` (from the bridge) →
`d555_color_optical_frame` (from `strafer_description`).

**Test B — `/scan` has content**

```bash
# Jetson shell 2
ros2 topic hz /scan
ros2 topic echo /scan --once | head -30
```

Expected: ~15 Hz, with a `ranges[]` array containing both `inf` and
finite distances.

**Go/no-go**: RTAB-Map's console shows `Adding node ...` entries,
meaning it's ingesting the simulated camera frames and building a
map. `/rtabmap/map` in `ros2 topic list`, with non-empty `data[]`
when echoed.

**Troubleshooting**

| Symptom | Most likely cause | What to check |
|---|---|---|
| Bridge topics missing on the Jetson, present on the DGX | DDS discovery fell over after the Jetson joined | Repeat Stage 1's discovery check from the Jetson side. |
| `/d555/color/image_sync` missing, raw topic present | `timestamp_fixer` node not running | Check the launch log for `timestamp_fixer` errors; re-launch with `output='screen'`. |
| `/scan` missing | `depthimage_to_laserscan` is part of `strafer_slam` — did it start? | Inspect `slam.launch.py` includes; common cause is a bad `frame_id` that makes the node reject the depth image. |
| RTAB-Map not accepting frames | Frame-id mismatch or TF race | `ros2 topic echo /d555/color/image_raw --once` — header.frame_id must be `d555_color_optical_frame`. TF from `base_link → d555_link` must exist before frames start arriving. |
| TF tree has `base_link → d555_link` missing | `strafer_description` (URDF) did not start | Confirm the launch includes `description.launch.py`; it publishes the URDF transform via `robot_state_publisher`. |
| Nav2 complains about `/map` or `/odom` not available | Start-ordering race — RTAB-Map isn't ready yet | Wait 20-30 s after launch; if Nav2 still refuses to finish, check `/tf` chain. |

---

## Stage 3.5 — Headless visual debugger (Foxglove over SSH)

**Goal**: an operator on a workstation can see the Jetson's live
camera + TF + map state without a monitor on the Jetson, and without
running Isaac Sim's editor viewport on the DGX.

**Why**: `make sim-bridge` (headless) is the daily-driver bridge
target because the editor viewport is the largest single perf cost.
Headless drops bridge throughput's biggest cost but loses the only
place the operator currently *sees* what the camera + robot are
doing. The Jetson side has every relevant topic already (RGB, depth,
`/tf`, `/scan`, `/rtabmap/map`, `/strafer/odom`); exposing them via
`foxglove_bridge` and viewing them in Foxglove Studio on the
operator's workstation closes the gap.

**Setup (one-time)**

On the **Jetson**:

```bash
sudo apt install ros-humble-foxglove-bridge
```

`bringup_sim_in_the_loop.launch.py` defaults to `viewer:=true`, which
starts `foxglove_bridge` on TCP 8765. Pass `viewer:=false` to disable
it (e.g. for non-debug missions or when the dep isn't installed).

**Connect from the operator's workstation**

```bash
# operator workstation — opens an SSH tunnel that forwards the bridge
# port back to the workstation's loopback
ssh -L 8765:localhost:8765 jetson-desktop
```

In Foxglove Studio (browser at <https://app.foxglove.dev/> or the
desktop app):

1. **Open connection** → **Foxglove WebSocket**.
2. URL: `ws://localhost:8765`.
3. **Layout** → **Import from file…** →
   `source/strafer_ros/strafer_bringup/foxglove/strafer_layout.json`
   (from a checkout of this repo on the workstation).

**Go/no-go**: the RGB panel shows `/d555/color/image_raw` updating at
~30 Hz, the depth panel renders `/d555/depth/image_rect_raw` with a
near-to-6 m colormap, and the 3D panel shows the URDF following
`base_link` with `/scan` and `/rtabmap/map` overlaid.

**Same launch on the real robot.** The visualizer subscribes to the
same `/d555/...` topic names that the real D555 driver publishes, so
Foxglove on the workstation works against the real robot with zero
config change — only the SSH target (`jetson-desktop`) changes.

**Troubleshooting**

| Symptom | Most likely cause | What to check |
|---|---|---|
| `foxglove_bridge` not in `ros2 node list` | Package not installed, or `viewer:=false` | `dpkg -s ros-humble-foxglove-bridge`; relaunch with `viewer:=true`. |
| Studio fails to connect to `ws://localhost:8765` | SSH tunnel down, or port already in use on the workstation | Check the `ssh -L` shell is still alive; `lsof -iTCP:8765` on the workstation; pick a different `viewer_port:=...` and update both `-L` and the URL. |
| Studio connects but topics list is empty | Jetson bringup isn't producing topics yet | `ros2 topic list` on the Jetson; rerun Stage 3 if RTAB-Map / timestamp_fixer haven't started. |
| RGB panel black, others fine | Camera not yet streaming | `ros2 topic hz /d555/color/image_raw` on the Jetson; is the DGX bridge running? |
| Depth all-white | Default colormap range bad for this scene | In Foxglove, open the Image panel settings and bump `maxValue` higher than 6 m, or switch to the `"viridis"` colormap. RGB now extends through the full room (frustum decoupled from the 6 m depth-sensor saturation), so the visible scene is wider than the depth saturation cap suggests. |

---

## Stage 4 — Manual mission submission end-to-end

**Goal**: an operator submits a natural-language mission via the CLI
on the Jetson. The executor talks to the DGX planner + VLM over
HTTP, builds a skill sequence, runs it against the simulated
sensors, and the simulated robot visibly moves in Isaac Sim.

**Setup**

Keep the DGX bridge running (`make sim-bridge`) and the Jetson
bringup running from Stage 3. The Jetson autonomy executor must be
up (`ros2 action list` shows `/execute_mission`).

**Test**

Open a third shell on the Jetson and submit a mission:

```bash
# Jetson shell 3
source /opt/ros/humble/setup.bash
source ~/workspaces/Sim2RealLab/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42

strafer-autonomy-cli submit "move forward 1 meter"
```

Expected console flow:

```
{"accepted": true, "request_id": "..."}
feedback: state=planning  step_id= skill=
feedback: state=in_progress  step_id=step_0  skill=navigate_to_pose
...
{"accepted": true, "mission_id": "...", "final_state": "succeeded", ...}
```

**cmd_vel normalization cross-check**

`/cmd_vel` arrives in physical units (m/s, rad/s); both bridge paths
divide by `MAX_LINEAR_VEL` / `MAX_ANGULAR_VEL` (from
`strafer_shared.constants`) before writing to the action tensor (see
[`docs/tasks/context/bridge-runtime-invariants.md`](tasks/context/bridge-runtime-invariants.md#cmd_vel-normalization-contract-both-paths)).
A Nav2 `cmd_vel` of `linear.x = 0.5 m/s` should produce a body
linear velocity of ~0.5 m/s in sim, **not** the 0.78 m/s a missing
normalization step would yield (the unnormalized action saturates
the per-wheel motor cap; the saturated action de-normalizes back to
~`MAX_LINEAR_VEL` ≈ 0.78 m/s).

While a mission is running, sample both ends of the path:

```bash
# DGX shell 2 — what Nav2 is asking for
ros2 topic echo /cmd_vel --once
```

```bash
# DGX shell 3 — what the env actually applied
# Look at the bridge's stdout. The bridge logs the env's body-frame
# linear velocity each loop when DEBUG logging is enabled, sourced
# from `env.unwrapped.scene["robot"].data.root_lin_vel_b`. If those
# logs aren't there, drop a one-shot print into _run_bridge_mode
# next to the action-tensor write.
```

The check: confirm `root_lin_vel_b[0]` magnitude tracks the
published `/cmd_vel.linear.x` within ~10 % at steady state. A
~1.57× discrepancy (e.g. 0.5 m/s commanded, ~0.78 m/s body) means
the normalization step regressed in one of the bridge paths — start
at
[`bridge-runtime-invariants.md`](tasks/context/bridge-runtime-invariants.md#cmd_vel-normalization-contract-both-paths)
and walk both code sites it names.

**Go/no-go**: the simulated robot visibly translates / rotates in
the Isaac Sim viewport (or its `/strafer/odom` position advances in
step with the mission), and `strafer-autonomy-cli` prints
`final_state: succeeded`. A `final_state: failed` with a sensible
`message` (e.g. "goal outside costmap") is also OK at this stage —
it means the control path works even if the specific command was
infeasible.

**Test variant — semantic mission**

```bash
# Jetson shell 3
strafer-autonomy-cli submit "go to the door"
```

Exercises the VLM grounding path: the planner inserts a
`scan_for_target` skill, the executor calls the VLM's `/ground`
endpoint, and when a match is found the goal-projection node turns
the bbox into a map-frame pose for Nav2. Expected to succeed if the
scene has a door the VLM can see at mission start.

**Troubleshooting**

| Symptom | Most likely cause | What to check |
|---|---|---|
| `strafer-autonomy-cli submit` hangs at "Action server not available" | Executor not running or on a different domain | `ros2 node list` should include `/strafer_autonomy_command_server`; `ros2 action list` should include `/execute_mission`. |
| Executor starts then immediately errors on health check | VLM or planner service unreachable from the Jetson | From the Jetson, `curl http://192.168.50.196:8100/health`. If that fails, LAN routing — not SSH — is the issue. |
| Mission sticks in `state=planning` | Planner timeout or parse error | Check the Jetson's `HttpPlannerClient` logs and the DGX planner's stdout. Common cause: planner returning malformed JSON. Retry with simpler wording. |
| `scan_for_target` always returns `not_found` | VLM's frame sampling is wrong, or grounding threshold too strict | Inspect the `/describe` output from the VLM on a known frame — does it mention the target at all? |
| Nav2 plans but `/cmd_vel` never fires | Controller crashed or costmap empty | Inspect Nav2 logs; ensure `/scan` was healthy at the time Nav2 initialised its costmap. |
| Robot moves at ~1.57× the commanded speed (e.g. `cmd_vel.linear.x=0.5` → body ~0.78 m/s) | cmd_vel normalization regressed in one of the bridge paths | See [`bridge-runtime-invariants.md`](tasks/context/bridge-runtime-invariants.md#cmd_vel-normalization-contract-both-paths). Both `_run_bridge_mode` (mainloop) and `runtime_env._build_action` (harness) must `_clamp_unit` after dividing by `MAX_LINEAR_VEL` / `MAX_ANGULAR_VEL`. |
| Mission times out at exactly the wall-clock 90 s mark while sim RTF is low | `STRAFER_NAVIGATION_TIMEOUT_S` not picked up in the executor's shell | Confirm the Jetson sourced `env_sim_in_the_loop.env` (it bumps the value to handle slow-RTF sim runs). The executor's nav timeout is sim-time-aware and respects `use_sim_time`, so the budget is in `/clock` advance, not wall clock. |
| Robot moves the wrong direction | Mecanum wheel-axis signs mismatched between sim and real (should be identical by contract) | Compare `strafer_shared.constants.WHEEL_AXIS_SIGNS` (sim) with the real robot's RoboClaw wiring — a mismatch is a contract break, not a sim-in-the-loop bug. |

---

## Stage 5 — Harness mode (autonomous mission sweep)

**Goal**: the DGX harness walks `scene_metadata.json`'s object list,
submits one mission per target, captures reachability-labelled
frames into `frames.jsonl`. No human driver.

**Prerequisites**
- Stages 1-4 green.
- One generated Infinigen scene under
  `Assets/generated/scenes/<scene_name>/` containing `scene.usdc`
  + `scene_metadata.json`. A real Infinigen scene with furniture is
  produced by `prep_room_usds.py` (which calls `generate_indoors`
  with `fast_solve.gin singleroom.gin`) followed by
  `generate_scenes_metadata.py` to author the combined
  `scenes_metadata.json` with `floor_top_z` per scene.

**Test**

Stop the DGX `make sim-bridge` run from Stage 2. On the DGX:

```bash
cd ~/Workspace/Sim2RealLab
SCENE_META=Assets/generated/scenes/<scene_name>/scene_metadata.json \
SCENE_USD=Assets/generated/scenes/<scene_name>/scene.usdc \
OUTPUT_DIR=data/sim_in_the_loop/<scene_name> \
MAX_MISSIONS=3 \
make sim-harness
```

Or call the script directly if you need flag overrides
`make sim-harness` doesn't expose:

```bash
$ISAACLAB -p source/strafer_lab/scripts/run_sim_in_the_loop.py \
    --mode harness \
    --scene-metadata Assets/generated/scenes/<scene_name>/scene_metadata.json \
    --scene-usd      Assets/generated/scenes/<scene_name>/scene.usdc \
    --output         data/sim_in_the_loop/<scene_name> \
    --max-missions   3 \
    --headless --enable_cameras
```

Keep the Jetson bringup from Stage 3 running.

**Go/no-go**: each mission in `--max-missions 3` gets logged with
`reachable=True|False`. The output directory now contains three
`episode_NNNN/` subdirs, each with a populated `frames.jsonl` and
`frame_*.jpg` files.

```bash
# DGX
ls data/sim_in_the_loop/<scene_name>/
head -n 2 data/sim_in_the_loop/<scene_name>/episode_0000/frames.jsonl
```

Expected fields: `frame_id`, `image_path`, `robot_pos`,
`robot_quat`, `bboxes`, `mission_id`, `target_label`,
`target_position_3d`, `reachability`, `mission_state`.

**Troubleshooting**

| Symptom | Most likely cause | What to check |
|---|---|---|
| `Ros2MissionApi` fails to open at harness start | `strafer_msgs` not on DGX `PYTHONPATH` | Confirm `python -c "from strafer_msgs.action import ExecuteMission"` works in `env_isaaclab3`. Re-source the colcon install. |
| `Action server ... did not become available` | Jetson executor not up or on wrong domain | Same as Stage 4: verify `/execute_mission` action exists from the DGX via `ros2 action list`. |
| Every mission ends `reachable=False` at timeout | Nav2 never finishes plans against the current scene | Start from Stage 4 — manually submit a mission and verify it works before letting the harness submit dozens. |
| Mission generator yields zero missions | `scene_metadata.json` has no parseable objects | `python -c "import json; print(len(json.load(open('.../scene_metadata.json'))['objects']))"`. If zero, re-run `extract_scene_metadata.py --from-usd`. |
| `frames.jsonl` exists but lacks `reachability` field | Outcome was never captured — harness crashed mid-mission | Search the DGX stdout for `[mission_id] harness crashed mid-mission`; the stack trace will point at the break. |

---

## Stage 6 — VLM/CLIP data-collection sweep through the bridge

**Goal**: produce VLM SFT data (grounding JSONL) and CLIP CSV from
the same bridge harness, so the next VLM/CLIP fine-tune runs against
frames that match what the bridge actually publishes (640×360
`d555_camera_perception` resolution, current scene set, current
prim-label set).

This stage exercises a separate output of the same harness path —
not the Nav2 mission control path of Stages 4-5. The bridge is still
the publisher; the consumer is `prepare_vlm_finetune_data.py` →
`finetune_clip.py`, not `bringup_sim_in_the_loop.launch.py`.

**Prerequisites**
- Stages 1-3 green (DDS + bridge + Jetson bringup are not strictly
  required for data collection, but Stage 1 catches RMW
  misconfiguration before it manifests later).
- An Infinigen scene set already generated under
  `Assets/generated/scenes/`. Use `prep_room_usds.py` to author new
  scenes if needed.

**Operator runs / agent verifies — per-scene metadata**

```bash
# DGX — author per-scene scene_metadata.json (rooms, polygons, semantic
# tags, relations) by walking the Infinigen Blender export. Run once
# per scene.
$ISAACLAB -p source/strafer_lab/scripts/extract_scene_metadata.py \
    --from-usd \
    --usd Assets/generated/scenes/<scene_name>/scene.usdc \
    --output Assets/generated/scenes/<scene_name>/ \
    --label-from-prim-names
```

Verify: `Assets/generated/scenes/<scene_name>/scene_metadata.json`
exists and has a non-empty `objects[]` list.

**Operator runs / agent verifies — combined metadata + floor heights**

```bash
# DGX — produce the combined scenes_metadata.json that the env's
# lift_ground_plane_to_floor startup event reads at bridge launch.
$ISAACLAB -p source/strafer_lab/scripts/generate_scenes_metadata.py
```

Verify: `Assets/generated/scenes/scenes_metadata.json` lists each
scene with a `floor_top_z` value. Without this file the bridge's
floor lift is a no-op and the robot can fall through the Infinigen
floor mesh — this is now baked at scene-generation time, so a
bouncing/sinking robot is a sign that one of these two metadata
files is missing or stale, **not** a bridge bug.

**Operator runs / agent verifies — capture frames over the bridge**

```bash
# DGX
SCENE_META=Assets/generated/scenes/<scene_name>/scene_metadata.json \
SCENE_USD=Assets/generated/scenes/<scene_name>/scene.usdc \
OUTPUT_DIR=data/sim_in_the_loop/<scene_name> \
MAX_MISSIONS=20 \
make sim-harness
```

Verify: `data/sim_in_the_loop/<scene_name>/episode_NNNN/frames.jsonl`
populated; `frame_*.jpg` files at 640×360.

**Operator runs / agent verifies — VLM SFT + CLIP CSV**

```bash
# DGX — single-object grounding + 1:3 negatives + ~20% multi-object +
# ~10% description preservation. Output is the JSONL the VLM LoRA SFT
# job consumes.
$ISAACLAB -p source/strafer_lab/scripts/prepare_vlm_finetune_data.py \
    --frames data/sim_in_the_loop/<scene_name>/ \
    --scene-metadata Assets/generated/scenes/<scene_name>/scene_metadata.json \
    --output data/vlm_sft/<scene_name>/

# DGX — CLIP contrastive fine-tune, exports clip_visual.onnx +
# clip_text.onnx for the Jetson semantic map.
$ISAACLAB -p source/strafer_lab/scripts/finetune_clip.py \
    --csv data/vlm_sft/<scene_name>/clip_pairs.csv \
    --output models/clip_<scene_name>/
```

**Go/no-go**: `data/vlm_sft/<scene_name>/grounding.jsonl` has ≥ one
example per object label in the scene; `models/clip_<scene_name>/`
contains `clip_visual.onnx` + `clip_text.onnx`. The MLflow run
records non-NaN contrastive loss decreasing across epochs.

**Troubleshooting**

| Symptom | Most likely cause | What to check |
|---|---|---|
| `extract_scene_metadata.py` reports `0 objects` | Blender export missing `semanticLabel` USD attrs | Pass `--label-from-prim-names`; re-export the scene via `prep_room_usds.py` if prim names lack semantic info. |
| Bridge harness frames have `frame_*.jpg` at 1280×720, not 640×360 | `IsaacCreateRenderProduct` resolution not pinned | See [`bridge-runtime-invariants.md`](tasks/context/bridge-runtime-invariants.md#camera-resolutions-sim-mirrors-real) — the `inputs:width` / `inputs:height` must be set explicitly on the render product node. |
| `prepare_vlm_finetune_data.py` complains about missing labels | `scene_metadata.json` and `frames.jsonl` disagree on label set | Re-run `extract_scene_metadata.py` then `generate_scenes_metadata.py` to refresh both per-scene + combined metadata. |
| `finetune_clip.py` MLflow loss = NaN immediately | Inputs unnormalized or empty positive pairs | Inspect `clip_pairs.csv` — must have ≥ 32 unique (image, text) pairs per epoch. |

---

## End-state artifacts

On successful completion of all stages:

- DGX: Isaac Sim + bridge running; one or more
  `data/sim_in_the_loop/<scene_name>/episode_*/` dataset directories
  (Stage 5/6); optionally `data/vlm_sft/<scene_name>/` and
  `models/clip_<scene_name>/` (Stage 6).
- Jetson: bringup still running; RTAB-Map database at
  `~/.ros/rtabmap.db` (Stage 3).
- Confirmed that the robot's motion in Isaac Sim is being driven by
  Nav2's `/cmd_vel` on the Jetson (Stage 4), with body velocity
  matching `/cmd_vel` magnitude within ~10 %.
- A datapoint per mission with a reachability label (Stage 5).

**Shut-down order**: Stop the harness run (or Stage 2 bridge run) on
the DGX first — that stops the simulator and the bridge topics. Then
stop the Jetson bringup. RTAB-Map persists its database on graceful
shutdown.

---

## Known caveats

- **IMU data is absent.** The bridge has no `ROS2PublishImu` node.
  RTAB-Map runs visual-only in this test. If SLAM quality proves too
  poor in a particular scene, this is the first thing to revisit.
  See [`docs/tasks/DEFERRED_WORK.md`](tasks/DEFERRED_WORK.md).
- **Scene swap requires re-launching.** The harness is single-scene
  per launch. `IsaacLabEnvAdapter.reset(scene_name=...)` rejects
  cross-scene calls on purpose — swap scenes by re-running the DGX
  script with a different `--scene-usd`.
- **`goal_position_noise` is not the default.** Pre-deployment RL
  policies were trained without goal noise; VLM-grounded goals
  carry ±0.2-0.5 m localization error. Expect some goal-pose
  oscillation near the target. Tracked under
  [`docs/tasks/active/policy-goal-noise-training.md`](tasks/active/policy-goal-noise-training.md).
- **Electronics masses are missing from the USD.** The chassis
  inertia underestimates the real robot. Fine control behaves
  differently in sim vs. real until this is fixed. See
  [`docs/tasks/DEFERRED_WORK.md`](tasks/DEFERRED_WORK.md).
