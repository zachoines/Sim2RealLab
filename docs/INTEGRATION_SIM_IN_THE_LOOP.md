# Sim-in-the-loop integration runbook

End-to-end test of the DGX Spark ↔ Jetson Orin Nano cross-host pipeline
using Isaac Sim instead of the real robot. Success criterion: an
operator submits a natural-language mission from an SSH session on the
Jetson, Nav2 plans a path against the simulated D555 streams the DGX
publishes, the Jetson publishes `/cmd_vel` in response, and the
simulated robot visibly moves in Isaac Sim to carry out the command.

This is the first cross-host integration test after the packages were
wired up independently. **No VLM or CLIP training is needed for this
test** — the VLM service is used only for ad-hoc grounding during
missions, and CLIP is used only for `verify_arrival` checks (optional
at this stage).

The runbook is designed to localize failures by building the pipeline
up in stages. Each stage has a clean go/no-go check; if a stage fails,
the troubleshooting tree at its end tells you what to inspect before
moving on. **Do not skip stages** — the later stages assume the
earlier ones work.

---

## Related docs

- [System flow diagrams](SYSTEM_FLOW_DIAGRAMS.md) — what the end state
  looks like, especially Flow 5 (sim-in-the-loop) and Flow 6
  (real-robot execution). The bridge mode of this runbook exercises
  Flow 5's outer shell; the manual-mission stage exercises Flow 6's
  control path with simulated sensors.
- [Isaac Sim install validation](VALIDATE_ISAAC_SIM_AND_INFINIGEN.md) —
  must be green on the DGX before starting this runbook. If
  `AppLauncher` doesn't boot cleanly, nothing else here will work.
- Agent prompts (if you're handing this to two parallel assistants):
  - [INTEGRATION_PROMPT_DGX.md](INTEGRATION_PROMPT_DGX.md)
  - [INTEGRATION_PROMPT_JETSON.md](INTEGRATION_PROMPT_JETSON.md)

---

## Scope and non-goals

**In scope**
- Cross-host DDS discovery verification.
- DGX-side ROS 2 bridge publishing the simulated D555 + odometry + TF
  topics that the Jetson's autonomy stack normally consumes.
- Jetson-side bringup (`bringup_sim_in_the_loop.launch.py`) — perception
  re-stamping, `/scan` synthesis, RTAB-Map, Nav2, goal projection,
  executor.
- Manual mission submission via `strafer-autonomy-cli` and visible
  robot motion in Isaac Sim.
- Autonomous mission sweep via `run_sim_in_the_loop.py --mode harness`
  against a scene's `scene_metadata.json`.

**Out of scope (explicitly deferred)**
- VLM or CLIP training runs — covered by Flow 1 and Flow 2 in the
  system flow diagrams. The VLM service is expected to be running but
  its weights don't need to be fine-tuned for this test.
- Real-robot execution on hardware — see Flow 6. This runbook is
  sim-only on the Jetson motion side.
- IMU-driven SLAM — the Isaac Sim ROS 2 bridge has no
  `ROS2PublishImu` node, so RTAB-Map runs visual-only for now. See
  [DEFERRED_WORK.md](DEFERRED_WORK.md).

---

## Host topology

| Host | Hostname | LAN IP | Role |
|---|---|---|---|
| DGX Spark | `dgx-spark` | `192.168.50.196` | Isaac Sim + bridge + VLM service + planner service + harness |
| Jetson Orin Nano | `jetson-desktop` | `192.168.50.24` | Executor + Nav2 + RTAB-Map + perception / timestamp fixer + CLI |

Both on the same LAN subnet. DDS middleware is Cyclone DDS
(`rmw_cyclonedds_cpp`); domain ID is `42`. No multicast-blocking
firewalls between the two machines.

---

## Prerequisites checklist

Before starting Stage 1, **all** of the following must be true. If
anything here is red, fix it first — do not paper over it in later
stages.

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
- [ ] [Install validation runbook](VALIDATE_ISAAC_SIM_AND_INFINIGEN.md)
      Phase A + Phase B are green.
- [ ] `strafer_msgs` is built / discoverable from
      `env_phase15` — the sim-in-the-loop harness imports
      `strafer_msgs.action.ExecuteMission` and
      `strafer_msgs.srv.GetMissionStatus`. If that fails, colcon-build
      `strafer_msgs` and add its `install/setup.bash` to the shell.
- [ ] VLM service reachable from the Jetson at
      `http://192.168.50.196:8100/health` — expected `{"status":"ok",
      "model_loaded":true}`. Run it with `make serve-vlm` in
      `.venv_vlm` if it's not already up.

**Jetson only**
- [ ] `colcon build` of `strafer_ros` is current; `install/setup.bash`
      sourced.
- [ ] `source source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env`
      sets `RMW_IMPLEMENTATION`, `ROS_DOMAIN_ID=42`,
      `HARDWARE_PRESENT=false`.
- [ ] `strafer-executor` and `strafer-autonomy-cli` on `PATH` (installed
      via `pip install -e source/strafer_autonomy`).

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
conda activate env_phase15
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
| Discovery fails after a reboot | Multicast blocked on the LAN | Try `export CYCLONEDDS_URI='<CycloneDDS><Domain><General><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'` on both hosts. |
| One host sees its own topics but not the other's | Firewall, hosts on different subnets | `ping` the other host; `ip route` on both; `sudo ufw status`. |
| ROS commands hang with no output | `ros2 daemon` stuck with stale settings from a prior shell | `ros2 daemon stop && ros2 daemon start` on both hosts. |

---

## Stage 2 — DGX bridge alone, manually driven

**Goal**: the DGX publishes simulated D555 + odometry + TF topics; an
operator can drive the simulated robot by manually publishing
`/cmd_vel`. The Jetson plays no role yet.

**Setup**

On the **DGX**:

```bash
cd ~/Workspace/Sim2RealLab
source env_setup.sh
conda activate env_phase15
isaaclab -p source/strafer_lab/scripts/run_sim_in_the_loop.py \
    --mode bridge \
    --headless
```

Expected console output includes, in order:

```
[INFO] ... isaac-sim ... starting
[sim_in_the_loop] bridge graph built at /World/ROS2Bridge
[sim_in_the_loop] chassis_prim=/World/envs/env_0/Robot/strafer/body_link
[sim_in_the_loop] color camera prim=/World/envs/env_0/Robot/strafer/body_link/d555_camera_perception
[sim_in_the_loop] action tensor shape = (1, 3)
```

Isaac Sim's headless mode will not show a GUI window; use
`--no-headless` (or remove `--headless`) if you want the viewport. On
the DGX without a display attached, pass `--headless` and inspect the
stream via ROS topics only.

**Test A — bridge topics are publishing**

Open a second shell on the DGX:

```bash
# DGX shell 2
source env_setup.sh
ros2 topic list | sort
```

Expected (may include extras; these are the must-haves):

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
you run Isaac Sim with the viewport enabled, the robot should visibly
translate forward at ~0.2 m/s.

**Troubleshooting**

| Symptom | Most likely cause | What to check |
|---|---|---|
| `run_sim_in_the_loop.py` crashes on `enable_extension("isaacsim.ros2.bridge")` | Extension not bundled with the Isaac Sim build | Check `ls $ISAACSIM_PATH/exts/isaacsim.ros2.bridge/` — directory must exist. |
| `build_bridge_graph` crashes on a missing camera prim | env not yet fully spawned when graph is built | Inspect the `chassis_prim=...` and `color camera prim=...` lines — both must point at prims that exist in the stage. The env registration change in [strafer_env_cfg.py](../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py) may have moved the path. |
| `/d555/color/image_raw` not in `ros2 topic list` | Bridge extension enabled but graph not evaluated | Check `[sim_in_the_loop] bridge graph built at ...` appeared in the console. If not, graph build silently failed — re-run with full Kit logs. |
| `ros2 topic hz` returns 0 Hz | Isaac Sim is stuck at iteration 0 | The camera render is not executing. Try launching without `--headless` once to confirm the viewport shows a rendered frame. |
| `/strafer/odom` pose doesn't advance under `/cmd_vel` | Bridge's `SubscribeTwist` node not wired or env action layout mismatch | Read the action-tensor shape line — must be `(1, 3)`. If it's `(1, 4)` the env config has changed and the launcher's injection map in `_run_bridge_mode` needs updating. |

---

## Stage 3 — Jetson consumes the bridge's sensor topics

**Goal**: the Jetson brings up its perception / SLAM / Nav2 stack
against the DGX-published topics, produces the
`*_sync` re-stamped variants and `/scan`, and runs RTAB-Map without
the real hardware drivers.

**Setup**

Keep the DGX bridge running from Stage 2 (`--mode bridge`).

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
source /opt/ros/<distro>/setup.bash
source ~/workspaces/Sim2RealLab/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42

ros2 topic list | sort
```

Expected to include the bridge-published raw topics plus the
Jetson-synthesized derived topics:

```
/cmd_vel
/d555/aligned_depth_to_color/image_sync        ← produced by timestamp_fixer
/d555/color/camera_info
/d555/color/image_raw
/d555/color/image_sync                          ← produced by timestamp_fixer
/d555/depth/camera_info
/d555/depth/image_rect_raw
/rtabmap/map
/scan                                           ← produced by depthimage_to_laserscan
/strafer/odom
/tf
```

Verify the TF tree is complete:

```bash
# Jetson shell 2
ros2 run tf2_tools view_frames
```

Expected chain: `map` → `odom` (from RTAB-Map) → `base_link` (from the
bridge) → `d555_link` (from the bridge) → `d555_color_optical_frame`
(from `strafer_description`).

**Test B — `/scan` has content**

```bash
# Jetson shell 2
ros2 topic hz /scan
ros2 topic echo /scan --once | head -30
```

Expected: ~15 Hz (the `depthimage_to_laserscan` default), with a
`ranges[]` array that contains both `inf` and finite distances.

**Go/no-go**: RTAB-Map's console shows `Adding node ...` entries,
meaning it's ingesting the simulated camera frames and building a map.
`/rtabmap/map` in `ros2 topic list`, with a non-empty `data[]` when
echoed.

**Troubleshooting**

| Symptom | Most likely cause | What to check |
|---|---|---|
| Bridge topics missing on the Jetson, present on the DGX | DDS discovery fell over after the Jetson joined | Repeat Stage 1's discovery check from the Jetson side. |
| `/d555/color/image_sync` missing, raw topic present | `timestamp_fixer` node not running | Check the launch log for `timestamp_fixer` errors; re-launch with `output='screen'`. |
| `/scan` missing | `depthimage_to_laserscan` is part of `strafer_slam` — did it start? | Inspect the `slam.launch.py` include chain; common cause is a bad `frame_id` that makes the node reject the depth image. |
| RTAB-Map not accepting frames | Frame-id mismatch or TF race | `ros2 topic echo /d555/color/image_raw --once` — header.frame_id must be `d555_color_optical_frame`. TF from `base_link → d555_link` must exist before frames start arriving. |
| TF tree has `base_link → d555_link` missing | `strafer_description` (URDF) did not start | Confirm the launch includes `description.launch.py`; it publishes the URDF transform via `robot_state_publisher`. |
| Nav2 complains about `/map` or `/odom` not available | Start-ordering race — RTAB-Map isn't ready yet | Wait longer (20-30 s) after launch; if Nav2 refuses to finish its startup, check `/tf` chain. |

---

## Stage 4 — Manual mission submission end-to-end

**Goal**: an operator submits a natural-language mission via the CLI
on the Jetson. The executor talks to the DGX planner + VLM over HTTP,
builds a skill sequence, runs it against the simulated sensors, and
the simulated robot visibly moves in Isaac Sim.

**Setup**

Keep the DGX bridge running (`--mode bridge`) and the Jetson bringup
running from Stage 3. The Jetson autonomy executor must be up
(confirm with `ros2 action list` — `/execute_mission` should appear).

**Test**

Open a third shell on the Jetson and submit a mission:

```bash
# Jetson shell 3
source /opt/ros/<distro>/setup.bash
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

While the mission runs, watch `/cmd_vel` on the DGX:

```bash
# DGX shell 2
ros2 topic echo /cmd_vel --once
```

The Twist should come from `nav2` (identified by its QoS profile if
you care — most of the time it's enough that a non-zero Twist arrives).

**Go/no-go**: the simulated robot visibly translates / rotates in the
Isaac Sim viewport (or its `/strafer/odom` position advances in step
with the mission), and `strafer-autonomy-cli` prints
`final_state: succeeded`. A `final_state: failed` with a sensible
`message` (e.g. "goal outside costmap") is also OK at this stage — it
means the control path works even if the specific command was
infeasible.

**Test variant — semantic mission**

```bash
# Jetson shell 3
strafer-autonomy-cli submit "go to the door"
```

This exercises the VLM grounding path: the planner inserts a
`scan_for_target` skill, the executor calls the VLM's `/ground`
endpoint, and when a match is found the goal-projection node turns the
bbox into a map-frame pose for Nav2. Expected to succeed if the scene
has a door the VLM can see at mission start.

**Troubleshooting**

| Symptom | Most likely cause | What to check |
|---|---|---|
| `strafer-autonomy-cli submit` hangs at "Action server not available" | Executor not running or on a different domain | `ros2 node list` should include `/strafer_autonomy_command_server`; `ros2 action list` should include `/execute_mission`. |
| Executor starts then immediately errors on health check | VLM or planner service unreachable from the Jetson | From the Jetson, `curl http://192.168.50.196:8100/health`. If that fails, SSH isn't the issue; LAN routing is. |
| Mission sticks in `state=planning` | Planner timeout or parse error | Check the Jetson's `HttpPlannerClient` logs and the DGX planner's stdout. Common cause: Qwen3-4B returning malformed JSON. Retry the mission with simpler wording. |
| `scan_for_target` always returns `not_found` | VLM's frame sampling is wrong, or grounding threshold too strict | Inspect the `/describe` output from the VLM on a known frame — does it mention the target at all? |
| Nav2 plans but `/cmd_vel` never fires | Controller crashed or costmap empty | Inspect Nav2 logs; ensure `/scan` was healthy at the time Nav2 initialised its costmap. |
| Robot moves the wrong direction | Mecanum wheel-axis signs mismatched between sim and real (should be identical by contract) | Compare `strafer_shared.constants.WHEEL_AXIS_SIGNS` (sim) with the real robot's RoboClaw wiring — a mismatch here is a contract break, not a sim-in-the-loop bug. |

---

## Stage 5 — Harness mode (autonomous mission sweep)

**Goal**: the DGX harness walks `scene_metadata.json`'s object list,
submits one mission per target, captures reachability-labelled frames
into `frames.jsonl`. No human driver.

**Prerequisites**
- Stages 1-4 green.
- One generated Infinigen scene under
  `Assets/generated/scenes/<scene_name>/` containing
  `scene.usdc` + `scene_metadata.json`. The validation runbook's
  Phase C produces a minimal smoke scene; a real Infinigen scene with
  furniture is produced by running `generate_indoors` with
  `fast_solve.gin singleroom.gin`.

**Test**

Stop the DGX `--mode bridge` run from Stage 2. On the DGX:

```bash
cd ~/Workspace/Sim2RealLab
source env_setup.sh
conda activate env_phase15
isaaclab -p source/strafer_lab/scripts/run_sim_in_the_loop.py \
    --mode harness \
    --scene-metadata Assets/generated/scenes/<scene_name>/scene_metadata.json \
    --scene-usd      Assets/generated/scenes/<scene_name>/scene.usdc \
    --output         data/sim_in_the_loop/<scene_name> \
    --max-missions   3 \
    --headless
```

Keep the Jetson bringup from Stage 3 running.

**Go/no-go**: each mission in `--max-missions 3` gets logged by the
harness with `reachable=True|False`. The output directory now contains
three `episode_NNNN/` subdirs, each with a populated `frames.jsonl`
and `frame_*.jpg` files.

```bash
# DGX
ls data/sim_in_the_loop/<scene_name>/
head -n 2 data/sim_in_the_loop/<scene_name>/episode_0000/frames.jsonl
```

Expected: `frame_id`, `image_path`, `robot_pos`, `robot_quat`,
`bboxes`, `mission_id`, `target_label`, `target_position_3d`,
`reachability`, `mission_state`.

**Troubleshooting**

| Symptom | Most likely cause | What to check |
|---|---|---|
| `Ros2MissionApi` fails to open at harness start | `strafer_msgs` not on DGX `PYTHONPATH` | Confirm `python -c "from strafer_msgs.action import ExecuteMission"` works in `env_phase15`. Re-source the colcon install. |
| `Action server ... did not become available` | Jetson executor not up or on wrong domain | Same as Stage 4: verify `/execute_mission` action exists from the DGX via `ros2 action list`. |
| Every mission ends `reachable=False` at timeout | Nav2 never finishes plans against the current scene | Start from Stage 4 — manually submit a mission and verify it works before letting the harness submit dozens. |
| Mission generator yields zero missions | `scene_metadata.json` has no parseable objects | `python -c "import json; print(len(json.load(open('.../scene_metadata.json'))['objects']))"`. If zero, re-run `extract_scene_metadata.py --from-usd`. |
| `frames.jsonl` exists but lacks `reachability` field | Outcome was never captured — harness crashed mid-mission | Search the DGX stdout for `[mission_id] harness crashed mid-mission`; the stack trace will point at the break. |

---

## End-state artifacts

On successful completion of all five stages:

- DGX: Isaac Sim + bridge running; one or more
  `data/sim_in_the_loop/<scene_name>/episode_*/` dataset directories
  (from Stage 5).
- Jetson: bringup still running; RTAB-Map database at
  `~/.ros/rtabmap.db` (from Stage 3).
- Confirmed that the robot's motion in Isaac Sim is being driven by
  Nav2's `/cmd_vel` on the Jetson (Stage 4).
- A datapoint per mission with a reachability label (Stage 5).

**Shut-down order**: Stop the harness run (or Stage 2 bridge run) on
the DGX first — that stops the simulator and the bridge topics. Then
stop the Jetson bringup. RTAB-Map persists its database on graceful
shutdown.

---

## Known caveats

- **IMU data is absent.** The bridge has no `ROS2PublishImu` node.
  RTAB-Map runs visual-only in this test. If SLAM quality proves too
  poor in a particular scene, this is the first thing to revisit. See
  [DEFERRED_WORK.md](DEFERRED_WORK.md) for the deferred item.
- **Scene swap requires re-launching.** The harness is single-scene per
  launch. `IsaacLabEnvAdapter.reset(scene_name=...)` rejects
  cross-scene calls on purpose — swap scenes by re-running the DGX
  script with a different `--scene-usd`.
- **`goal_position_noise` is not the default.** Pre-deployment RL
  policies were trained without goal noise; VLM-grounded goals carry
  ±0.2-0.5 m localization error. Expect some goal-pose oscillation
  near the target. See [DEFERRED_WORK.md](DEFERRED_WORK.md).
- **Electronics masses are missing from the USD.** The chassis
  inertia underestimates the real robot. Fine control behaves
  differently in sim vs. real until this is fixed. See
  [DEFERRED_WORK.md](DEFERRED_WORK.md).
