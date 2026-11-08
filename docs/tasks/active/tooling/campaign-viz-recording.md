# Record every lane or robot campaign for Foxglove, from the tree

**Type:** task (a recording that a lane campaign ran as a deposited ride-along becomes a tracked launch, a Makefile
target and a tracked layout)
**Owner:** Jetson (the recorder, the launch and the image), with DGX for the offline collage and the layout
**Priority:** P2. The maintainer asked to follow a campaign remotely in Foxglove (the map, the goal and the robot in one
layout). The inside-goal campaign recorded every boot that way, from a deposited script.
**Estimate:** S–M
**Branch:** `task/campaign-viz-recording`

## Story

As **the maintainer following a lane or robot campaign from elsewhere**, I want **every boot recorded in a format
Foxglove opens, with a layout that shows the map, the goal and the robot together**, so that **a campaign can be watched
afterwards, or followed, without rebuilding its tools each time.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`lane-inside-goals-2026-11-02`](../../../measurements/lane-inside-goals-2026-11-02/README.md) §9, and its deposit's
  `session/fh_up.sh` (the recorder, `viz_start` and `viz_stop`), `campaign/viz_feasibility.txt`,
  `campaign/viz_qos_check.txt`, and `viz/` (the offline collage).

## Context

Read at `f16d18d6`.
- **The image has no MCAP storage plugin.** `Dockerfile.cpu` builds on `ros:humble-ros-base` with `rosdep` over the
  package manifests. No manifest depends on `rosbag2_storage_mcap`, so `ros2 bag record -s mcap` is refused and
  sqlite3 is the only storage (rosbag2 0.15.16). Foxglove opens both. Its bundled definitions decode the standard
  message packages; custom types such as `rtabmap_msgs` do not decode there.
- **What the campaign recorded,** inside `strafer_navigation`, beside the two ride-alongs:
  - **Topics:** `/rtabmap/map`, `/tf`, `/tf_static`, `/robot_description`, `/strafer_inference/active_goal`,
    `/strafer/subgoal`, `/cmd_vel`, and the `nav_msgs/msg/Odometry` topics in the graph (`/odom`, `/strafer/odom`). It
    recorded eight: its log shows no subscription to `/odom`.
  - **QoS:** transient-local for `/robot_description` and `/tf_static`.
  - **The stop:** by its recorded PID with SIGINT, before rtabmap.
  - **Size and cost:** 26.3–59.0 MB per completed segment (the first boot's 81.5 MB includes the hours after a sim-host
    freeze). The bring-up's load averaged 9.04–10.50 against 9.07–9.78 in boots without it, and the container drew
    137–178 % CPU against 139–167 %; nothing tests a difference.
- **Live viewing exists.** `foxglove_bridge` is in the image, and the compose `viewer` profile (`strafer_viewer`,
  ws :8765, `make viewer`) attaches it to a running stack. From elsewhere it needs a tunnel to the robot host.
- **A layout is tracked:** `source/strafer_ros/strafer_bringup/foxglove/strafer_layout.json`, documented in `Readme.md`.
  Its topics are the navigation stack's (`/rtabmap/map`, `/tf`, `/plan`, the costmaps, `/scan`); it does not show
  `/strafer_inference/active_goal`.
- **The offline collage** (`viz/` in that deposit) rewrites each bag as an MCAP with a derived scene
  of the session's goals and events. It is written with the Foxglove SDK on the DGX, from the deposit and the bags kept
  on the sim host by digest.

## Decisions (the maintainer's)

| # | Decision | Options |
|---|---|---|
| 1 | The topic set | the campaign's eight; fewer; more (images and point clouds were left out for size) |
| 2 | The storage plugin | add `ros-humble-rosbag2-storage-mcap` to `Dockerfile.cpu` (a rebuild, on the Jetson's side); stay on sqlite3 |
| 3 | A derived scene in the tree | the offline collage as a tracked tool; keep it per campaign, in deposits |
| 4 | The layout | extend the tracked `strafer_layout.json` with the active goal and a campaign scene; add a campaign layout beside it; a how-to in the deploy README |

## Acceptance criteria

- [ ] Decisions 1–4 recorded, each with its date.
- [ ] **A bringup launch** for the recorder, with decision 1's topics and the two transient-local overrides, started and
      stopped with the stack, and never a stop condition for it.
- [ ] **A Makefile target** that brings it up beside a running stack and reports where the bag is.
- [ ] If decision 2 adds the plugin, the images rebuild, and `ros2 bag record -s mcap` is accepted.
- [ ] If decision 4 tracks a layout, it opens in Foxglove against one recorded bag, the 3D panel showing the map, TF,
  the
      robot model and the active goal.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).
- [ ] No regression: `make test-ros` passes.

## Investigation pointers

- `source/strafer_ros/deploy/docker/Dockerfile.cpu`; `source/strafer_ros/strafer_bringup/` (launch files; the
  `foxglove_bridge` exec_depend); the compose `viewer` profile in `source/strafer_ros/deploy/docker-compose.yml`.
- The deposit's `session/fh_up.sh`: `viz_start`, `viz_stop` and `VIZ_QOS`.

## Out of scope

- **Any read that depends on a recording.** The recording is for viewing.
- **Live remote viewing over the internet.** `make viewer` and a tunnel already cover a trusted network.
