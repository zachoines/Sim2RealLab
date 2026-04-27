# Jetson-side visual debugger over SSH (no monitor)

**Type:** task / new feature
**Owner:** Jetson agent
**Priority:** P1
**Estimate:** S–M (couple of days incl. SSH-tunnel docs and a Foxglove
layout file; smaller if going with the lightweight MJPEG path)

## Story

As a **bridge / real-robot operator who SSH'es into the Jetson without
a local monitor or keyboard**, I want **a single command on the Jetson
that exposes a live visualizer of the robot's camera + detection
state**, so that **I can run sim missions headless on the DGX (`make
sim-bridge` instead of `make sim-bridge-gui`) and still debug what the
robot perceives during a mission, without paying the ~80 ms/loop cost
of the Isaac Sim editor viewport on the DGX side**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](context/bridge-runtime-invariants.md)

## Context

The headless-vs-`--viz kit` cost split is summarized in
[`context/bridge-runtime-invariants.md`](context/bridge-runtime-invariants.md#headless-vs---viz-kit-defaults)
and detailed in `docs/PERF_INVESTIGATION_SIM_IN_THE_LOOP.md` Findings
8-10. Switching to headless mode (`make sim-bridge`) recovers ~85 ms
per loop, lifting bridge throughput from ~5 Hz to ~8.5 Hz, and lifts
further once async camera publishers land (see
[async-camera-publishers task](async-camera-publishers.md)).

The blocker for using headless as the default bridge mode: the
operator currently relies on the editor viewport to confirm scene
state, robot motion, and what the camera is seeing during a mission.
Headless mode kills that visibility on the DGX side, but **all the
useful information (RGB, depth, robot pose, VLM detections, /tf) is
already published as ROS 2 topics that the Jetson side has access
to**. Surfacing them in a headless-friendly visualizer on the Jetson
side closes the gap.

The Jetson is accessed exclusively via SSH; there is no local
monitor, keyboard, or mouse. The visualizer must therefore work
remotely — typically by exposing a TCP/HTTP port that's tunneled
back to the operator's workstation and viewed in a browser or
desktop client there.

## Recommended approach: Foxglove Studio + foxglove_bridge

[`foxglove_bridge`](https://docs.foxglove.dev/docs/connecting-to-data/ros-foxglove-bridge/)
is a ROS 2 node maintained by the Foxglove team that exposes a
WebSocket on a configurable port (default 8765). The operator opens
[Foxglove Studio](https://foxglove.dev/) on their workstation
(browser or native app) and connects to the WebSocket via an SSH
tunnel. Foxglove Studio has built-in panels for:

- Image topics (RGB, with bbox / detection annotation overlays).
- Depth images with selectable colormaps (jet, viridis, etc.).
- TF tree.
- Point clouds (via `depth_image_proc`'s point cloud, if useful).
- 3D scene visualization with TF + URDF + camera frustum.
- Time-aligned playback if you also record a bag.

This is the production path. Minimal custom code on the Jetson —
just adding `foxglove_bridge` to a launch file.

Operator workflow:

```bash
# DGX side
make sim-bridge   # headless

# Jetson side
ros2 launch strafer_bringup bringup_sim_in_the_loop.launch.py …
# (foxglove_bridge launched as part of the bringup, port 8765 exposed)

# Operator workstation
ssh -L 8765:localhost:8765 jetson
# Open Foxglove Studio, connect to ws://localhost:8765
```

A Foxglove "layout" JSON file checked into the repo (e.g.
`source/strafer_ros/strafer_bringup/foxglove/strafer_layout.json`)
pre-configures the panels so anyone who connects sees a sensible
default arrangement: RGB on the left, depth top-right, TF tree
bottom-right, current goal and odom in a status panel.

## Lighter alternative: custom MJPEG image viewer

If `foxglove_bridge` proves too heavyweight (extra dep on the Jetson
or operator UX friction), a small Python ROS 2 node can subscribe to
the topics, draw VLM bbox overlays with cv2, and serve the result as
an MJPEG HTTP stream. Operator views in any browser at
`http://localhost:8080/stream` after SSH-tunneling 8080. ~80 lines of
code, zero new system dependencies beyond `cv2` and `flask`/`http.server`.

Trade-offs:

- Lighter / simpler to maintain.
- Image-only — no TF tree, no point cloud, no depth colormap (would
  need additional plumbing per channel).
- Best when the Foxglove path turns out to be impractical for the
  Jetson's storage / dependency constraints.

## Acceptance criteria

- [ ] Adding the visualizer is a single-line addition to
      `bringup_sim_in_the_loop.launch.py` (plus optional disable flag).
- [ ] One published port (default 8765 for Foxglove, 8080 for MJPEG)
      with the SSH-tunnel command documented in the launch's docstring
      or in `docs/INTEGRATION_SIM_IN_THE_LOOP.md`.
- [ ] At minimum: live `/d555/color/image_raw` viewable from the
      operator's workstation.
- [ ] Better: VLM detection bboxes overlaid on the RGB feed (subscribe
      to whichever topic carries grounding results — verify by reading
      `strafer_autonomy.clients.ros_client` for the publish path).
- [ ] Nice-to-have: live `/d555/depth/image_rect_raw` with a colormap
      so the operator can sanity-check depth output.
- [ ] Works with both `make sim-bridge` (sim/headless) AND on the
      real robot (since the visualizer subscribes to the same topic
      names regardless of source).
- [ ] Visualizer launches as part of `bringup_sim_in_the_loop.launch.py`
      by default; a `viewer:=false` arg disables it for non-debug
      missions.
- [ ] No measurable impact on the rest of the Jetson stack — the
      visualizer's CPU should be < 10 % at idle, < 30 % when streaming.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- Existing bringup: `source/strafer_ros/strafer_bringup/launch/bringup_sim_in_the_loop.launch.py`
  is where the new node gets added. Pattern after the existing
  RTAB-Map / Nav2 / executor launches.
- VLM detection topic: search the Jetson side for the publish path
  (`strafer_autonomy.clients.ros_client` is a likely starting point).
  `vision_msgs/Detection2DArray` is the standard ROS 2 message; if
  we're not using that today, this is the natural place to start.
- Foxglove install on Jetson: `apt install ros-humble-foxglove-bridge`
  (Humble repository).
- Real-robot parity: when the real D555 driver runs, it publishes the
  same `/d555/color/image_raw` and `/d555/depth/image_rect_raw` topics.
  The visualizer should not care about source.

## Out of scope

- Recording bags for offline replay (already possible via `ros2 bag
  record` outside this task).
- A web UI that submits missions / commands / sends Nav2 goals — this
  task is read-only visualization. Mission control stays on the
  `strafer-autonomy-cli`.
- DGX-side viewer; this task lives entirely on the Jetson.
