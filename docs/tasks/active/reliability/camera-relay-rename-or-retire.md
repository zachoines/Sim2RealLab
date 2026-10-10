# Decide whether the camera relay stays, and name it for what it does

**Type:** refactor
**Owner:** Jetson
**Priority:** P3 — the relay works. Its name describes a re-stamping mode it no longer has, and
whether its consumers need a copy of the camera topics at all has not been measured.
**Estimate:** S to keep and rename; M to retire (every consumer on every lane, plus a link
measurement)
**Branch:** `task/camera-relay-rename-or-retire`

## Story

As **a reader of the perception and SLAM launches**, I want **the node that republishes the
camera topics either named for what it does or removed**, so that **nobody looks for re-stamping
in a node called `timestamp_fixer` that only copies messages.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

`timestamp_fixer` (`strafer_perception`) subscribes to the D555's colour image, aligned depth and
both camera_info topics, and republishes each message unchanged as `/d555/**_sync`. Its
re-stamping mode, which gave it its name, was removed in
[#236](https://github.com/zachoines/Sim2RealLab/pull/236). Re-stamping gave a frame's image and its
camera_info different stamps and broke `depth_to_pointcloud`'s exact sync, and every launch had
already pinned it off.

**Who reads the `*_sync` topics:**

| consumer | topics | where |
|---|---|---|
| RTAB-Map | colour, aligned depth, colour camera_info | `strafer_slam/launch/slam.launch.py` remaps |
| `depth_to_pointcloud` | aligned depth and its camera_info | the same launch |
| `goal_projection_node` | aligned depth, colour camera_info | `strafer_perception/goal_projection_node.py` |
| the executor's `JetsonRosClient` | colour, aligned depth, colour camera_info | `strafer_autonomy/clients/ros_client.py` |
| the Foxglove layout | colour and its camera_info (its depth panel reads `/d555/depth/image_rect_raw` directly) | `strafer_bringup/foxglove/strafer_layout.json` |
| `ros_test_slam.py` (manual check) | colour, aligned depth | `source/strafer_ros/` |

**Where it runs.** Three launches start it: `perception.launch.py` on the real lane,
`sim_bridge_support.launch.py` on the sim-bridge lane and `bringup_sim_in_the_loop.launch.py` on
the sim-in-the-loop lane. On the two sim lanes it also renames. The bridge publishes depth as
`/d555/depth/image_rect_raw` and `/d555/depth/camera_info`, and the relay's remaps publish them
under the aligned names. Retiring the relay there needs the bridge to publish the aligned names,
or a remap on each consumer for each lane.

**What the copy buys on the sim-bridge lane.**
- Colour and depth reach the robot over the cross-host link.
- DDS delivers one unicast copy per subscribing process. With 1, 2 and 3 subscriber processes the
  sim host sent 67.5, 102.0 and 127.6 Mbit/s
  ([`depth-receiver-host-capacity`](../../completed/depth-receiver-host-capacity.md)). That was
  over a WiFi uplink that saturated near 60 Mbit/s, so congestion compresses the rise per process.
  Uncongested, each extra depth copy costs about 221 × RTF Mbit/s
  ([`depth-subscriber-consolidation`](depth-subscriber-consolidation.md)).
- The relay's one cross-link subscription to colour and depth serves every consumer in the table.
  The exception is the Foxglove depth panel, which takes its own copy while a client displays it.
- The inference node also takes its own copy of depth, and `depth-subscriber-consolidation` plans
  to move it onto a local republication.
- Pointing the consumers at the raw topics would add one remote copy per consumer process on this
  lane.

On the real lane every consumer runs on the robot, and the relay's cost is one extra local
publish of every colour and depth frame.

## Acceptance criteria

- [ ] The brief records the decision, to keep or retire the relay, with its evidence:
  - on the sim-bridge lane, the link bytes (sim-host TX and robot RX counters) with the relay and
    with the consumers on the raw topics, or a design that keeps one remote subscriber per stream
    without the relay;
  - on the real lane, the relay's CPU share.
- [ ] **If kept,** the executable, node name and module are `camera_relay`: in `setup.py`, the
      three launches, the docs that name it, and the comments that call the `*_sync` topics
      "timestamp-fixed" (`slam.launch.py`, `pointcloud_to_laserscan.yaml`, `ros_client.py`). The
      `*_sync` topic names do not change, so no consumer changes.
- [ ] **If retired,** every consumer in the table reads the driver's topics on every lane, and the
      sim lanes publish or remap to the aligned names. On the sim-bridge lane, sim-host TX does not
      rise against the before, and `depth-subscriber-consolidation`'s plan is updated to match.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression: `make test-ros` passes; on the sim-bridge lane RTAB-Map and
      `depth_to_pointcloud` publish with no sync warnings, and `goal_projection_node` answers a
      projection.

## Out of scope

- Stamps. The relay passes them through, and whatever replaces it must too.
- The relay's QoS.
- The inference node's own depth subscription, which is `depth-subscriber-consolidation`.
- Renaming the `*_sync` topics.
