# Publish VLM detections as a ROS topic + overlay them on the Foxglove RGB feed

**Type:** task / new feature
**Owner:** Jetson agent
**Priority:** P2
**Estimate:** S (a few hours; one publisher in the executor + a Foxglove
layout tweak + a smoke test)

## Story

As a **mission operator watching a sim-in-the-loop or real-robot run
in Foxglove Studio**, I want **the VLM's grounding bboxes drawn on
top of the live `/d555/color/image_raw` feed**, so that **I can see
what the planner is actually grounding to during a `scan_for_target`
or `go_to_target` mission — instead of inferring it from log lines
or trusting that the projection service got the right pixel
coordinates**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)

## Context

The headless visualizer
([`completed/jetson-headless-viewer.md`](completed/jetson-headless-viewer.md),
shipped in `179275e`) brought the live RGB / depth / TF / map state
to Foxglove Studio over an SSH-tunneled WebSocket. It deliberately
deferred one acceptance bullet — bbox overlay on the RGB feed — because
**no detection topic exists today**. The executor's grounding flow is
HTTP-only:

1. `MissionRunner` (in `strafer_autonomy.executor.mission_runner`)
   captures a `SceneObservation` from `JetsonRosClient`.
2. The observation's RGB image goes to
   `HttpGroundingClient.ground()` over LAN HTTP to the DGX VLM
   service (`source/strafer_vlm/strafer_vlm/service/app.py`).
3. The parsed grounding result (label + bbox) is handed to
   `goal_projection_node` via the `ProjectDetectionToGoalPose`
   service to compute a Nav2 goal.

The bbox never lives on a ROS topic, so nothing is available for
Foxglove (or RViz, or `ros2 bag record`) to subscribe to. Adding a
publisher in the executor — stamped with the source image's
`header.stamp` — gives every downstream consumer (visualizer, bag
replay, future analytics) the same bbox stream.

`vision_msgs/Detection2DArray` is the standard message and Foxglove
Studio's Image panel has built-in support for overlaying it on a
matching image topic, with no custom annotation work required.

## Approach

1. **Publisher in `JetsonRosClient`.** Add a `Detection2DArray`
   publisher on a topic alongside the camera (e.g.
   `/d555/color/detections`). Expose a `publish_detections(...)`
   method that takes a list of bboxes + labels + the source image's
   stamp, builds the message, and publishes.
2. **Wire into `MissionRunner`.** Whenever `scan_for_target` or any
   skill that performs a grounding call gets a result back from
   `grounding_client.ground()`, call `ros_client.publish_detections(
   bboxes, labels, image_stamp)` before returning the skill outcome.
   Publish even on empty results (an empty `Detection2DArray` clears
   the previous overlay in Foxglove instead of leaving a stale box).
3. **Foxglove layout update.** Edit
   [`source/strafer_ros/strafer_bringup/foxglove/strafer_layout.json`](../../source/strafer_ros/strafer_bringup/foxglove/strafer_layout.json)
   so the RGB Image panel subscribes to the new detections topic as
   an annotation source. The schema field is roughly
   `imageMode.annotations: { "/d555/color/detections": { visible: true } }`
   — verify against Foxglove's current layout schema before
   committing.
4. **Smoke test.** A unit-test mock or a quick manual check that
   sending a known bbox through the publisher produces a
   syntactically-correct `Detection2DArray` whose `header.stamp`
   matches the image stamp.

## Acceptance criteria

- [ ] `JetsonRosClient` publishes `vision_msgs/Detection2DArray`
      on a documented topic (`/d555/color/detections` unless there's
      a reason to deviate). Topic name + message type added to
      [`source/strafer_ros/README.md`](../../source/strafer_ros/README.md)'s
      "Topics published / consumed" table.
- [ ] Each published message's `header.stamp` matches the source
      image's stamp (NOT `node.now()`), so Foxglove can pair the
      bbox with the right RGB frame.
- [ ] Empty grounding results publish an empty `Detection2DArray`
      (clears stale overlays).
- [ ] `strafer_layout.json` enables the annotation overlay on the
      RGB panel by default. Operators who pull the latest layout see
      bboxes without manual UI tweaking.
- [ ] Works in both `make sim-bridge` (sim) AND on the real robot
      (the executor is the same code; only the upstream image source
      differs).
- [ ] No measurable impact on mission throughput. Grounding-call
      cadence is unchanged; this just adds one ROS publish per call.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.
- [ ] Smoke test: manual mission `strafer-autonomy-cli submit "go to
      the door"` end-to-end through `scan_for_target`. The Foxglove
      RGB panel shows a bbox on the door (or the last-grounded
      object) at the moment the executor logs the grounding result.

## Investigation pointers

- VLM client: [`source/strafer_autonomy/strafer_autonomy/clients/vlm_client.py`](../../source/strafer_autonomy/strafer_autonomy/clients/vlm_client.py)
  — `HttpGroundingClient.ground()` returns parsed detections.
- ROS client (where the publisher lives):
  [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
  — see the existing `_setup_subscriptions` for the publisher /
  subscriber pattern. Add a `_detections_pub` alongside.
- Where to call the publisher:
  [`source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py`](../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
  (or wherever `scan_for_target` resolves the grounding response —
  trace from `command_server.py:294`'s `MissionRunner` import).
- VLM payload types:
  [`source/strafer_vlm/strafer_vlm/service/payloads.py`](../../source/strafer_vlm/strafer_vlm/service/payloads.py)
  — `confidence` and `bbox_2d` fields are what need to land in the
  `Detection2D.results[].hypothesis` and `Detection2D.bbox` fields.
- Foxglove Image annotation schema reference:
  <https://docs.foxglove.dev/docs/visualization/panels/image/#annotations>.
  Note that Foxglove also defines its own `foxglove_msgs/ImageAnnotations`
  — prefer `vision_msgs/Detection2DArray` for portability (RViz,
  bag replay tooling, future ROS consumers).
- Bbox coordinate frame: VLM service returns Qwen-normalized
  `[0, 1000]` coords (see [`goal_projection_node.py`](../../source/strafer_ros/strafer_perception/strafer_perception/goal_projection_node.py)
  for the conversion). The ROS detection message must carry **pixel
  coordinates** matching the published image's resolution, so do
  the rescale before publishing.

## Out of scope

- VLM detection on every camera frame. Grounding only fires during
  specific skills (`scan_for_target`, `verify_arrival`); publishing
  detections at that cadence is intentional and sufficient. A
  continuous-tracker is a separate, much larger project.
- Real-time bbox tracking across frames (Kalman / SORT / etc.).
  Each grounding call is a one-shot detection on a single frame; no
  inter-frame association.
- Editing the Foxglove layout schema beyond the annotation toggle.
  Other panels (depth, 3D scene, raw odom) stay as-is.
- Replacing the `ProjectDetectionToGoalPose` service. The existing
  bbox → 3D goal flow is independent and stays unchanged; this
  task only adds a parallel publish for visualization.
