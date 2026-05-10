# Extract grounding/Foxglove publishers from `JetsonRosClient`

**Type:** task / refactor
**Owner:** Jetson agent
**Priority:** P2
**Estimate:** S (a few hours; pure structural move + test relocation + smoke test)
**Branch:** task/grounding-publisher-extraction

## Story

As a **developer maintaining the executor's ROS adapter**, I want **the
Foxglove / vision-overlay publishers split out of `JetsonRosClient` into
a dedicated collaborator**, so that **the mission-critical path
(sensor caching, Nav2, projection, rotation) is easier to read, the
optional viz deps (`vision_msgs`, `foxglove_msgs`, `cv_bridge`) stop
leaking into the core client, and future debug topics have a clear
home instead of piling onto an already ~1000-line class.**

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)

## Context

`vlm-bbox-overlay` ([`completed/vlm-bbox-overlay.md`](../completed/vlm-bbox-overlay.md),
PR [#22](https://github.com/zachoines/Sim2RealLab/pull/22)) added three
publishers to `JetsonRosClient` to drive the Foxglove bbox overlay:

1. `/d555/color/detections` — `vision_msgs/Detection2DArray` (canonical
   semantic output; portable to RViz, bag replay, future ROS consumers).
2. `/d555/color/detections_fg` — `foxglove_msgs/ImageAnnotations`
   (Foxglove-native render path; Studio 2.x doesn't auto-decode
   `Detection2DArray` for image overlays).
3. `/d555/color/grounding_frame` — `sensor_msgs/Image` (the exact RGB
   frame the VLM grounded against, latched `TRANSIENT_LOCAL` so
   Foxglove can render a stable image+bbox pair that doesn't drift as
   the robot moves).

Each publisher has graceful-degradation logic for a missing optional
dep, a custom QoS profile (latched), and ~200 lines of test
infrastructure that stubs `vision_msgs` / `foxglove_msgs` /
`cv_bridge` / `builtin_interfaces` in `sys.modules`. None of this is
load-bearing for missions — it's additive observability.

The smell that motivated this brief: `mission_runner._publish_grounding_overlay`
already accesses these methods via `getattr(self._ros_client,
"publish_detections", None)` and `getattr(..., "publish_grounding_frame",
None)` — i.e., the caller is duck-typing what should be a real
contract because the methods aren't part of the `RosClient` Protocol.

`JetsonRosClient` now mixes three orthogonal concerns:

| Concern | Lines (rough) | Optional deps |
|---|---|---|
| Sensor caching + TF + costmap + observation capture | ~250 | `cv_bridge`, `tf2_ros` |
| Skills (Nav2 client, projection service, rotate_in_place) | ~350 | none beyond core ROS |
| Viz publishers (Detection2DArray, ImageAnnotations, Image) | ~250 | `vision_msgs`, `foxglove_msgs` |

The third row is the cleanest seam to extract — it's the newest, the
least entangled with the rest, and the one with the heaviest
test-stub overhead.

## Approach

1. **New file `source/strafer_autonomy/strafer_autonomy/clients/grounding_publisher.py`** containing:
   - `GroundingPublisher` Protocol (typed: `publish_detections(...)`,
     `publish_grounding_frame(...)`). Mirrors the
     `RosClient` / `GroundingClient` Protocol style already in the
     codebase — no ABC, no inheritance.
   - `FoxgloveGroundingPublisher` class — owns the three topics, the
     latched QoS, and the optional-dep ImportError handling for
     `vision_msgs` / `foxglove_msgs` / `cv_bridge`. Constructor takes a
     `rclpy.node.Node` (no separate ROS executor — re-uses the
     `JetsonRosClient` node).
   - `NullGroundingPublisher` class — no-op implementation for tests
     and for the "viz explicitly disabled" config path.
   - Constants `TOPIC_DETECTIONS`, `TOPIC_DETECTIONS_FG`,
     `TOPIC_GROUNDING_FRAME` move here (currently on
     `JetsonRosClient`).

2. **`JetsonRosClient` shrinks.** `_setup_publishers()` collapses to a
   single line constructing `FoxgloveGroundingPublisher(self._node)`
   and assigning to `self._grounding_publisher: GroundingPublisher`.
   `publish_detections()`, `_publish_foxglove_annotations()`,
   `publish_grounding_frame()` move out. The three TOPIC constants
   move out. Net: ~250 fewer lines in `ros_client.py`.

3. **`mission_runner._publish_grounding_overlay` stops doing
   defensive `getattr(..., None)`** — it accesses the typed Protocol
   method directly. Best-effort `try/except` around the publish stays
   (viz failure must never block a mission), but the "method might
   not exist" branch is gone because the contract is now real.

4. **Tests move with the code.**
   - New file `source/strafer_autonomy/tests/test_grounding_publisher.py`
     owns the `_stub_vision_msgs`, `_stub_grounding_frame_deps`,
     `_add_foxglove_msgs_stub`, `_StubMsg`, `_StubMsgLegacyBbox` test
     fixtures (currently sitting in `test_ros_client.py`), plus the
     `TestPublishDetections`, `TestPublishGroundingFrame`,
     `TestPublishDetectionsFG` classes (and `TestPublishDetections::test_falls_back_to_legacy_pose2d_shape`).
   - `tests/test_ros_client.py` loses the publisher-related classes
     and their imports/fixtures. The remaining surface (sensor
     capture, Nav2, projection, rotate, helpers, robot state)
     remains.

## Acceptance criteria

- [ ] `source/strafer_autonomy/strafer_autonomy/clients/grounding_publisher.py`
      exists and exports `GroundingPublisher` (Protocol),
      `FoxgloveGroundingPublisher` (default impl), and
      `NullGroundingPublisher` (no-op for tests / disabled-viz path).
- [ ] `JetsonRosClient.publish_detections`,
      `_publish_foxglove_annotations`, `publish_grounding_frame`, the
      three `TOPIC_*` viz constants, and the body of `_setup_publishers`
      have all moved to the new module. `JetsonRosClient` retains a
      `self._grounding_publisher` member and (optionally) thin
      delegating methods if any external code still calls them by name
      — but the canonical path is via the Protocol.
- [ ] `mission_runner._publish_grounding_overlay` no longer uses
      `getattr(self._ros_client, "publish_detections", None)`. It
      accesses the typed Protocol member directly. Best-effort
      `try/except` around publish stays (mission must not be blocked).
- [ ] Optional-dep handling (`vision_msgs` / `foxglove_msgs` / `cv_bridge`
      missing) is centralized inside `FoxgloveGroundingPublisher` and
      preserves current loud-warn-at-startup-then-degrade semantics.
      The warning is emitted exactly once per missing package at
      executor start.
- [ ] All publisher tests live in `tests/test_grounding_publisher.py`.
      `tests/test_ros_client.py` keeps only its non-publisher classes
      and loses the `sys.modules` stubbing fixtures.
- [ ] Same observable behavior: same three topic names, same QoS
      (`TRANSIENT_LOCAL`, depth=1, `RELIABLE`), same payload shapes.
      Validate by running the
      [`vlm-bbox-overlay`](../completed/vlm-bbox-overlay.md)
      smoke test (Foxglove RGB panel + grounding_frame panel both
      render correctly during a `strafer-autonomy-cli submit "go to
      the door"` mission) before and after, no behavioral diff.
- [ ] `colcon test` (or the matching `pytest` subset) at green parity
      with main, plus the new test file's coverage. No optional-dep
      regressions: tests pass with `ros-humble-vision-msgs` and
      `ros-humble-foxglove-msgs` both installed AND both uninstalled
      (the test stubs already simulate both states; just confirm).
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- Current viz publisher code (the move target):
  [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
  - `TOPIC_DETECTIONS` / `TOPIC_GROUNDING_FRAME` / `TOPIC_DETECTIONS_FG`
    constants near the top of `JetsonRosClient`.
  - `_setup_publishers()` — builds the three publishers with latched
    QoS, contains the `vision_msgs` / `foxglove_msgs` ImportError
    branches.
  - `publish_detections()` — Qwen-[0,1000]→pixel rescale, the
    Humble `vision_msgs/Pose2D` hasattr branch, the call into
    `_publish_foxglove_annotations`.
  - `_publish_foxglove_annotations()` — LINE_LOOP corners +
    TextAnnotation label.
  - `publish_grounding_frame()` — `cv_bridge` BGR → Image encode,
    same stamp/frame as the detections message.
- The single caller (the smell that motivated this brief):
  [`source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py`](../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
  — `_publish_grounding_overlay()` around line 2152. The
  `getattr(..., None)` checks for `publish_detections` and
  `publish_grounding_frame` are the lines that go away.
- Existing protocol style to mirror:
  [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
  `RosClient` Protocol near the top.
  [`source/strafer_autonomy/strafer_autonomy/clients/vlm_client.py`](../../../source/strafer_autonomy/strafer_autonomy/clients/vlm_client.py)
  `GroundingClient` Protocol.
- Test fixtures to relocate (currently in `test_ros_client.py`):
  `_stub_vision_msgs` (~40 lines), `_stub_grounding_frame_deps`
  (~50 lines), `_add_foxglove_msgs_stub` (~60 lines), `_StubMsg` +
  `_StubMsgLegacyBbox` (~25 lines), plus the three `TestPublish*`
  classes (~180 lines).
- Brief that introduced this code:
  [`completed/vlm-bbox-overlay.md`](../completed/vlm-bbox-overlay.md).

## Out of scope

- **Functional / behavioral changes.** Same topics, same QoS, same
  payload shapes, same Foxglove layout. This is a pure code-move.
  Operators should see zero diff.
- **Adding new debug topics.** Mission-state telemetry, planner
  traces, semantic-map debug, etc. each gets its own brief — extract
  what exists first, *then* expand.
- **Promoting `GroundingPublisher` into `strafer_shared`.** The
  publisher lives Jetson-side because `foxglove_msgs` and
  `cv_bridge` are ROS-runtime-only deps with no DGX-side parallel.
- **Replacing the best-effort `try/except` in
  `_publish_grounding_overlay` with a hard dependency.** Viz failure
  must continue to be non-fatal — the executor must keep running
  missions even if the overlay publisher raises (publisher race during
  shutdown, transient `cv_bridge` issue, etc.). The smell removed is
  the *existence check* (`getattr(..., None)`), not the
  failure-tolerance.
- **Reorganizing the rest of `JetsonRosClient`** (sensor caching,
  Nav2, projection, rotate). Each of those concern groups has its
  own complexity story; treat the viz extraction as a precedent for
  later splits if they're worth doing, not as a blanket refactor.
