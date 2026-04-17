# Strafer Autonomy Command Ingress

This document defines how natural-language commands should enter the Strafer autonomy system in the MVP and in later deployed stages.

## Recommendation

Use a stable robot-local command API on the Jetson from day one.

For the MVP, the operator should connect over SSH and use a thin CLI or ROS action client.
Do not treat raw ROS topic publishing as the primary command interface.

Later frontends such as a workstation UI, web app, mobile app, or cloud gateway should adapt into the same robot-local command API instead of bypassing it.

## Core Design Principle

Separate these concerns:

1. Human command ingress
   - how a person submits a natural-language request
2. Mission execution
   - how the Jetson executor accepts, runs, cancels, and reports missions
3. Planning and grounding
   - how the executor uses the remote planner and VLM services

The stable internal boundary should be between command ingress and the Jetson-resident executor.

## Why A Raw ROS Topic Is Not Enough

A raw topic like `std_msgs/String` is too weak for mission submission.

Problems:
- no acceptance or rejection response
- no mission identifier returned to the caller
- no built-in cancel semantics
- no structured feedback channel tied to one submitted command
- harder to evolve once multiple clients exist

A raw topic may still be useful later for telemetry or operator notifications, but it should not be the primary command-ingress contract.

## Better First Boundary: Robot-Local ROS Action

The right first ingress contract is a Jetson-local ROS action owned by `strafer_autonomy.executor`.

Proposed action name:
- `strafer_msgs/action/ExecuteMission.action`

Why an action is a good fit:
- command submission is request-response, not fire-and-forget
- missions are long-running
- feedback is needed while steps are executing
- cancel is built into the transport
- the same interface works for SSH-driven MVP use and for later UI adapters

## Proposed MVP Ingress Path

```text
Operator terminal over SSH
  -> strafer_autonomy CLI or ros2 action client on the Jetson
  -> ExecuteMission.action goal
  -> Jetson executor accepts command
  -> planner_client.plan_mission() over LAN
  -> ros_client / vlm_client execute the resulting steps
  -> action feedback and result flow back to the operator terminal
```

This keeps the MVP simple while still establishing the correct robot-local execution boundary.

## Proposed `ExecuteMission.action`

Goal fields:
- `string request_id`
- `string raw_command`
- `string source`
- `bool replace_active_mission`

Feedback fields:
- `string mission_id`
- `string state`
- `string current_step_id`
- `string current_skill`
- `string message`
- `float32 elapsed_s`

Result fields:
- `bool accepted`
- `string mission_id`
- `string final_state`
- `string error_code`
- `string message`

Notes:
- `request_id` lets the caller deduplicate retries.
- `source` identifies who sent the command, for example `ssh_cli`, `web_ui`, `mobile_app`, or `cloud_gateway`.
- `replace_active_mission` makes interruption behavior explicit.

## Additional Robot-Local Interfaces

The action alone is not quite enough. Add small supporting interfaces:

1. `strafer_msgs/srv/GetMissionStatus.srv`
- returns the current executor snapshot
- useful after CLI reconnects or UI refreshes

Suggested fields:
- `string mission_id`
- `string state`
- `string raw_command`
- `string current_step_id`
- `string current_skill`
- `string message`
- `bool active`
- `float32 elapsed_s`

2. `strafer_msgs/msg/MissionStatus.msg`
- optional status topic for dashboards or logging
- not required for the first cut if action feedback is enough

## Why This Beats A Custom HTTP Server On The Jetson For MVP

A Jetson-local HTTP server is possible, but it is not the best first internal boundary.

Reasons:
- ROS already provides action semantics that match long-running mission execution
- the executor already lives in the robot runtime
- using ROS locally reduces duplicate transport logic on the robot
- a future HTTP or WebSocket gateway can be added as an adapter without changing the executor contract

## MVP Operator Experience

For the MVP, the operator experience should be:

1. SSH into the Jetson.
2. Start the ROS stack and autonomy executor in separate terminals.
3. Submit a natural-language command through a thin CLI.
4. Watch mission feedback in the terminal.
5. Cancel the mission from the same CLI if needed.

Recommended first operator command shape:

```text
python -m strafer_autonomy.cli submit "wait by the door for me"
python -m strafer_autonomy.cli status
python -m strafer_autonomy.cli cancel
```

Important point:
- the CLI is only a frontend
- the real contract is still the robot-local action and status interface

## Final-Stage Ingress Shape

In later stages, human commands should still terminate at the same robot-local executor boundary, but the frontend changes.

### Workstation Or LAN UI

```text
Operator UI on workstation
  -> LAN gateway or direct action client
  -> ExecuteMission.action on the Jetson
  -> Jetson executor
```

### Deployed Web Or Mobile UI

```text
Web or mobile client
  -> cloud command API
  -> robot session gateway
  -> ExecuteMission.action on the Jetson
  -> Jetson executor
```

Important principle:
- public clients should not talk to ROS directly over the internet
- they should talk to a gateway or adapter layer

## Recommended Final External Pattern

For deployed operation, use two layers:

1. External command API
- HTTP or WebSocket for web and mobile clients
- authentication, audit, and client session management live here

2. Robot session bridge
- maintains a robot-safe connection to the deployed system
- forwards accepted commands into the local `ExecuteMission.action`
- forwards mission feedback back upstream

Good later transport choices for the robot session bridge:
- MQTT
- WebSocket
- HTTPS polling if simplicity matters more than latency

The exact deployed transport can change later without changing the Jetson executor contract.

## Recommended Package Placement

The command ingress pieces should live with `strafer_autonomy`, not `strafer_ros` and not `strafer_vlm`.

Suggested layout:

```text
source/strafer_autonomy/
  strafer_autonomy/
    executor/
      mission_runner.py
      command_server.py
    cli.py
    clients/
      planner_client.py
      vlm_client.py
      ros_client.py
    bridges/
      lan_gateway.py
      cloud_gateway.py
```

Why:
- command ingress is part of autonomy orchestration
- `strafer_ros` should stay focused on robot runtime skills and safety
- `strafer_vlm` should stay focused on grounding

## Recommended Decision

Adopt this now:

1. Robot-local ingress contract is `ExecuteMission.action` on the Jetson.
2. MVP operator path is SSH plus a thin CLI that sends that action goal.
3. Planner and VLM remain remote services from the executor's point of view.
4. Later workstation, web, mobile, or cloud frontends become adapters into the same robot-local command contract.

This gives the MVP a minimal operator workflow without locking the system into an SSH-only or topic-only design.

## What This Changes In The Build Order

Before implementing planner HTTP and VLM HTTP transports, the command-ingress contract should be fixed.

Updated recommended order:

1. `strafer_msgs/action/ExecuteMission.action`
2. `strafer_msgs/srv/GetMissionStatus.srv`
3. `strafer_autonomy.executor.command_server`
4. `strafer_autonomy.cli`
5. `strafer_autonomy.executor.mission_runner`
6. `strafer_msgs/srv/ProjectDetectionToGoalPose.srv`
7. `strafer_autonomy.clients.ros_client`
8. planner and VLM transport implementations

The exact order between the runner, `ros_client`, and transport clients can still move slightly, but the command-ingress boundary should be decided first.
