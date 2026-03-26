# Strafer Autonomy Local Development

This document describes the chosen local-development topology for Strafer when the Jetson cannot host both the autonomy LLM and the VLM.

The chosen MVP split is:
- Jetson: `strafer_ros` and `strafer_autonomy.executor`
- Windows workstation: `strafer_autonomy.planner` service and `strafer_vlm` service

## Important Terminology

There are three different meanings of "local" here.

1. Robot-local on the Jetson
   - `strafer_ros` and the autonomy executor run on the robot computer
   - this is where sensing, TF, depth projection, navigation execution, cancel, and safety-critical behavior stay

2. `localhost` on the Windows workstation
   - used only inside the workstation host
   - for example, a planner API and a VLM API may each call their local models in-process or on `127.0.0.1`

3. Local network between Jetson and workstation
   - from the Jetson's point of view, the workstation is not `localhost`
   - the Jetson talks to workstation-hosted planner and VLM services over a LAN address such as `http://192.168.1.50:8000`

That distinction matters because the architecture should be built around stable service boundaries that can later move from LAN to cloud without changing the skill contracts.

## Chosen MVP Topology

```text
Jetson Orin Nano                         Windows Workstation
----------------                         -------------------
strafer_ros                              planner service
  - sensors                                - text LLM
  - TF                                     - MissionPlan output
  - depth projection                      VLM service
  - local execution modes                  - Qwen grounding
    (Nav2 first; direct and hybrid RL later)
strafer_autonomy.executor
  - mission runner
  - ros_client (local)
  - planner_client (LAN)
  - vlm_client (LAN)
```

In this mode:
- the robot keeps all sensing, TF, depth projection, and actuation local
- the Jetson owns mission execution state for the MVP
- the workstation hosts the heavy models
- planner and VLM are both workstation services called over LAN

## Why This Is The Right First Shape

This is the recommended first mode because it balances product shape and implementation effort:
- it removes the Jetson model-hosting bottleneck immediately
- it keeps mission execution close to the robot
- it matches the likely long-term deployed shape better than a workstation-hosted executor
- it avoids introducing cloud hosting before the autonomy contracts are proven
- it keeps the robot-safe boundary local even if the planner or VLM later move to AWS or Databricks

## Recommended Local Development Flow

The chosen first call path is:

```text
User command
  -> planner service on workstation
  -> MissionPlan returned to Jetson executor
  -> executor calls ros_client.capture_scene_observation() locally
  -> executor calls VLM service on workstation over LAN
  -> executor calls ros_client.project_detection_to_goal_pose() locally
  -> executor calls ros_client.navigate_to_pose() locally
```

Expanded view:

```text
User command
  -> strafer_autonomy.executor on Jetson
  -> planner_client.plan_mission() over LAN
  -> MissionPlan
  -> ros_client.capture_scene_observation() on Jetson
  -> vlm_client.locate_semantic_target() over LAN
  -> ros_client.project_detection_to_goal_pose() on Jetson
  -> ros_client.navigate_to_pose() via the selected local execution mode
  -> executor monitors status, cancel, retry, timeout
```

## What Runs Where

| Component | Jetson | Workstation |
|----------|--------|-------------|
| `strafer_ros` | Yes | No |
| `strafer_autonomy.executor` | Yes | No |
| `strafer_autonomy.planner` | No | Yes |
| `strafer_vlm` | No | Yes |
| Qwen weights | No | Yes |
| small planning LLM | No | Yes |

This is the simplest MVP that keeps the robot execution boundary local while still offloading model compute.

## Chosen Transport Boundaries

### Executor to `strafer_ros`

Chosen first mode:
- local ROS2 subscriptions, services, and actions on the Jetson

Why:
- no LAN dependency for sensing or navigation execution
- robot-critical execution stays local
- easiest path to safe cancellation and retry logic

Important clarification:
- the autonomy skill stays `navigate_to_pose`
- the robot-local execution mode may be `nav2`, `strafer_direct`, or `hybrid_nav2_strafer`

### Executor to planner

Chosen first mode:
- LAN HTTP or gRPC request-response service

Why:
- the planner is remote from the executor in this MVP
- planner requests are small and synchronous
- this is a clean boundary that can later move from workstation to cloud

### Executor to `strafer_vlm`

Chosen first mode:
- LAN HTTP request-response service

Important note:
- once the executor moves to the Jetson, the runtime VLM path is no longer an in-process Python call
- in-process calls remain valid only inside the workstation service implementation itself

Why this is acceptable:
- VLM inference is low-rate
- the request boundary is already naturally typed
- this is the same style of boundary the deployed system will need later

## Workstation Service Shape

The workstation may run these as:
1. two separate services
   - planner API
   - VLM API
2. one host process exposing two endpoints
   - `/plan`
   - `/ground`

For the MVP, either is acceptable as long as the Jetson sees stable logical clients:
- `planner_client`
- `vlm_client`

## Why Not Keep The Executor On The Workstation

Not for the MVP, because that would keep too much mission state away from the robot.

Moving the executor to the Jetson now is better because:
- cancel and timeout handling stay robot-local
- mission progression stays closer to local robot state
- the cloud or workstation later only need to supply planning and perception
- the deployment path is cleaner

The cost of this decision is explicit network boundaries earlier in the process.
That is acceptable and worth paying now.

## Local Development Failure Model

The robot should remain safe if the workstation is unavailable.

Expected behavior:
- if the planner service is offline, the robot does not start new missions
- if the VLM service is offline, `locate_semantic_target` fails cleanly and the mission stops or retries according to policy
- if the workstation disconnects mid-mission, the Jetson executor remains in control of mission state and can cancel or stop safely
- local navigation should not continue under undefined remote assumptions

The key rule is:
- loss of workstation should degrade the robot to a safe local state, not an undefined distributed state

## Networking Checklist For Workstation Mode

1. Static IP or stable hostname for the workstation
2. Same `ROS_DOMAIN_ID` on Jetson for local ROS components that need it
3. Reliable LAN path between Jetson and workstation
4. Wired Ethernet preferred over Wi-Fi for first integration
5. Windows firewall rules for planner and VLM service ports
6. GPU-enabled Python environment only on the workstation
7. Health checks for both planner and VLM services

## Why This Mode Is Worth Building First

This mode proves the important architecture decisions without committing to cloud deployment.

It answers the important questions first:
- can the planner produce usable plans
- can the VLM ground live robot observations
- can the robot execute resulting skills while keeping mission control local
- can the same planner and VLM boundaries later be moved to deployed hosting without changing the executor contracts

## Exit Criteria For Local Development Mode

This mode is successful when all of the following are true:
1. Jetson runs robot-side ROS execution and the autonomy executor
2. workstation runs planner and VLM reliably on GPU
3. executor can request plans and VLM grounding over LAN reliably
4. the robot executes navigation-related skills without moving mission state off the Jetson
5. the same typed interfaces can later be swapped to deployed backends
6. the robot safely handles planner or VLM service loss
