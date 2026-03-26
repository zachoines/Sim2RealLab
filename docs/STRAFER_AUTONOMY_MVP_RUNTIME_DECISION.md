# Strafer Autonomy MVP Runtime Decision

This document records the chosen runtime target for the first autonomy-layer MVP.

It consolidates the earlier discussions in:
- `STRAFER_AUTONOMY_LOCAL_DEVELOPMENT.md`
- `STRAFER_AUTONOMY_DEPLOYMENT_MODES.md`
- `STRAFER_AUTONOMY_INTERFACES.md`

## Decision Summary

The first Strafer autonomy MVP will run as follows:

| Component | Runtime location |
|----------|------------------|
| `strafer_ros` | Jetson |
| `strafer_autonomy.executor` | Jetson |
| `strafer_autonomy.planner` | Windows workstation |
| `strafer_vlm` | Windows workstation |
| executor to planner path | LAN service call |
| executor to VLM path | LAN service call |
| planner and VLM model execution | workstation-local |

This is the single concrete MVP runtime target.

## Why These Are Good MVP Decisions

### 1. Executor on the Jetson

This is the right first target because:
- mission state stays robot-local from day one
- cancel, timeout, and retry logic stay close to the robot
- the MVP already matches the likely long-term product shape better
- the robot remains the owner of execution even if workstation services fail

### 2. Planner on the workstation

This is the right first target because:
- it removes the Jetson LLM compute burden immediately
- it keeps planner iteration fast on the workstation
- it creates a clean planner service boundary that can later move to cloud hosting

### 3. VLM on the workstation

This is the right first target because:
- it removes the Jetson VLM compute burden immediately
- it keeps Qwen on the workstation GPU where it is already validated
- it creates a clean grounding service boundary that can later move to cloud hosting

### 4. LAN service calls from executor to planner and VLM

This is the right first target because:
- the executor is no longer co-located with either heavy model
- the service boundaries now match the likely deployed topology
- the transport complexity is acceptable and worth paying now to keep execution local

## Important Correction To The Earlier In-Process Idea

The earlier idea of an in-process `strafer_autonomy -> strafer_vlm` call is no longer the runtime MVP choice.

Why:
- the executor now runs on the Jetson
- the VLM runs on the workstation
- therefore the runtime grounding path must be a network call

What remains true:
- the workstation-hosted VLM service may still call Qwen in-process inside its own implementation
- but from the executor's point of view, the VLM is now a remote service over LAN

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

## Chosen Call Path

```text
User command
  -> Jetson executor receives command
  -> planner_client.plan_mission() over LAN
  -> MissionPlan
  -> ros_client.capture_scene_observation() on Jetson
  -> vlm_client.locate_semantic_target() over LAN
  -> ros_client.project_detection_to_goal_pose() on Jetson
  -> ros_client.navigate_to_pose() on Jetson
  -> executor monitors retry, timeout, cancel, and completion
```

## Why This Is Better Than A Workstation Executor

It is better because:
- mission control stays local to the robot
- local robot state does not need to be mirrored into a remote executor loop from day one
- moving planner and VLM to cloud later becomes a service-hosting problem, not an execution-architecture redesign

The cost is:
- explicit network boundaries arrive earlier

That is an acceptable trade.

## Risks Accepted In This MVP

1. Workstation availability is required for planning and grounding.
2. LAN connectivity between Jetson and workstation is required.
3. Planner and VLM are not yet deployed away from home.
4. The first implementation needs explicit planner and VLM service clients.

These are acceptable risks because the goal is to validate:
- planner output quality
- VLM grounding utility
- real robot execution through typed skill interfaces
- the correct robot-local execution boundary

## Exit Criteria For This Runtime Choice

This runtime decision is successful if all of the following are true:
1. the executor on Jetson can run missions end to end
2. planner service can return bounded plans over LAN reliably
3. VLM service can ground live robot observations over LAN reliably
4. `strafer_ros` can execute resulting robot skills locally and safely
5. the same planner and VLM boundaries can later move to cloud hosting without redesigning the executor

## What This Enables Next

Once this MVP works, the next architecture step becomes much clearer:
- move planner to cloud while keeping executor local
- move VLM to cloud while keeping executor local
- or move both planner and VLM to deployed services while preserving the same executor contracts

The same runtime choice also preserves the robot-local execution abstraction:
- `navigate_to_pose` remains the stable skill
- the Jetson may later satisfy it through `nav2`, `strafer_direct`, or `hybrid_nav2_strafer`

Until then, this document is the source of truth for the MVP runtime target.
