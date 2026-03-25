# Strafer Autonomy Systems Overview

This document is the high-level visual companion to the detailed autonomy docs.

It summarizes the main decisions already captured in:
- `STRAFER_AUTONOMY_COMMAND_INGRESS.md`
- `STRAFER_AUTONOMY_DEPLOYMENT_MODES.md`
- `STRAFER_AUTONOMY_INTERFACES.md`
- `STRAFER_AUTONOMY_LOCAL_DEVELOPMENT.md`
- `STRAFER_AUTONOMY_MVP_RUNTIME_DECISION.md`
- `STRAFER_AUTONOMY_ROADMAP.md`

## System Roles

```mermaid
flowchart LR
    U[Operator]
    A[strafer_autonomy\nplanner + executor + mission state]
    V[strafer_vlm\nsemantic grounding]
    LAB[strafer_lab / Isaac Lab\nRL training and evaluation]
    RL[Trained RL behavior policies]
    R[strafer_ros\nrobot runtime skills]
    N[Nav2, behavior trees,\nand local navigation services]
    S[Sensors, depth, TF, odometry]

    U --> A
    A --> V
    A --> R
    LAB --> RL
    RL --> R
    R --> N
    S --> R
```

## Chosen MVP Runtime

```mermaid
flowchart LR
    subgraph Jetson[Jetson Orin Nano]
        CLI[SSH CLI or ros2 action client]
        EX[strafer_autonomy.executor]
        RC[ros_client]
        ROS[strafer_ros]
        RLP[RL policy inference\nand robot behaviors]
        NAV[Nav2 and behavior trees]
        SNS[Sensors + TF + depth]

        CLI --> EX
        EX --> RC
        RC --> ROS
        ROS --> RLP
        ROS --> NAV
        SNS --> ROS
    end

    subgraph Workstation[Windows Workstation]
        PL[Planner service]
        LLM[Text LLM]
        VS[VLM service]
        QW[Qwen2.5-VL]
        LAB[strafer_lab / Isaac Lab]
        CKPT[RL policy checkpoints]

        PL --> LLM
        VS --> QW
        LAB --> CKPT
    end

    EX -. LAN HTTP .-> PL
    EX -. LAN HTTP .-> VS
    CKPT -. deployment artifact .-> RLP
```

## MVP Command Ingress

```mermaid
flowchart LR
    OP[Operator over SSH] --> CLI[strafer_autonomy CLI]
    CLI --> ACT[ExecuteMission.action]
    ACT --> EX[Jetson executor]
    EX --> FB[Action feedback and final result]
    FB --> CLI

    CLI -. optional .-> ST[GetMissionStatus.srv]
    ST --> EX
```

## High-Level Mission Execution Flow

```mermaid
flowchart TD
    C[User command: wait by the door for me] --> P[planner_client.plan_mission]
    P --> MP[MissionPlan]
    MP --> OBS[ros_client.capture_scene_observation]
    OBS --> G[vlm_client.locate_semantic_target]
    G --> PROJ[ros_client.project_detection_to_goal_pose]
    PROJ --> NAV[ros_client.navigate_to_pose]
    NAV --> WAIT[executor wait step / hold state]
    WAIT --> DONE[Mission complete or next command]
```

## Stable Interface Boundaries

```mermaid
flowchart LR
    subgraph Edge[Robot edge runtime]
        EX[strafer_autonomy.executor]
        RC[ros_client]
        ROS[strafer_ros]
    end

    subgraph Remote[Remote model services]
        PC[planner_client]
        VC[vlm_client]
        PL[Planner service]
        VS[VLM service]
    end

    EX --> RC
    RC --> ROS
    EX --> PC
    EX --> VC
    PC --> PL
    VC --> VS
```

## Deployment Evolution

```mermaid
flowchart LR
    subgraph Robot[Always on robot]
        EX[Jetson executor]
        ROS[strafer_ros]
    end

    subgraph Home[Home or lab mode]
        PL1[Planner on workstation]
        V1[VLM on workstation]
    end

    subgraph Cloud[Later deployed mode]
        API[Command API / gateway]
        PL2[Planner in AWS or Databricks]
        V2[VLM in AWS or Databricks]
    end

    EX --> ROS
    EX -. current LAN path .-> PL1
    EX -. current LAN path .-> V1
    EX -. later remote path .-> PL2
    EX -. later remote path .-> V2
    API -. later operator path .-> EX
```

## Architecture Summary

- `strafer_ros` stays robot-local and owns sensing, TF, projection, navigation, and safety-critical execution.
- `strafer_autonomy.executor` stays robot-local and owns mission state, retries, timeout, cancel, and skill sequencing.
- `strafer_lab` trains RL navigation and behavior policies that are later deployed onto the robot as inference artifacts.
- The robot execution layer is not just Nav2: it also includes trained RL behavior policies that sit alongside or underneath the classical navigation stack.
- Planner and VLM are heavy remote services from the executor's point of view.
- The first operator path is SSH plus CLI, but the long-term stable command boundary is robot-local `ExecuteMission.action`.
- Future workstation, web, mobile, or cloud frontends should adapt into the same robot-local executor contract rather than replacing it.
