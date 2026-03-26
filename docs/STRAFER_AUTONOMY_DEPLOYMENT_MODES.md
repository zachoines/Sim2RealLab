# Strafer Autonomy Deployment Modes

This document describes how the local-development architecture evolves into deployed hosting.

The deployment problem is not just "where do we host Qwen?"
It is:
- where does the planner live
- where does the VLM live
- what must remain robot-local
- which boundaries should stay stable across LAN and cloud deployment

## Core Deployment Rule

The robot execution boundary must remain local.

That means these stay on or next to the robot:
- `strafer_ros`
- sensing
- TF
- depth projection
- robot-local navigation and execution modes
- cancel and safety-critical stop behavior
- the autonomy executor for the chosen MVP and likely for the long-term product shape

The planner and VLM can move.
The robot execution layer should not.

## Three Deployment Modes

### Mode 1: Workstation-hosted planning and VLM, robot-local executor

This is the chosen MVP target.

```text
Jetson: strafer_ros + strafer_autonomy.executor
Workstation: planner service + VLM service
Transport: local ROS on robot + LAN service calls
```

Use this for:
- early integration
- prompt iteration
- low-cost development
- proving the skill contracts
- keeping mission state local while offloading model compute

### Mode 2: Cloud-hosted planning and VLM, robot-local executor

This is the recommended deployed target.

```text
Jetson: strafer_ros + strafer_autonomy.executor
Cloud: planner API + VLM API
Transport: local ROS on robot + HTTPS to cloud services
```

Use this for:
- demos away from home
- cases where Jetson cannot host the models
- environments where you want deployed hosting but still need local safety and mission execution control

### Mode 3: Split hosted services

```text
Jetson: strafer_ros + strafer_autonomy.executor
Workstation or cloud: planner API
Different workstation or cloud GPU: VLM API
```

Use this only if needed for cost, latency, or experimentation reasons.

This is acceptable because the planner and VLM are already separate logical services.

## Recommended Deployment Split

For both the MVP and the likely long-term system, the best split is:

| Component | Recommended location |
|----------|----------------------|
| `strafer_ros` | Jetson |
| `strafer_autonomy.executor` | Jetson |
| `strafer_autonomy.planner` | workstation or cloud |
| `strafer_vlm` | workstation or cloud GPU |

Why this is the right split:
- mission progression, retries, timeouts, and cancellation stay close to the robot
- the planner can be remote because it mostly handles text and structured plans
- the VLM can be remote because it is heavy and runs at low rate
- the robot remains safe if planner or VLM connectivity is lost
- the deployed architecture becomes a straightforward extension of the MVP rather than a redesign

Robot-local execution note:
- the stable autonomy skill remains `navigate_to_pose`
- the robot may satisfy that skill through different local execution modes such as `nav2`, `strafer_direct`, or `hybrid_nav2_strafer`
- deployment mode changes where planner and VLM live, not the autonomy skill contract

## Which Components Fit Which Hosting Options

### `strafer_vlm`

Best hosting candidates:
1. workstation LAN service for MVP
2. EC2 GPU service
3. SageMaker Async for low-rate, scale-to-zero VLM jobs
4. Databricks custom model serving if you want platform-level ML serving

Important AWS constraint:
- API Gateway HTTP APIs have a payload limit of `10 MB`, so direct large image uploads through a simple synchronous API are constrained by that limit.
- SageMaker async supports request payloads up to `1 GB` via S3 and can scale endpoint instances to zero when idle.

Implication:
- SageMaker Async fits the VLM service well
- it does not fit the robot-local executor loop well

### `strafer_autonomy.planner`

Best hosting candidates:
1. workstation service during development
2. small EC2-hosted API for deployed use
3. Databricks model serving if the planner itself is packaged as a served model and you want platform features

Less suitable:
- SageMaker Async

Reason:
- the planner is a low-payload, interactive, synchronous text-to-plan service
- it benefits more from a straightforward request-response API than an async queue/S3 pattern

### `strafer_autonomy.executor`

Best hosting candidate:
- robot-local

Reason:
- it owns mission progression, retries, timeouts, and cancellation
- it should survive WAN latency and remote service failures
- it should not require cloud round-trips to advance local mission state

### `strafer_ros`

Best hosting candidate:
- always robot-local

No change here.

## AWS-Oriented Deployed Shape

The most practical AWS-oriented deployed shape is:

```text
Jetson
  - strafer_ros
  - strafer_autonomy.executor

AWS
  - planner API
  - VLM API
  - optional object storage for images/results
```

Recommended service mapping:
- planner API: EC2 or containerized service
- VLM API: EC2 GPU first, SageMaker Async second if low-rate and cold-start is acceptable
- storage: S3 if using async VLM jobs or saving artifacts
- auth and ingress: API Gateway or direct HTTPS depending on maturity

Why not put the executor in SageMaker or Databricks:
- those are model-serving platforms, not robot mission-runtime platforms

## Databricks-Oriented Deployed Shape

Databricks can make sense for the planner or VLM service, but not for robot execution.

What Databricks is good for here:
- managed model serving
- model registry and MLflow workflows
- fast experimentation with serving endpoints
- serving custom models with scale-to-zero as an option

What Databricks is not good for here:
- owning robot mission execution state
- acting as a robot-side control runtime
- replacing `strafer_ros`

Important Databricks constraints from the docs:
- custom model serving payload size is `16 MB` per request
- request execution duration is `297 seconds`
- endpoints can scale to zero, but Databricks notes that capacity is not guaranteed when scaled to zero and cold-start latency is expected
- custom model scoring is shaped around DataFrame or tensor inputs, which is workable but less robot-native than your own custom API

Implication:
- Databricks is a reasonable hosted backend for the planner or VLM if you want a stronger ML platform
- it is probably not the simplest first deployed target for Strafer

## Stable Contracts Across All Modes

These boundaries should not change whether you run on workstation, EC2, SageMaker, or Databricks:

1. planner output
   - `MissionIntent`
   - `MissionPlan`
   - `SkillCall`

2. VLM output
   - `GroundingResult`

3. robot-side projection output
   - `GoalPoseCandidate`

4. executor-facing skill result
   - `SkillResult`

This is the key design constraint that keeps local-development and deployed modes aligned.

## Practical Recommendation

Build toward this sequence:

1. workstation-hosted planner and VLM, robot-local executor
   - `strafer_autonomy.executor` on Jetson
   - planner and VLM services on Windows workstation

2. cloud-hosted planner and VLM, robot-local executor
   - keep executor and robot execution local
   - move planner and VLM services to cloud-hosted endpoints as needed

3. only later decide whether the remote services are best on:
   - EC2
   - SageMaker for VLM specifically
   - Databricks for model-serving-heavy workflows

## Chosen MVP Runtime

The chosen first implementation target is:
- executor on Jetson
- planner on workstation
- VLM on workstation
- planner and VLM both accessed by the Jetson over LAN service calls

This is better than a workstation-hosted executor because it keeps mission control local while still offloading model compute.

## References

AWS and Databricks details used in this document:
- SageMaker async supports payloads up to 1 GB and can scale endpoint instances to zero: https://docs.aws.amazon.com/en_us/sagemaker/latest/dg/async-inference.html
- SageMaker async autoscaling to zero: https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference-autoscale.html
- API Gateway HTTP API payload limit is 10 MB: https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-quotas.html
- Databricks custom model serving can scale to zero but documents cold-start/capacity caveats: https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints
- Databricks custom model serving limits include 16 MB payloads and 297 second execution duration: https://docs.databricks.com/aws/en/machine-learning/model-serving/model-serving-limits
- Databricks custom model endpoints are invoked with DataFrame or tensor request formats: https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints
