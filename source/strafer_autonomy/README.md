# strafer_autonomy

Autonomy planning, mission execution, shared schemas, and client interfaces for Strafer.

## Install

```powershell
python -m pip install -e source/strafer_autonomy
```

## Package Layout

```text
strafer_autonomy/
  clients/   # Planner, VLM, and ROS client protocols plus first transport stubs
  schemas/   # Typed data contracts for planner, executor, ROS, and VLM boundaries
```
