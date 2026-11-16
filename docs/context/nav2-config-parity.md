# Nav2 config parity — one configuration, both lanes

The Nav2 configuration is **byte-identical on every lane by
construction**. Sim-in-the-loop and real-robot bringup load the same
[`nav2_params.yaml`](../../../source/strafer_ros/strafer_navigation/config/nav2_params.yaml)
and receive the same launch-time overrides. There is no sim-vs-real
branch in the launch file and no environment knob that scales Nav2's
velocity caps.

---

## What `_patch_params` does

[`navigation.launch.py`](../../../source/strafer_ros/strafer_navigation/launch/navigation.launch.py)'s
`_patch_params` layers only **physical constants** on top of the YAML:

| Injected at launch | Source |
|---|---|
| MPPI / behavior / smoother velocity caps | `NAV_LINEAR_VEL`, `NAV_ANGULAR_VEL`, `NAV_REVERSE_VEL` (an indoor-safety fraction of the chassis max, `strafer_shared.constants`) |
| costmap resolution | `MAP_RESOLUTION` |
| footprint | `CHASSIS_LENGTH` × `TRACK_WIDTH` |
| scan raytrace / obstacle ranges | `DEPTH_MIN`, `DEPTH_MAX` |
| `default_nav_to_pose_bt_xml` | resolved from `ament_index` (can't be pinned in YAML) |

Everything else — MPPI sampling stds, critic weights, `gamma`,
`prune_distance`, planner and costmap tuning — is a plain YAML default.
The velocity caps come from **one number for both lanes**: the indoor
cap (`NAV_VEL_SCALE = 0.5` of chassis max, ~0.78 m/s). The same number
also caps the trained policy's L1 clamp in `strafer_inference`, so the
`nav2` and trained-policy backends agree on the velocity envelope without
any operator-set variable.

## Introducing a lane divergence

A value that should differ between sim and real does **not** get a
standing gate. Follow the lifecycle, which ends by deleting the
temporary mechanism:

1. **Sim first.** Tune and validate the change in sim on the branch.
2. **Temporary flag on the robot.** Land it behind a clearly-scoped,
   *temporary* flag (env var or launch arg named for the one experiment)
   so the robot can run with and without it.
3. **A/B on the robot.** Compare the flagged and unflagged runs on a
   concrete, falsifiable observable.
4. **Universalize the winner and delete the flag.** Move the winning
   value into `nav2_params.yaml` (or the constants path) and remove the
   flag. A losing experiment is reverted, not left gated.

Permanent lane-gated knobs are not a thing. If an experiment can't be
universalized, it's a finding to record, not a config branch to keep.

## Why it matters on the hybrid lane too

The trained-policy backend (`hybrid_nav2_strafer`) does local control
itself and never engages Nav2's **controller**, so the MPPI critic knobs
only bite on the `nav2` backend lane and the hybrid fallback paths. But
the **planner / costmap / BT / constants** the same config governs are
exactly what the hybrid mission path exercises — the real robot's global
plans come from whatever Nav2 configuration this file emits. Parity is a
sim-to-real fidelity requirement for every backend, not only `nav2`.

---

## Pointers

- Launch-time injection: [`navigation.launch.py`](../../../source/strafer_ros/strafer_navigation/launch/navigation.launch.py) `_patch_params`.
- The shared config: [`nav2_params.yaml`](../../../source/strafer_ros/strafer_navigation/config/nav2_params.yaml).
- Both-lanes-identical pin: [`test_nav_config.py`](../../../source/strafer_ros/strafer_navigation/test/test_nav_config.py) — `TestPatchByteIdentical` (single golden fixture = the shipped config) + `TestConstantsInjection`.
- Trained-policy velocity clamp (same indoor cap): `strafer_inference/inference_node.py` `_DEFAULT_VEL_CAP_*`.
- MPPI sweep capture harness for the sim-first step: [`scripts/tune_capture.py`](../../../source/strafer_ros/strafer_navigation/scripts/tune_capture.py).

## Maintenance contract

A brief that retunes a universalized value, or runs a divergence
experiment through the lifecycle, updates this module in the same PR —
same rule as the other context modules.
