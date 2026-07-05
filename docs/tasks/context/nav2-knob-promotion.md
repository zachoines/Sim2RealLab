# Nav2 knob promotion — the sim→real graduation contract for `_patch_params`

How a currently sim-only Nav2 override earns its way onto the real
robot, and how to tell the two kinds of override apart before you add a
new one. The mechanism this governs is the `envelope_factor` gate in
[`navigation.launch.py`](../../../source/strafer_ros/strafer_navigation/launch/navigation.launch.py)'s
`_patch_params`.

---

## Why the split exists

`_patch_params` layers launch-time values on top of
[`nav2_params.yaml`](../../../source/strafer_ros/strafer_navigation/config/nav2_params.yaml).
`envelope_factor` is `STRAFER_NAV_VEL_SCALE / NAV_VEL_SCALE` — `1.0` on
the real robot, `2.0` in sim (`STRAFER_NAV_VEL_SCALE=1.0`). The
`envelope_factor > 1.0` gate is for changes that genuinely depend on the
lifted sim velocity envelope. Used as a catch-all for any "sim-only
change" — chosen by matching precedent rather than by that test — it
grows a silent sim-to-real gap: real-robot bringup runs largely default
Nav2 tuning while a value validated only in sim never reaches it.

The fix is a structural split — three labeled sections in `_patch_params`,
enforced by the section-invariant tests in
[`test_nav_config.py`](../../../source/strafer_ros/strafer_navigation/test/test_nav_config.py):

| Bucket | Gate | Graduates? |
|---|---|---|
| **Universal defaults** — physical constants (velocity caps, costmap resolution, footprint, scan ranges) + promoted behavioral defaults (smoothing-BT injection; `allow_unknown` / SmacPlanner2D pinned in YAML) | every lane | n/a — already universal |
| **Velocity-envelope coupling** — MPPI `vx/vy/wz_std` + `prune_distance` scaling, `vy_std` un-scale | tracks the factor | **never** — coupled by construction |
| **Behavioral overrides under promotion** — sim-only MPPI critic / convergence tuning (`PreferForwardCritic`, `PathFollowCritic`, `gamma`, `PathAlignCritic`) | `envelope_factor > 1.0` | **candidate** — via the lap below |

Only the third bucket is negotiable. The first is universal by
definition; the second must scale with the cap or MPPI plateaus
mid-envelope, so it stays gated forever.

## Why it matters on the hybrid lane too

The trained-policy backend (`hybrid_nav2_strafer`) does local control
itself and never engages Nav2's **controller**, so the bucket-3 MPPI
critic knobs only bite on the `nav2` backend lane and the hybrid
fallback paths. But the **planner / costmap / BT / constants** the same
split governs are exactly what the hybrid mission path exercises — the
real robot's global plans come from whatever Nav2 configuration survives
this split. Keeping the split honest is a sim-to-real fidelity
requirement for every backend, not only the `nav2` one.

---

## The five-step promotion contract

For each currently-gated knob:

1. **Identify.** Velocity-coupled (the value is only correct *because*
   the cap is lifted → stays gated, bucket 2) or behavioral (useful at
   any velocity, just tuned in sim → graduates, bucket 3)?
2. **Sim observable.** Capture the current sim behavior as a baseline
   snapshot — Foxglove screencap, `/plan` or `/cmd_vel` sample, mission
   log — so "no regression" is a comparison, not a memory.
3. **Real-robot validation lap.** Name the mission, the observable, and
   a falsifiable regression criterion. Disprovable: "at the indoor cap
   `/cmd_vel.vy` peak stays ≤ 0.3 × `/cmd_vel.vx` peak", not "drives
   well".
4. **Promote OR document.** Lap passes → move the value to the universal
   YAML default (or drop the gate in `_patch_params`) and delete its
   entry from the behavioral-override section. Lap fails → keep the gate
   and rewrite its inline comment with the refreshed justification (the
   observed failure mode), or file a follow-up for the regression.
5. **Cross-link.** Record the disposition on the owning brief's lap card
   and in [`BOARD.md`](../BOARD.md).

Promoting a knob is a one-line move in both `_patch_params` (gated block
→ universal section or YAML) and `test_nav_config.py` (its entry moves
from `_BEHAVIORAL_OVERRIDES` to `_UNIVERSAL_KNOBS`); the tests then
enforce the new contract.

---

## Worked example — `PreferForwardCritic.cost_weight`

First knob in the lap order (smallest blast radius).

| Step | For `PreferForwardCritic.cost_weight` (baseline `6.0` → sim `10.0`) |
|---|---|
| **Identify** | *Behavioral.* Biases the MPPI sampling distribution toward forward motion over strafe/spin — useful at any velocity, more load-bearing in sim only because the wider exploration envelope surfaces more lateral rollouts. → graduation candidate. |
| **Sim observable** | On a `translate forward 1 m` + cornering smoke in sim, `/cmd_vel` stays vx-dominant and `/optimal_trajectory` tracks forward along `/plan`. Capture the Foxglove trace as the baseline. |
| **Real lap** | Same mission on the real robot at the indoor cap (~0.78 m/s). Observable: no stall, no reverse-along-path, arrival within `xy_goal_tolerance`. Regression: over-biases forward and overshoots, or (if too weak on real) drives tangent to / backward along the path. |
| **Promote / document** | Holds → set `PreferForwardCritic.cost_weight: 10.0` in `nav2_params.yaml`, drop it from the gated block, move its test entry to `_UNIVERSAL_KNOBS`. Fails → keep gated, rewrite the inline comment with the observed cap-specific failure. |
| **Cross-link** | Disposition on the lap card in the owning brief + a BOARD row. |

---

## Pointers

- Gate + three sections: [`navigation.launch.py`](../../../source/strafer_ros/strafer_navigation/launch/navigation.launch.py) `_patch_params`.
- Universal YAML defaults land here: [`nav2_params.yaml`](../../../source/strafer_ros/strafer_navigation/config/nav2_params.yaml).
- Split invariants + byte-identical pin: [`test_nav_config.py`](../../../source/strafer_ros/strafer_navigation/test/test_nav_config.py) — `TestPromotionSplitInvariants`, `TestPatchByteIdentical`.
- Origin of the gate (the velocity-coupled MPPI rebalance): [`completed/mppi-critic-tuning-for-sim-envelope.md`](../completed/mppi-critic-tuning-for-sim-envelope.md).
- The two knobs already promoted (smoothing BT, `allow_unknown`): [`completed/nav2-commit-and-follow-path-stability.md`](../completed/nav2-commit-and-follow-path-stability.md).

## Maintenance contract

A brief that promotes a knob, adds a new gated knob, or retunes one
updates the bucket table and (if the example knob changes) the worked
example here in the same PR — same rule as the other context modules. A
change to the *gate mechanism* itself (the `envelope_factor` definition)
is an architecture change and revises this module.
