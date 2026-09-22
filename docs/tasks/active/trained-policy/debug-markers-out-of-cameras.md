# Keep the command debug markers out of every camera the policy, a stream or a dataset reads

**Type:** bug
**Owner:** DGX
**Priority:** P1 — a bridge session puts a phantom goal sphere and cone into the perception
stream the Jetson consumes, and any `--video` training run puts positioned markers into the
policy's depth; the sim-bridge rig gate cannot run until neither can happen.
**Estimate:** M
**Branch:** `task/debug-markers-out-of-cameras`

## Story

As the **depth-subgoal policy and everything that records what it sees**, I want **the goal
and subgoal debug markers to be drawn only where a human looks at them**, so that **no
training frame, bridge frame or captured frame contains an object the world does not.**

## Context bundle

- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/env-composition-contract.md](../../context/env-composition-contract.md)
- [context/repo-topology.md](../../context/repo-topology.md)

## Context

The command terms in `mdp/commands.py` draw their debug markers — a goal sphere and cone, a
subgoal sphere and cone, and the path as dots — as USD point instancers under `/Visuals`.
Those are scene geometry: the D555 policy camera renders them in RGB and in
`distance_to_image_plane` depth. Isaac Lab's `primvars:invisibleToSecondaryRays` on the
prototypes hides them from neither. Every registered navigation env enables them through the
shared command cfgs, so:

- on this Isaac Lab version the markers are positioned whenever a visualizer is registered —
  the play script always registers one, the train script does under `--video`, and a bridge
  does under `--viz kit` — and otherwise sit frozen at the world origin, which is the room
  centre at one environment;
- the bridge composes the goal objective, so it draws a goal sphere at a goal the sim picks
  for itself, unrelated to the subgoal the Jetson sends.

Separately, `--headless` on the command line makes the play script, and the train script
under `--video`, raise at env construction: the deprecated flag disables every visualizer
while both scripts request a Kit one. The forced visualizer existed only to position the
markers.

## Acceptance criteria

- [ ] The command terms create no scene geometry for debug visualisation:
      `set_debug_vis(True)` adds no prim and changes nothing any camera renders.
- [ ] `debug_vis` is off in the four shared command cfgs, and the composition goldens move by
      exactly `commands.goal_command.debug_vis`, attributed by field name; the observation
      and layout goldens hold.
- [ ] A Kit-free contract test fails if any camera-bearing navigation env enables command
      debug visualisation, and is shown to fail on a single-cfg mutation.
- [ ] A Kit test shows the policy depth is bit-identical with the markers requested and not
      requested, with a positive control that proves the comparison sees real geometry at the
      same point, and is shown to fail on the tree before this change.
- [ ] Recorded play and training video draws the goal, subgoal and path as a 2-D overlay from
      the command state, with no scene geometry.
- [ ] `--headless` records video in both scripts.
- [ ] The finding is recorded with its evidence, including that the v3 training contract
      differs from the post-fix contract by exactly this field.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance).
- [ ] No regression in the workflows the touched code supports.

## Investigation pointers

- `mdp/commands.py`: `GoalCommand` and `SubgoalCommand` `_set_debug_vis_impl` /
  `_debug_vis_callback`; `GoalCommandProcRoom` and `CaptureSubgoalCommand` inherit them.
- `strafer_env_cfg.py`: the four `debug_vis=True` command cfgs.
- `test_sim/env/test_composition_contract.py`: the goldens and the serializer.
- `test_sim/sensors/test_d555_camera_prim_jitter.py`: the in-place re-render comparison.

## Out of scope

- Retraining. v3 was trained with the markers frozen at the world origin, outside every room,
  so it never saw an informative marker.
- Hiding geometry from one camera but not another. Per-camera visibility in RTX is unverified.
- Markers in the livestream viewport: filed as `livestream-command-markers`.
