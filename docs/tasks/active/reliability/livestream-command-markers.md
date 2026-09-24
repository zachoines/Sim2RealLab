# Show the navigation command in the livestream without putting it in the scene

**Type:** task
**Owner:** DGX
**Priority:** P3 — recorded video already draws the command; only the live viewport lacks it.
**Estimate:** S
**Branch:** `task/livestream-command-markers`

## Story

As an **operator watching a run over the livestream**, I want **the goal, the rolling
subgoal and the planned path drawn over the viewport**, so that **I can see what the robot is
steering toward without any camera the policy or a stream reads seeing it too.**

## Context bundle

- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/tools-and-scripts-map.md](../../context/tools-and-scripts-map.md)

## Context

The command terms create no scene geometry for debug visualisation, because anything in the
stage is rendered by every camera in it, the D555 policy camera included
(`debug-marker-leak-2026-09-21`). Recorded `--video` frames draw the command as a 2-D
overlay (`strafer_lab.tools.command_overlay`), but a livestream shows the Kit viewport, which
that overlay never touches.

Candidates: a viewport-space overlay (`omni.ui.scene`), or Isaac Sim's debug-draw interface.
Debug-draw is now measured and **disqualified**: with a marker drawn, both the policy and the
perception camera render it in RGB, about 2 150 pixels of it, while depth is untouched
(`debug-marker-leak-2026-09-21` §5). Teleop's target marker used it on the opposite claim and
is opt-in since. So whatever mechanism this brief picks must be shown not to reach any render
product, and the viewport draws in a different place from the Replicator capture `--video`
reads.

## Acceptance criteria

- [ ] The goal, subgoal and path are visible in the livestream viewport.
- [ ] The Kit test `test_sim/sensors/test_command_markers.py` is extended to the chosen
      mechanism: with it drawing, `d555_camera` and `d555_camera_perception` carry neither the
      drawing's colour in RGB nor any change in depth. RGB is not reproducible across
      re-renders, so equality is not the test there; the drawing's own colour is.
- [x] Teleop's debug-draw target marker is covered by the same test — it reaches both cameras'
      RGB and neither depth, which is why it is opt-in.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance).

## Out of scope

- Recorded video, which the frame overlay already covers.
