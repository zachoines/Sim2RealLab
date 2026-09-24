# Let a play recording show lit rooms, and leave the robot outline off on request

**Status:** Shipped 2026-09-23 in `b4e7e47` (DGX).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/226

**Type:** task
**Owner:** DGX
**Priority:** P3 — recorded video is already legible through the robot outline; this is for
cleaner shots of the policy tracking its subgoals.
**Estimate:** S
**Branch:** `task/play-open-rooms`

## Story

As an **operator recording a trained policy to show how it tracks its subgoals**, I want **the
rooms in a play recording drawn without a ceiling, and the robot outline optional**, so that
**the room reads lit and the video shows the robot itself rather than a drawn stand-in.**

## Context bundle

- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)
- [context/tools-and-scripts-map.md](../context/tools-and-scripts-map.md)
- [measurements/debug-marker-leak-2026-09-21](../../measurements/debug-marker-leak-2026-09-21/README.md)
  — the overlay, the robot outline, and why the enclosed rooms render dark from above.

## Context

The enriched rooms carry a ceiling with probability 0.7 per episode (`p_ceil` on the
`generate_room` event). The ceiling is culled for the overhead camera, but it still shades
the room: in an enclosed room most of the floor renders black and the chassis with it, which
is why the video overlay outlines the robot.

Parking the ceiling, as an open episode does, lights the floor: crushed pixels in the room
fall from 60 % to 2 % in the same room. An open room is an ordinary training episode, not an
out-of-distribution one. So `p_ceil = 0` in play changes which episodes are sampled, not
what the policy has seen. Play is not the scored evaluator — `eval_cadence_emulation.py` is —
so no metric moves.

What a play recording with every room open gives up is the enclosed mode: most of the
training distribution, and the one that looks like a real room indoors. The flag is
therefore opt-in, and training never takes it: there a recording flag would quietly change
the data a run trains on.

## Acceptance criteria

- [x] `play_strafer_navigation.py --open_rooms` sets `p_ceil` to 0 before the env is built,
      and refuses an env whose room generator draws no ceiling.
- [x] `--no_robot_outline` records the command overlay without the robot outline; the outline
      stays on by default.
- [x] A Kit-free test covers the overlay wrapper end to end — the recording camera read from a
      USD stage, the robot pose from articulation data — with the outline on and off.
- [x] A recording with `--open_rooms` has no enclosed room in it, and with `--no_robot_outline`
      no outline.
- [x] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../context/conventions.md#user-facing-documentation-maintenance).

## Out of scope

- Training and `test_strafer_env` recordings: neither takes `--open_rooms`.
- Lighting the enclosed rooms: a light, or a ceiling that stops casting shadows, changes the
  cameras' RGB and needs its own measurement.
