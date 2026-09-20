# Size the policy camera's unconsumed colour channel

**Type:** task (one measurement, then a decision)
**Owner:** DGX
**Priority:** P2 — a per-iteration cost the depth lane pays for nothing, on the
env count the retrain runs at.
**Estimate:** S (one 96-env smoke per arm)
**Branch:** `task/policy-camera-unconsumed-rgb`

## Story

As the **depth-subgoal retrain**, I want **the policy camera's colour channel
priced**, so that **a render product no observation reads is kept for a reason
rather than by inheritance.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [measurements/deploy-resolution-depth-2026-09-19](../../../measurements/deploy-resolution-depth-2026-09-19/README.md)
  — the render change that made this channel full-resolution.

## Context

`_prune_scene_cameras` force-unions `rgb` into the policy camera's
`data_types` on every depth variant, and the composition contract asserts it:
the RTX viewport and `--video` colour pipeline only come up when some camera
renders colour, and a depth-only variant has no other candidate.

The policy camera renders the deploy resolution, so that channel is a 640×360
RGB render product per environment that no observation term reads. At 96
environments it is rendered 96 times per step and consumed zero times.

Two things are known and bound the question. The depth annotator does not pass
through DLSS — measured: `antialiasing_mode` "DLSS" against "Off" leaves
`distance_to_image_plane` bit-identical over 3600 pixels while the colour
channel's mean moves, so the upscaler's cost is the colour channel's alone. And
the full render change costs 1.128× per iteration at 96 environments, which is
the figure this channel sits inside.

What is not known is its share of that, or whether the viewport tolerates its
absence headless — the contract's claim is that clips go black and the headed
viewport stalls, which was written when the alternative was never tried.

## Acceptance criteria

- [ ] One 96-environment headless smoke per arm, same task, seed and iteration
      count as the 2026-09-19 smoke, differing only in whether `rgb` is in the
      policy camera's `data_types`. Per-iteration collection and learning times
      and peak memory for both.
- [ ] Whether a headless run without the channel still produces usable
      `--video` output, tested rather than assumed. If it does not, the finding
      is that the channel is load-bearing and the measurement prices what the
      viewport costs.
- [ ] If the channel can go, the composition contract's assertion that the
      policy camera renders colour is revised with the measurement behind it,
      and the layout goldens are shown to hold.
- [ ] A decision either way, recorded: dropped, or kept with its price stated.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Out of scope

- The perception camera's colour channel, which the bridge streams and
  Replicator reads.
- Environment counts other than 96.
- Any change to the depth path, which this measurement must hold fixed.
