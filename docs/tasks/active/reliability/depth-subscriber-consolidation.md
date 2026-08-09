# Stop paying the wire twice for one depth stream

**Type:** task (deploy runtime)
**Owner:** Jetson
**Priority:** P2 — a real, measured halving of the depth lane's wire cost, but
**not sufficient on its own**: it is a margin lever behind
[`sim-bridge-link-transport-capacity`](sim-bridge-link-transport-capacity.md),
which is the one that changes the order of magnitude.
**Estimate:** S (one subscription retargeted, or one republisher; the care is
all in not breaking the `_sync` contract RTAB-Map depends on)
**Branch:** `task/depth-subscriber-consolidation`

## Story

As the **sim-bridge deploy lane**, I want **one process on this host to receive
the raw depth stream and everything else to consume it locally**, so that **the
cross-host link carries one copy of a 921 KB frame instead of two.**

## Context

Filed by
[`depth-receiver-host-capacity`](../../completed/depth-receiver-host-capacity.md).
Its census of the raw 640×360 stream, measured 2026-08-08:

| raw topic | payload / frame | remote subscribers |
|---|---|---|
| `/d555/depth/image_rect_raw` | 921 600 B (`32FC1`) | `strafer_inference`, `timestamp_fixer` |
| `/d555/color/image_raw` | 691 200 B (`rgb8`) | `timestamp_fixer` |

**DDS data on this path is unicast per subscribing process** — measured, not
assumed: multicast share of Jetson receive traffic is 0.01%, and DGX TX rises
67.5 → 102.0 → 127.6 Mbit/s for 1 → 2 → 3 subscriber processes while delivered
goodput *falls*. So the second depth subscriber costs a full extra copy:
**221 × RTF Mbit/s**, against a path measured at 20–60 Mbit/s loss-free.

**The consolidation point already exists.** `timestamp_fixer` receives the raw
depth and republishes `/d555/**_sync`, and rtabmap, `depth_image_proc`'s
`depth_to_pointcloud` and `pointcloud_to_laserscan` all consume those local
republications at no wire cost. The inference node is the **only** consumer
still reaching across the link for its own copy. Pointing it at a local
republication is the same move the SLAM chain already makes.

**The per-process qualifier is the whole mechanism.** Readers that share a DDS
participant share its receive locators, so one copy serves all of them. The
deployed stack shares nothing — every service is its own container, its own
participant, its own unicast locator. A consolidation that leaves the node in a
separate participant from the receiver **buys nothing**, so the design has to be
checked against that, not just against subscriber count.

## Acceptance criteria

- [ ] **Measure the before, on the same rig session as the after.** DGX TX and
      Jetson RX with wire counters at both ends, node `depth rx`, and the
      bracketed link capacity. Link capacity moved 3× inside one session in the
      predecessor, so an unbracketed before/after is not a comparison.
- [ ] **One remote copy of the raw depth stream**, verified by
      `ros2 topic info -v /d555/depth/image_rect_raw` showing a single remote
      subscriber, **and** by DGX TX falling by ~`221 × RTF` Mbit/s. The topic
      census alone is not proof — the byte counters are.
- [ ] **The consolidated receiver and its consumers must not be in separate
      participants**, or the fan-out is unchanged. State how the design
      guarantees this and show it in the TX counter, not in the topic graph.
- [ ] **No added staleness at the node.** A republication hop can trade a rate
      win for a latency loss, which is exactly the regression the depth-1
      history exists to prevent. Report `depth_age` p50/p95/max at the node
      before and after; the sim-lane figure to beat is the ~0.0066 s sim the
      sim-host probe reads at source.
- [ ] **RTAB-Map's `_sync` contract is unchanged.** `timestamp_fixer` runs with
      `restamp: False` because stamps already come off `/clock` and restamping
      breaks the exact synchronizer; whatever consolidation lands must not
      quietly acquire a restamp.
- [ ] **State plainly whether this closes the gap. It does not.** Combined with
      [`sim-bridge-16uc1-depth-publish`](../sim-performance/sim-bridge-16uc1-depth-publish.md)
      the census falls 608 → 277 × RTF Mbit/s, still 64 Mbit/s at RTF 0.231.
      Report the residual against the measured loss-free capacity rather than
      declaring victory on the delta.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports.

## Out of scope

- **Raising the link's capacity** —
  [`sim-bridge-link-transport-capacity`](sim-bridge-link-transport-capacity.md)
  owns it, and it is the lever that matters most.
- **Halving the bytes per frame** —
  [`sim-bridge-16uc1-depth-publish`](../sim-performance/sim-bridge-16uc1-depth-publish.md)
  owns it, DGX-side.
- **The depth QoS default.** Reliability did not resolve on the collapsed rig;
  do not churn it.
- **The real-robot lane**, which has no cross-host camera hop at all.

## Investigation pointers

- The census, the fan-out ramp, and the multicast measurement:
  [`depth-receiver-host-capacity`](../../completed/depth-receiver-host-capacity.md).
- `timestamp_fixer` is
  [`strafer_perception/timestamp_fixer.py`](../../../../source/strafer_ros/strafer_perception/strafer_perception/timestamp_fixer.py),
  wired for the sim lane in
  [`sim_bridge_support.launch.py`](../../../../source/strafer_ros/strafer_bringup/launch/sim_bridge_support.launch.py)
  with the raw depth remapped onto its aligned input.
- The node's depth subscription is `depth_topic` in
  `strafer_inference/inference_node.py`, defaulted in
  `strafer_inference/config/inference.yaml`.
- **A viz hazard worth closing while you are here:** `foxglove_bridge`
  subscribes lazily, so attaching a Foxglove client that displays depth adds a
  *fourth* image copy to the link mid-session. It cost nothing in the
  predecessor's arms only because no client was attached.
