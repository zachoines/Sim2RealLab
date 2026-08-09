# Publish 16UC1 millimetre depth from the sim bridge

**Type:** task (bridge runtime)
**Owner:** DGX
**Priority:** P2 — halves the depth stream's wire cost and closes a sim↔real
encoding disparity in one move. **Not sufficient on its own**: it is a margin
lever behind
[`sim-bridge-link-transport-capacity`](../reliability/sim-bridge-link-transport-capacity.md).
**Estimate:** S (one encoding on the publish side; the coupling and the parity
argument are the work)
**Branch:** `task/sim-bridge-16uc1-depth-publish`

## Story

As the **sim-bridge camera publisher**, I want **to publish depth as `16UC1`
millimetres rather than `32FC1` metres**, so that **the cross-host link carries
461 KB per frame instead of 921 KB, and the sim lane presents the same encoding
the real D555 does.**

## Context bundle

- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

Carried across from
[`depth-receiver-host-capacity`](../../completed/depth-receiver-host-capacity.md),
which recorded this as a cross-lane candidate it could not scope: the bridge
half is DGX-owned.

The measurement that motivates it. `/d555/depth/image_rect_raw` is 640×360
`32FC1` = **921 600 B per frame**, and one copy at the full 30 Hz sim cadence
costs **221 × RTF Mbit/s**. The DGX → Jetson path measures **20–60 Mbit/s
loss-free**. `16UC1` halves the per-copy cost to **111 × RTF Mbit/s**.

Why the halving matters more than it sounds: a frame is fragmented into UDP
datagrams and **all of them must arrive** for the sample to be delivered.
`32FC1` at 640×360 is 686 datagrams at Cyclone's 1344 B fragment size, and
measured 17.2% datagram loss produces **89% frame loss**. Halving the byte count
halves the fragment count, which raises the completion probability at any given
datagram loss rate — a second-order win on top of the first-order bandwidth one.

**The coupling is the reason to do it here rather than anywhere else.**
[`d555-depth-decode-validity`](../trained-policy/d555-depth-decode-validity.md)
already needs the node to decode `16UC1` (Z16, millimetres) because that is what
the **real** D555 publishes, and the node currently drops 100% of real depth at
the encoding gate. That brief owns the *node-side decode*; this one owns the
*bridge-side publish*. Landing both makes the two lanes agree on encoding
instead of diverging, so the sim lane stops being the only place `32FC1` exists.

Current state, for whoever picks this up: the bridge sets `img.encoding =
"32FC1"` in
[`async_camera_publisher.py`](../../../../source/strafer_lab/strafer_lab/bridge/async_camera_publisher.py),
and the code comments there already note the `32FC1` choice was made for
Jetson-side consumers.

## Acceptance criteria

- [ ] **The bridge publishes `16UC1` millimetres** on
      `/d555/depth/image_rect_raw`, and the wire cost per copy is measured at
      ~`111 × RTF` Mbit/s — measured off interface counters, not computed.
- [ ] **Sequence it behind the node-side decode, or ship them together.**
      Flipping the bridge encoding while the node still gates on `32FC1` takes
      the sim lane from degraded to dead. State explicitly which order you are
      landing in and what the intermediate state is.
- [ ] **Every existing consumer of the raw stream still works**, including
      `timestamp_fixer`'s republication and, through it, RTAB-Map,
      `depth_image_proc` and `pointcloud_to_laserscan`. `depth_image_proc`
      handles both encodings; verify rather than assume.
- [ ] **No loss of range or resolution that matters to the policy.** `16UC1`
      millimetres quantises to 1 mm and saturates at 65.535 m; the renderer's
      far clip is `D555_RENDER_FAR_CLIP_M = 50.0` and the policy clamps at
      `DEPTH_CLIP_FAR = 6.0`, so both should be comfortable — show it rather
      than assert it, including what happens to the values past 6 m that
      currently do reach the bridge.
- [ ] **Report the obs-parity consequence.** The policy's own 80×60 depth
      observation never leaves the env and is untouched; the 640×360 perception
      stream is what moves. Say explicitly which consumers see a semantic change
      and which see none.
- [ ] **State plainly whether this closes the gap. It does not.** Combined with
      [`depth-subscriber-consolidation`](../reliability/depth-subscriber-consolidation.md)
      the census falls 608 → 277 × RTF Mbit/s, still 64 Mbit/s at RTF 0.231
      against 20–60 Mbit/s of measured capacity.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit — **`bridge-runtime-invariants.md` names the depth encoding
      and will need it.** See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports.

## Out of scope

- **The node-side `16UC1` decode and the validity mask** —
  [`d555-depth-decode-validity`](../trained-policy/d555-depth-decode-validity.md)
  owns those, and this brief depends on that one rather than duplicating it.
- **Raising the link's capacity** —
  [`sim-bridge-link-transport-capacity`](../reliability/sim-bridge-link-transport-capacity.md).
- **The camera resolution.** 640×360 is locked to the real D555 native rate;
  lowering it sim-side is a deliberate sim-to-real gap, not an optimisation.
- **The sim's duplicate depth content** (50.2–67.0% consecutive-identical frames
  measured at the sim host). Separate DGX-owned defect, and a real one — it
  bounds the achievable cadence independently of any of this.
- **The color stream.** `rgb8` at 166 × RTF Mbit/s per copy is the second
  largest item on the wire and has exactly one subscriber; whether it needs to
  cross the link at all is a question for the consolidation brief, not an
  encoding change here.
