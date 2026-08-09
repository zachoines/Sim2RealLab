# Give the sim-bridge lane a link that can carry its camera streams

**Type:** task (rig transport)
**Owner:** Jetson + DGX
**Priority:** P1 — it is the **binding constraint** on the depth lane. Measured
2026-08-08: the deployed subscriber census needs 65–211 Mbit/s and the path
delivers 20–60 Mbit/s loss-free, so the inference node receives **zero** complete
depth frames in the deployed configuration. Nothing downstream of this — the
achievable-cadence ceiling, `depth_age` at inference, the anchoring re-run —
can be measured until it moves.
**Estimate:** S–M (a cable and two `iw` settings are the whole first arm; the
re-measure and the inherited post-fix plan are the rest)
**Branch:** `task/sim-bridge-link-transport-capacity`

## Story

As the **operator running a depth policy against the sim bridge**, I want **the
link between the two hosts to carry the camera streams the bridge already
publishes**, so that **the policy is limited by what the sim renders and what
the node can consume, rather than by an uplink that discards 75% of the packets
before they arrive.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

Filed by
[`depth-receiver-host-capacity`](../../completed/depth-receiver-host-capacity.md),
which closed premise-refuted: the fault it was written to locate on the receiver
host is on **the link**. That brief's measurement section is the evidence base;
the load-bearing facts are:

- **The DGX transmits every camera frame over WiFi.** All four of its wired NICs
  are down (`wlP9s9`, mt7925e, channel 40 / 160 MHz, RSSI −60 dBm, negotiated
  PHY 1297 Mbit/s TX). The Jetson is already on wired gigabit. The DGX uplink is
  the only path and the narrow hop.
- **Measured capacity DGX → Jetson: loss-free to 60 Mbit/s early in a session,
  and to only 20 Mbit/s fifty minutes later**, saturating at 66 / 32 — a ~3×
  swing with RSSI and negotiated PHY rate unchanged, so this is airtime
  contention or AP scheduling, not signal quality. It delivers ~5% of its
  negotiated PHY rate.
- **`iw` also reports `power_save on` and `txpower 3.00 dBm`** on that
  interface. Neither has been tested as a cause; both are free to change.
- **Neither host drops a packet.** Zero UDP `InErrors` / `RcvbufErrors` /
  NIC `rx_drop` on the Jetson, zero `SndbufErrors` / `tx_drop` on the DGX, in
  every arm. The Jetson idles at 9.8% CPU and 0% GPU while starving.
- **Why the margin has to be generous.** A 640×360 `32FC1` frame is 921 600 B =
  **686 UDP datagrams**, and all must arrive. Measured: 17.2% datagram loss is
  **89% frame loss**. The budget to size against is the *loss-free* rate, not
  the saturation goodput.

This **discharges the transport half of mode 3** in
[`enriched-lane-rig-stability`](enriched-lane-rig-stability.md), whose ask —
"move the DDS traffic to wired and re-measure" — was applied at the Jetson end
only, and whose 251.6 Mbit/s powerline figure describes neither current end of
the path.

**Sequencing note.** The two sibling levers filed alongside this one —
[`depth-subscriber-consolidation`](depth-subscriber-consolidation.md) and
[`sim-bridge-16uc1-depth-publish`](../sim-performance/sim-bridge-16uc1-depth-publish.md)
— reduce demand rather than raise supply, and **together they do not close the
gap**: 608 → 277 × RTF Mbit/s still needs 64 Mbit/s at RTF 0.231. This brief is
the one that changes the order of magnitude. The other two buy margin on top of
it and should not be treated as substitutes for it.

## Acceptance criteria

- [ ] **Re-measure the current path first, and bracket it.** Capacity moved 3×
      inside one session, so a single reading is not a capacity. Report
      loss-free rate and saturation goodput at the start and end of every
      measurement window, and treat arms taken in different brackets as
      incomparable. Method and tooling in the predecessor's investigation
      pointers.
- [ ] **Try the two free levers before the cable**, one at a time, each with a
      bracketed re-measure: `iw dev wlP9s9 set power_save off`, and raise
      `txpower` from 3.00 dBm. Report whether either moves the loss-free rate.
      They are cheap, reversible, and if one of them explains the ~20× PHY
      deficit the cable may be unnecessary.
- [ ] **Move the DGX's DDS traffic to wired** and re-measure. Report the
      loss-free rate on the wired path against the 65–211 Mbit/s census
      requirement, and state whether the census now fits **with margin at the
      worst RTF the rig produces**, not just at the best.
- [ ] **Re-run the predecessor's discriminating arms on the new path**: the
      subscriber census unchanged, node attached, with wire counters at both
      ends. The node's `depth rx` must go from 0 to a real number, and the
      arm must record `depth_age` at the node.
- [ ] **Do not switch transports mid-session** — it changes latency and jitter
      and confounds any arm-to-arm comparison. This is the rig-stability
      brief's standing rule and it applies to this work most of all.
- [ ] Keep wired as the SSH path regardless, so a transport wedge no longer
      costs rig control.

### Inherited post-fix plan — this brief owns it, because it is the one that moves the ceiling

Moved here from
[`depth-receiver-host-capacity`](../../completed/depth-receiver-host-capacity.md),
which could not run any of it: no frame reached the node, so SLAM never seeded,
`map→base_link` never published, and the watchdog held every tick.

- [ ] **Produce the achievable-cadence ceiling** the setpoint rule consumes.
      The rule itself is unchanged and lives in
      [`depth-qos-reliable-flip`](../../completed/depth-qos-reliable-flip.md):
      Arm D first; then ≥ 27 / 20–27 / < 20 Hz. **Report the figure as a number
      with its spread, never as a verdict against 30 Hz.**
      Two things must be reported alongside it or the ceiling is not
      interpretable: the **RTF** it was measured at (the wire cost is
      `608 × RTF` Mbit/s, so the ceiling is RTF-dependent by construction), and
      the **consecutive-identical depth content**, measured at 50.2–67.0% at the
      sim host — a perfectly delivered 30 Hz sim stream carries only ~10–15 Hz
      of *new* content, and that is the DGX-owned duplicate-content defect, not
      a transport one.
- [ ] **`depth_age` at inference.** The instrument ships; the reading has never
      been obtainable.
- [ ] **Re-run at least one arm of the anchoring re-validation goal set** and
      report whether the advance numbers move at all. **Whether they move is a
      finding either way — do not pre-commit to an expectation.**

- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports.

## Out of scope

- **The depth QoS default.** Given its first real test on a collapsed rig,
  reliability did not resolve: the two QoS cells differed by less than the
  round-to-round spread of the same cell. `reliable` stays set on the sim lanes
  — it is the only thing that recovers a fragment-lossy 686-datagram sample —
  and `best_effort` stays in `inference.yaml`. Do not churn it.
- **CPU pinning / affinity.** Retired by the measurement, not deferred: the
  Jetson runs 9.8% mean CPU with the node attached, 0% GPU, and is *busier* with
  it detached. There is no contention to pin away from.
- **The sim's duplicate depth content** and the bridge's own render rate — both
  DGX-owned, and neither is a transport defect.
- **Reducing the offered load.** The two sibling briefs own that.

## Investigation pointers

- Method, tooling and per-arm data:
  [`depth-receiver-host-capacity`](../../completed/depth-receiver-host-capacity.md)'s
  measurement section. Tools in `~/strafer_v2_validation/tools/`
  (`capacity_probe.py` parameterised by reliability and subscriber count,
  `wire_accounting.py` for both-end counters, `capacity_suite.sh` for
  interleaved arms with the build tells asserted, `burst_capacity.py` for the
  frame-shaped loss test, `suite_table.py` to reduce a run).
- `iperf3` is not installed on the DGX (it ships iperf2). Both hosts are
  aarch64, so staging the Jetson's `iperf3` plus `libiperf.so.0` and
  `libsctp.so.1` into `/tmp` on the DGX works and is what the predecessor did.
- The node's build tells are load-bearing and the shipped `strafer-gpu` image
  does not carry them — see the predecessor's note on building **after** the
  force-recreate.
