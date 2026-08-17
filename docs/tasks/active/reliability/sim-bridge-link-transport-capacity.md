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
host is on **the link**. That brief's measurement section is the evidence base.

### Current transport topology — this brief owns it

[`repo-topology.md`](../../context/repo-topology.md) states that the constraint
exists and points here for its value, because the value moves and a context
module is the wrong durability level for it. **Keep this block current; it is
what other work reads before sizing anything against the link.**

| | host | interface | media |
|---|---|---|---|
| sim host (publisher) | `gx10-d1d8`, 192.168.50.196 | `wlP9s9` (mt7925e) | **WiFi** — channel 40 / 5200 MHz, 160 MHz wide, RSSI −60 dBm, negotiated PHY 1297 Mbit/s TX / 2041 RX |
| robot host (subscriber) | `strafer-nx`, 192.168.50.161 | `enP8p1s0` | wired gigabit |

All four wired NICs on the sim host are down, so its WiFi uplink is the only
path and the narrow hop. `iw` also reports **power save on** and **txpower
3.00 dBm** on that interface. The publisher runs with no `CYCLONEDDS_URI` — the
tuned receive xml is robot-side only.

**Measured budget, 2026-08-08 — a bracketed range, not a number:**

| | start of window | ~50 min later |
|---|---|---|
| UDP loss-free to | **60 Mbit/s** | **20 Mbit/s** |
| UDP saturation goodput | ~66 | ~32 |
| TCP, 4 streams | 51.7 | 26.2 |

RSSI and negotiated PHY rate were unchanged across the drop, so the swing is
airtime contention or AP scheduling rather than signal quality. **Size against
20 Mbit/s, not 60.** A single reading is not a capacity; bracket every
throughput measurement with a capacity reading at each end, and treat runs from
different brackets as incomparable.

**What the link is asked to carry.** One 640×360 `32FC1` depth stream at the
full 30 Hz sim cadence needs `221 × RTF` Mbit/s; color `rgb8` needs
`166 × RTF`. DDS data on this path is **unicast per subscribing process**
(multicast share of robot-host receive traffic: 0.01%), so cost multiplies with
remote subscriber processes rather than being shared. The deployed census — 2
depth copies + 1 color + 2 `camera_info` — is `608 × RTF` Mbit/s, i.e.
**65–211 Mbit/s** across the RTFs this rig produces.

**Re-measured 2026-08-16 — the census arithmetic reproduces at the node, and
the degradation is a cliff.** The 2026-08-08 budget was built from iperf plus a
subscriber census; this reading takes the same quantity from the inference
node's own `cadence:` counters, where `depth rx` against `ticks timer` (a 30 Hz
sim timer, so one frame per tick *is* the contract) measures what the policy
actually receives:

| remote depth subscriber processes | depth reliability | robot-host rx | depth at the node |
|---|---|---|---|
| 1 (inference alone) | `reliable` | 26.6 Mbit/s | 354 frames / 354 ticks — **30.0 Hz sim, the full contract** |
| 2 (inference + `timestamp_fixer`) | `reliable` | 49.3 Mbit/s | **0.26 Hz sim** |
| 2 | `best_effort` | 44.8 Mbit/s | **0.00 Hz sim** (1 frame in 474 ticks) |

Two consequences for anything sizing against this link:

- **Demand does not degrade gracefully across the ceiling; it collapses.** One
  copy costs 26.6 Mbit/s, so two cost ~48.6 against a path measured at 49.3
  Mbit/s that day — oversubscribed by roughly 1.5%, and losing ~99% of frames.
  The 686-datagram framing above is why: at the margin, nearly every frame is
  missing at least one datagram. Plans that assume a proportional slowdown near
  capacity are wrong.
- **`best_effort` is not a way to buy margin, and the `reliable` default is
  correct.** Flipping the per-lane `STRAFER_DEPTH_RELIABILITY` lever made
  delivery strictly worse — zero complete frames — because a lost datagram
  discards the whole sample where `reliable` recovers it. This is the
  [`depth-qos-reliable-flip`](../../completed/depth-qos-reliable-flip.md)
  ruling holding under a condition that could have overturned it. The lever was
  returned to `reliable`.

Robot-host counters stayed at `rx_drop=0 rx_err=0 tx_drop=0` throughout, at
49.3 Mbit/s — consistent with every prior reading that nothing is lost at the
receiver.

The sim host's radio negotiated differently on this date than on 2026-08-08 —
RSSI −54 dBm, 80 MHz on 5805 MHz, PHY 960.7 Mbit/s TX / 1200.9 RX, against the
table's −60 dBm / 160 MHz / 5200 MHz. Achieved goodput was ~49 Mbit/s either
way, which is further evidence the constraint is airtime or scheduling rather
than signal quality, and another reason to size against the low end.

### The load-bearing findings behind it

- **The path delivers ~5% of its negotiated PHY rate**, and neither the low
  `txpower` nor `power_save` has been tested as the reason. Both are free to
  change and both are first arms below.
- **Neither host drops a packet.** Zero UDP `InErrors` / `RcvbufErrors` /
  NIC `rx_drop` on the robot host, zero `SndbufErrors` / `tx_drop` on the sim
  host, in every measured condition. The robot host idles at 9.8% CPU and 0%
  GPU while starving, so there is no receive-side headroom to buy.
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
- [ ] **Record RTF concurrently with every wire-counter reading.** The
      predecessor could not decompose its attached-vs-detached wire change
      (observed +194% against a census-predicted +57%) because it took byte
      counters without a paired RTF, leaving the residual split between an RTF
      difference and RELIABLE retransmission. Wire demand is linear in RTF and
      RTF is not constant on this rig — it moved 0.347 → 0.229 across one
      subscriber ramp. A wire figure without its RTF cannot be compared to
      another wire figure.
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
