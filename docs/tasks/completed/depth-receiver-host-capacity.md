# Locate the depth receiver-host capacity ceiling

**Status:** Shipped 2026-08-08 in `4ad29a8` (Jetson) — **measurement delivered,
premise refuted.** Every measurement acceptance criterion read out, and the
answer renames the fault: it is not the receiver host. The Jetson drops nothing
at any layer and idles at 9.8% CPU while starving; the constraint is the **link
between the hosts** — a DGX WiFi uplink measured at 20–60 Mbit/s loss-free
carrying a subscriber census that needs 65–211 Mbit/s, unicast-multiplied once
per subscribing process. Levers are sized against the numbers; per this brief's
own rule **none is adopted here**, and all fix-work is filed as the follow-ups
below. This is the fourth mechanism proposed for this defect and the fourth
displaced by a measurement designed to discriminate.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/196
**Follow-ups:**
[`sim-bridge-link-transport-capacity`](../active/reliability/sim-bridge-link-transport-capacity.md)
— the P1 lever, the only one that changes the order of magnitude; also inherits
this brief's post-fix plan (achievable-cadence ceiling, `depth_age` at
inference, the anchoring re-run), because it is the brief that moves the
ceiling.
[`depth-subscriber-consolidation`](../active/reliability/depth-subscriber-consolidation.md)
— removes one of two remote depth copies.
[`sim-bridge-16uc1-depth-publish`](../active/sim-performance/sim-bridge-16uc1-depth-publish.md)
— halves the per-copy cost, DGX-side.

**Type:** investigation → task (deploy runtime)
**Owner:** Jetson
**Priority:** P1 — it costs 4–6× of the depth cadence on every mission, and it
is now the *only* surviving mechanism for the shortfall three briefs have chased.
**Estimate:** M (measurement first; levers scoped only after it names the fault)
**Branch:** `task/depth-receiver-host-capacity`

> **STATUS 2026-08-08 — the measurement has read out, and it renames the fault.**
> Every measurement acceptance criterion below is met. The bottleneck is **not
> the receiver host**: the Jetson drops nothing at any layer and idles at 9.8%
> CPU while starving. It is the **link between the hosts** — a DGX WiFi uplink
> measured at 20–60 Mbit/s loss-free, carrying a subscriber census that needs
> 65–211 Mbit/s, unicast-multiplied once per subscribing process. Numbers in
> [Rig measurement](#rig-measurement-2026-08-08--the-constraint-is-the-link-and-neither-host-loses-the-frames);
> levers are sized but, per this brief's own rule, **none is adopted here**. The
> title's "receiver-host" framing is retained as the brief's history; the
> successor scope is transport, not host.

## Story

As the **operator running a depth policy against the sim bridge**, I want **the
depth frames the bridge already publishes to survive the trip to the node while
the rest of the stack is running**, so that **the policy sees the ~28.5 Hz sim
the host demonstrably receives when nothing else is attached, instead of the
5–8 Hz it sees under load.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/bridge-runtime-invariants.md](../context/bridge-runtime-invariants.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)

## Context

[`depth-qos-reliable-flip`](depth-qos-reliable-flip.md) closed
premise-refuted and handed this over. Its measurement, 2026-08-06, sim-bridge
lane, RTF 0.107–0.112 **unchanged across every condition** so the sim never
slowed:

| condition | probe rate |
|---|---|
| inference node **stopped** | **28.4 Hz sim** (repeats 26.9, 30.2) |
| node attached, `reliable` | 8.2 Hz sim |
| node attached, `best_effort` | 5.0 Hz sim |

The probe is an independent subscriber on the node's exact QoS (`KEEP_LAST`
depth 1). `best_effort` is *worse*, so the loss tracks the node's **presence**,
not its subscription policy. An interleaved four-arm A/B at the node
(25.0 / 22.0 / 11.1 / 17.7 Hz sim) has a within-arm spread larger than its
between-arm difference — the rig's own variance exceeds any QoS treatment.

**This is the third mechanism proposed for one defect**, each displaced by a
better-designed measurement: queue overwrite → fragment loss in transit → host
capacity. The completed brief tabulates what killed the first two. The standing
lesson: *a rate measured at one subscriber cannot attribute a loss* — only
paired arms with one variable moved can.

**What is not yet known: which host.** One piece of evidence favours the
Jetson — while depth was collapsed, `/d555/color/image_raw` (same publisher,
same worker thread, same size class) kept flowing at ~3 Hz wall, and a stalled
publisher would have killed both. That is suggestive, not conclusive, and the
first job below settles it before any lever is chosen.

Prior art this supersedes and absorbs: the two follow-ups filed inside
[`depth-reception-reliability`](depth-reception-reliability.md)
— the **frame-skip-0 whole-Jetson capacity ceiling** and the **`timestamp_fixer`
still-RELIABLE perception receive QoS** named there as its lever — plus items
2–4 of the 2026-08-03 cadence addendum's node-consumption list. Those were
findings inside a shipped brief with no active owner; this brief owns them.

## Rig measurement, 2026-08-08 — the constraint is the link, and neither host loses the frames

Sim-bridge lane, `ProcRoom-Enriched-v0`, `--decimation 4 --render-interval 4`,
bridge cadence contract `publish 30.00 Hz sim | frame_skip=0 (derived, derived 0)
| bridge tick 30.00 Hz | renders/tick 1.00`. The standing pre-launch resident-PID
check was clean, and **the render stall did not recur**: the bridge warmed on
~185% CPU / ~50% GPU and produced first frames inside ~12 min, so the
rig-stability brief's forensics protocol was not triggered. Session tooling and
per-arm output in `~/strafer_v2_validation/` (`tools/capacity_probe.py`,
`tools/wire_accounting.py`, `tools/capacity_suite.sh`, `tools/burst_capacity.py`,
`tools/suite_table.py`; `raw/capacity_20260808_210051/`).

**Both build tells hold on every arm.** The `strafer-gpu` image on this host is
`revision=25305a4c4103`, which **predates `e97c4d6`** — the shipped image has
neither `STRAFER_DEPTH_RELIABILITY` nor `depth_age`, exactly the trap this brief
warns about. Arms ran on the dev overlay with an in-container
`colcon build --symlink-install --packages-select strafer_inference` **after**
each `up -d --force-recreate`, then `docker restart`. Each arm asserted the
startup `depth_reliability overridden to '<rel>' via STRAFER_DEPTH_RELIABILITY`
line and a `depth_age` field in the `cadence:` line before its numbers were kept.

### 0. This is not the rig the brief was written against

| | 2026-08-02 (rig-stability brief) | 2026-08-08 |
|---|---|---|
| Jetson link | `mt7921u` USB WiFi dongle, 126.9 Mbit/s | **wired GigE** `enP8p1s0`, 192.168.50.161 |
| DGX link | wired | **WiFi** `wlP9s9` (mt7925e), 192.168.50.196 |
| narrow hop | the dongle | **the DGX's WiFi uplink** |

Every wired NIC on the DGX is down, so its WiFi is the only path. The
rig-stability brief's ask — *move the DDS traffic to wired* — is applied at the
Jetson end and **not** at the DGX end, which is the end that transmits the
camera streams; its 251.6 Mbit/s powerline figure does not describe this path.
Channel 40 (5200 MHz) at 160 MHz, RSSI −60 dBm, negotiated PHY 1297 Mbit/s TX.
Also recorded because they bound how much is recoverable: **power save is on**
and **txpower reads 3.00 dBm**. The DGX publisher runs with **no
`CYCLONEDDS_URI`** — the tuned xml is Jetson-side only.

### 1. AC1's gate — the sim host has every frame the Jetson never sees

Probe run **on the sim host** beside the bridge, on the node's exact QoS
(`KEEP_LAST` depth 1):

| condition | sim-host probe | RTF |
|---|---|---|
| no Jetson subscribers at all | **30.12 Hz sim** (260 frames / 25 s) | 0.347 |
| full Jetson stack attached, node reading `depth rx=0` | **30.28 Hz sim** (104 frames / 15 s), age at receipt p50 0.0066 s sim | 0.231 |

The bridge supplies the full training cadence, unchanged, at the moment the
Jetson node is receiving nothing. By the criterion's own rule that is "the wire
or the Jetson", and the publisher is exonerated.

### 2. Of those two — the wire. Neither host drops anything.

Across **every arm this session**, on both hosts:

```
NX  UDP InErrors = 0   RcvbufErrors = 0   NIC rx_drop = 0   rx_fifo = 0
DGX UDP SndbufErrors = 0                  NIC tx_drop = 0
```

Not one frame is lost in either host's network stack. The Jetson's tuned receive
path (16 MB socket buffer, defrag headroom 32, the `rmem_max` drop-in) has
headroom it never uses — the packets do not arrive to be buffered. The DGX's
station counters agree the frames left and were acknowledged: over a 20 s live
sample, +273 239 TX packets, +7 585 retries (2.8%), **zero** `tx failed`. The
loss is in the path between the two NICs.

A control sits inside the same arms: `/odom`, `/tf`, `/joint_states` and
`/clock` all arrive from the same publisher, over the same link, between the
same participants, while depth gets nothing
(`stale_sources[depth=735 … joint_states=3 odom=1 tf=2]`). **Message size is the
discriminator — not discovery, not QoS, not configuration.**

### 3. Measured link capacity — the number the brief was missing

`iperf3` 3.9 staged onto the DGX (which ships only iperf2), DGX → Jetson, link
otherwise idle. **Bracketed**, because it turned out not to be a constant:

| test | early (20:35) | late (21:20, post-suite) |
|---|---|---|
| TCP, 4 streams | 51.7 Mbit/s | 26.2 |
| UDP @ 20 Mbit/s | 19.9, **0/17 856 lost** | 19.9, **0/17 849 lost** |
| UDP @ 40 | 39.8, **0/35 711** | 32.5, 12% lost |
| UDP @ 60 | 59.7, **0/53 567** (repeated) | 31.8, 43% lost |
| UDP @ 80 | 65.3, 15% | 32.7, 56% |
| UDP @ 100 | 66.3, 31% | — |

**The loss-free capacity fell from 60 Mbit/s to 20 Mbit/s over ~50 minutes**,
with RSSI (−60 dBm) and negotiated PHY rate (1297 Mbit/s) unchanged across the
drop — so this is airtime contention or AP scheduling, not signal quality. Two
consequences: the honest capacity figure is a **range, 20–60 Mbit/s**, and arms
are only comparable inside a bracket. This instability is also a candidate
explanation for the predecessor's unresolvable four-arm A/B, whose within-arm
spread (11.1–25.0 Hz) exceeded its between-arm difference — a link whose
capacity wanders 3× produces exactly that.

### 4. Why a few percent of packet loss destroys nearly all frames

A 640×360 `32FC1` frame is 921 600 B — **686 UDP datagrams** at Cyclone's 1344 B
fragment size — and every one must arrive for the sample to be delivered, or it
must be retransmitted. Emulating that burst shape directly
(`tools/burst_capacity.py`, matched mean rates against the smooth `iperf3` runs
above, measured post-drift):

| offered mean | datagram loss | **frames complete** |
|---|---|---|
| 20 Mbit/s | 0.0% | **100%** (50/50) |
| 40 Mbit/s | 17.2% | **11.1%** (11/99) |

So burst shape is not itself the problem — datagram loss at 40 Mbit/s is much
the same bursty as smooth. The **amplification** is: 17% datagram loss is 89%
*frame* loss. The operative budget is therefore the **loss-free** rate, not the
saturation goodput, and this is also the mechanism behind the predecessor's
otherwise anomalous finding that `best_effort` was *worse* than `reliable` —
reliable retransmits the missing fragments and recovers samples that best-effort
discards whole.

### 5. Subscriber census on the raw 640×360 stream

`ros2 topic info -v`, deployed lane. Publisher on all four:
`strafer_sim_bridge_camera_publisher`, RELIABLE `KEEP_LAST` 10.

| raw topic | payload / frame | remote subscribers | their QoS |
|---|---|---|---|
| `/d555/depth/image_rect_raw` | 921 600 B (`32FC1`) | `strafer_inference`; `timestamp_fixer` | RELIABLE `KEEP_LAST` 1; RELIABLE `KEEP_LAST` 10 |
| `/d555/color/image_raw` | 691 200 B (`rgb8`) | `timestamp_fixer` | RELIABLE `KEEP_LAST` 10 |
| `/d555/depth/camera_info` | ~0.4 KB | `timestamp_fixer` | RELIABLE `KEEP_LAST` 10 |
| `/d555/color/camera_info` | ~0.4 KB | `timestamp_fixer` | RELIABLE `KEEP_LAST` 10 |

**Costing nothing on the wire:** rtabmap, `depth_image_proc`'s
`depth_to_pointcloud` and `pointcloud_to_laserscan` all consume
`timestamp_fixer`'s `/d555/**_sync` republications, which are Jetson-local. The
brief's census list anticipated "rtabmap's aligned consumer" as a wire cost — it
is not one, because `timestamp_fixer` is already the consolidation point for the
SLAM chain. `foxglove_bridge` subscribes lazily and had no client attached;
**attaching a Foxglove client that displays depth adds a fourth image copy**,
which is an operator hazard rather than a neutral act.

Raw-stream fan-out is therefore **three image copies**: depth ×2, color ×1.

### 6. The fan-out is unicast, and it multiplies per *process*

Two independent measurements, neither assumed. **Multicast share of Jetson
receive traffic: 0.01%** (3 of 54 106 packets over 20 s of live depth traffic).
And DGX TX against remote subscriber-process count (probe containers, RELIABLE,
40–45 s arms, measured in the early bracket):

| remote subscriber processes | DGX TX | Jetson RX | packet shortfall | measured probe |
|---|---|---|---|---|
| 0 | 0.01 Mbit/s | 0.01 | — | — |
| 1 | 67.5 | **61.0** | 9.2% | 11.72 Hz sim |
| 2 | 102.0 | 58.5 | 41.4% | **0 frames / 45 s** |
| 3 | 127.6 | 34.4 | 71.1% | **0 frames / 40 s** |

Offered load rises monotonically with subscriber count while delivered goodput
*falls* — congestion collapse, and the signature of a writer sending an
independent copy per participant; multicast would have held TX flat. Peak
delivered across the whole session was 61.0 Mbit/s, within 2% of the
independently measured 60 Mbit/s loss-free ceiling — two unrelated instruments
agreeing on where the path tops out.

The **per-process** qualifier is what the consolidation lever turns on: readers
sharing a participant share its receive locators and one copy serves all of
them. The deployed stack shares nothing — every service is its own container,
its own participant, its own unicast locator.

### 7. Traffic accounting against the measured capacity

One copy of a stream at the full 30 Hz sim cadence, as a function of RTF:

- depth `32FC1`: 921 600 × 8 × 30 × RTF = **221 × RTF** Mbit/s
- color `rgb8`: 691 200 × 8 × 30 × RTF = **166 × RTF** Mbit/s

The census (2 depth + 1 color + 2 `camera_info`) needs **≈ 608 × RTF Mbit/s**:

| RTF | census requirement | vs 60 Mbit/s (best measured) | vs 20 Mbit/s (worst) |
|---|---|---|---|
| 0.107 — the 2026-08-06 session | 65.1 Mbit/s | **1.09× over** | 3.3× over |
| 0.231 — this session, stack attached | 140.5 | 2.34× over | 7.0× over |
| 0.347 — this session, no remote subscribers | 211.1 | 3.52× over | 10.6× over |

**The link is the binding constraint.** It binds at every RTF this rig has ever
produced, including the most favourable on record, and against the most
favourable capacity measured.

**On this brief's own "~68 Mbit/s per-stream requirement".** That figure is not
per-stream — it is the *aggregate census* requirement at RTF ≈ 0.11, and it is
where the rig-stability brief's "live traffic requirement of only 67.8 Mbit/s"
came from. Read correctly it was never comfortable: it already exceeded this
path's capacity. What was missing was a capacity measurement on the path that
actually carries the traffic; the word "only" was doing the damage.

### 8. Host CPU, node attached vs detached — the Jetson idles while it starves

`tegrastats`, 60 s each, rest of the stack up:

| condition | CPU mean | busiest core | GPU | RAM | DGX TX | NX RX | shortfall |
|---|---|---|---|---|---|---|---|
| node **attached** | **9.8%** | 21.0% | 0% | 5390 MB | 125.0 Mbit/s | 24.6 | 75.0% |
| node **detached** | **15.5%** | 30.0% | 0% | 4348 MB | 42.5 | 21.6 | 44.5% |

The host is **busier without the node**, because with the node gone the fan-out
drops and depth actually flows, giving the SLAM chain real work. Across all
eight suite arms the peak busiest core was 62% and GPU was 0% throughout, with
the CPU largely parked at its 729 MHz idle floor. **There is no whole-Jetson CPU
ceiling here.** What the node's presence costs is not CPU: it is a second
unicast copy, which triples what the DGX puts on the wire (42.5 → 125.0 Mbit/s)
while what arrives barely moves (21.6 → 24.6).

**Decode and inference halves could not be separated**, for the same structural
reason the predecessor hit and one worse: `inferences=0` *and* `depth rx=0` in
every arm, so the node performed neither decode nor inference. There is no
inference cost to measure because no frame ever reached the node, and SLAM
cannot seed while depth is degraded, so `map→base_link` never published.

### 9. Single- vs multi-subscriber at both reliabilities, on the collapsed regime

Eight arms, 60 s measured after 25 s settle, run forward then in reverse order so
a warm-up trend would show as drift down the column rather than as a treatment
effect. `depth rx` is the node's own counter delta over the window.

| arm (run order) | subs | node QoS | depth rx | DGX TX | NX RX | shortfall | CPU |
|---|---|---|---|---|---|---|---|
| r1 | multi | reliable | **0** | 119.8 | 23.2 | 75.0% | 21.5% |
| r1 | multi | best_effort | **0** | 115.3 | 19.4 | 77.5% | 19.3% |
| r1 | single | best_effort | **0** | 50.0 | 16.7 | 58.1% | 16.7% |
| r1 | single | reliable | **0** | 52.9 | 17.6 | 58.1% | 20.8% |
| r2 | single | reliable | **0** | 46.8 | 15.8 | 57.6% | 16.0% |
| r2 | single | best_effort | **0** | 50.4 | 18.3 | 55.5% | 17.7% |
| r2 | multi | best_effort | **0** | 112.8 | 17.6 | 78.8% | 22.7% |
| r2 | multi | reliable | **0** | 125.8 | 21.1 | 78.1% | 18.6% |

Cell means over the two rounds:

| cell | DGX TX | NX RX | shortfall |
|---|---|---|---|
| single × `reliable` | 49.8 | 16.7 | 57.9% |
| single × `best_effort` | 50.2 | 17.5 | 56.8% |
| multi × `reliable` | 122.8 | 22.2 | 76.5% |
| multi × `best_effort` | 114.0 | 18.5 | 78.2% |

Three read-outs:

1. **Subscriber count is the variable.** It moves offered load 2.4× (50 → 118
   Mbit/s) and shortfall 57% → 77%.
2. **Reliability is not.** Within each subscriber count the two QoS cells differ
   by less than the round-to-round spread of the same cell. Given a real test on
   the collapsed regime — which the completed brief never had, having compared
   them only on an uncollapsed rig and on container loopback — `reliable` vs
   `best_effort` still does not resolve. It remains the right default for the
   reason §4 gives (retransmission is the only thing that recovers a
   fragment-lossy 686-datagram sample), but it is not a lever on this shortfall.
3. **`depth rx = 0` in all eight arms.** In the deployed configuration, at this
   session's link capacity, the node receives *no complete depth frame at all* —
   not a degraded rate. Round 2 reproduces round 1, so this is not warm-up.

Recorded for the ceiling work, not scoped here: consecutive-identical depth
content measured **50.2–67.0%** at the sim host, so even a perfectly delivered
30 Hz sim stream carries only ~10–15 Hz of new content. That is the DGX-owned
duplicate-content defect this brief lists as out of scope, but the
achievable-cadence ceiling cannot be stated without it.

### What this displaces

The brief's title names the receiver host. The measurement says the receiver
host is innocent: it drops nothing at any layer, and it is *less* busy when the
node is attached. "Receiver-host capacity" was the fourth mechanism proposed for
this defect and it is displaced the same way the first three were — by a
measurement designed to discriminate.

| # | mechanism | displaced by |
|---|---|---|
| 1 | queue overwrite | own callback group since `70323c8`; `timer_deadline_missed` 0 in replay |
| 2 | fragment loss in transit at a best-effort reader | an independent *reliable* probe read ~28.5 Hz sim with the node stopped |
| 3 | receiver-**host** capacity | zero receiver-side drops at every layer in every arm; Jetson CPU 9.8% attached vs 15.5% detached, GPU 0% |
| 4 | **link capacity between the hosts** — a 20–60 Mbit/s path carrying a 65–211 Mbit/s census, unicast-multiplied per subscriber | *standing* |

The lesson the predecessor paid for still holds and did the work again here: a
rate measured at one subscriber cannot attribute a loss. What settled it was
counters at **both ends of the link** plus an independent capacity measurement —
neither of which any single-subscriber rate could have supplied.

## Acceptance criteria — measurement first, no fixes until it reads out

- [x] **Which host loses the frames.** → **Neither.** Sim-host probe reads
      **30.28 Hz sim** with the full stack attached and the node at `depth rx=0`
      (30.12 Hz sim with no Jetson subscriber at all), so the publisher is
      exonerated. Within "the wire or the Jetson", the Jetson is exonerated too:
      **zero** UDP `InErrors`, zero `RcvbufErrors`, zero NIC `rx_drop`/`rx_fifo`
      in every arm, and zero `SndbufErrors`/`tx_drop` on the DGX. The loss is in
      the path between the two NICs — see
      [§1](#1-ac1s-gate--the-sim-host-has-every-frame-the-jetson-never-sees) and
      [§2](#2-of-those-two--the-wire-neither-host-drops-anything).
- [x] **Subscriber census + traffic accounting** on the raw 640×360 stream.
      → **The link is the binding constraint.** Census is **three image copies**
      (depth ×2 — `strafer_inference` + `timestamp_fixer`; color ×1 —
      `timestamp_fixer`); rtabmap and the scan chain consume Jetson-local
      `_sync` republications and cost nothing on the wire. Fan-out is **unicast
      per process** (multicast share 0.01%; DGX TX 67.5 → 102.0 → 127.6 Mbit/s
      for 1 → 2 → 3 subscriber processes). The census needs **608 × RTF
      Mbit/s** = 65–211 Mbit/s across the RTFs this rig produces, against a
      **measured 20–60 Mbit/s** loss-free capacity. See
      [§3](#3-measured-link-capacity--the-number-the-brief-was-missing),
      [§5](#5-subscriber-census-on-the-raw-640360-stream),
      [§6](#6-the-fan-out-is-unicast-and-it-multiplies-per-process),
      [§7](#7-traffic-accounting-against-the-measured-capacity).
      **The brief's "~68 Mbit/s per-stream" figure was mislabelled** — it is the
      aggregate census requirement at RTF ≈ 0.11, and it already exceeded this
      path's capacity.
- [x] **Host CPU, node attached vs detached** (`tegrastats`). → **No host CPU
      ceiling.** Attached **9.8%** mean / 21% busiest core; detached **15.5%** /
      30%; GPU 0% in both; peak busiest core across all eight suite arms 62%.
      The host is *busier without the node*, because depth then flows and the
      SLAM chain has work. What the node's presence costs is a second unicast
      copy — DGX TX 42.5 → 125.0 Mbit/s — not CPU.
      **Decode and inference halves could not be separated:** `inferences=0`
      *and* `depth rx=0`, so the node performed neither. See
      [§8](#8-host-cpu-node-attached-vs-detached--the-jetson-idles-while-it-starves).
- [x] **Single- vs multi-subscriber arms at both reliabilities**, on the
      collapsed regime. → Eight arms.
      **Subscriber count is the variable** (offered load 50 → 118 Mbit/s,
      shortfall 58% → 77%); **reliability is not** — the two QoS cells differ by
      less than the round-to-round spread of the same cell, so `reliable` vs
      `best_effort` still does not resolve even given the real test on a
      collapsed rig. `depth rx = 0` in **all eight arms**. See
      [§9](#9-single--vs-multi-subscriber-at-both-reliabilities-on-the-collapsed-regime).
- [x] Arms are **paired and interleaved**, with the warm-up trend controlled —
      run forward then in reverse order, each cell twice. Round 2 reproduces
      round 1, so no warm-up trend is masquerading as a treatment effect. The
      link capacity itself was **bracketed** before and after the suite, which
      is how the 60 → 20 Mbit/s drift was caught.
- [x] Every arm records that the node under test is the intended build, by both
      tells: the startup `depth_reliability overridden to …` line and a
      `depth_age` field in the `cadence:` line. → Asserted per arm by
      `capacity_suite.sh` before its numbers are kept. The image on this host
      (`revision=25305a4c4103`) **predates `e97c4d6`** and carries neither, so
      every arm ran the dev overlay with an in-container `colcon build` **after**
      the force-recreate, then `restart`.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports.

## Candidate levers — now sized against the measurement

Still candidates, not conclusions: nothing here is adopted in this brief. But
the measurement has read out, so each carries the number it has to beat. The
budget to clear is **608 × RTF Mbit/s of census demand against 20–60 Mbit/s of
loss-free path** — and because a frame is 686 datagrams, the target is the
*loss-free* rate, not the saturation goodput.

- **~~CPU pinning / affinity~~ — dead, and it should not be re-proposed.** The
  host-contention trigger it rested on has *not* fired: the Jetson runs at 9.8%
  mean CPU with the node attached, 0% GPU, and is *busier* with the node
  detached. There is no contention to pin away from. This retires the
  "frame-skip-0 whole-Jetson capacity ceiling" follow-up this brief absorbed.
- **Move the DGX onto wired — the largest lever, and it is the one the
  rig-stability brief already asked for.** That ask was applied at the Jetson
  end only; the DGX still transmits every camera frame over WiFi, and every
  wired NIC on it is down. This is the narrow hop, and it is the difference
  between a 20–60 Mbit/s path and a gigabit one. Cross-host, so it needs the
  DGX lane. Cheaper first probes on the same path, both free: **WiFi power save
  is on** and **txpower reads 3.00 dBm** on `wlP9s9`.
- **Subscriber consolidation** — one host-side receiver of the raw stream that
  republishes locally. Now quantified: it removes **one of two** depth copies,
  i.e. **221 × RTF Mbit/s**, and the consolidation must be *per DDS participant*
  (readers sharing a participant already share one copy — the deployed stack
  shares nothing). `timestamp_fixer` already plays this role for the SLAM chain;
  pointing the inference node at a local republication rather than at the raw
  remote topic is the same move.
- **Halve the wire cost: publish 16UC1 millimetres from the sim bridge**
  (~921 KB → ~461 KB per frame), i.e. **221 → 111 × RTF Mbit/s** per depth copy.
  Still a **cross-lane candidate**, flagged not scoped: the bridge half is
  DGX-owned. It couples with
  [`d555-depth-decode-validity`](../active/trained-policy/d555-depth-decode-validity.md)
  — the 16UC1 decode path the node already needs for real hardware would let the
  sim lane carry half the bytes *and* close the encoding disparity between the
  two lanes in one move. Raise it with the DGX lane rather than implementing it
  here.
- **No single lever above closes the gap at every RTF.** Consolidation *and*
  16UC1 together take the census from 608 to 277 × RTF Mbit/s — still 64 Mbit/s
  at RTF 0.231, i.e. above the good bracket and 3× the bad one. The transport
  lever is the one that changes the order of magnitude; the others buy margin on
  top of it.

## Post-fix plan (only once the ceiling moves)

- **Re-run at least one arm of the anchoring re-validation goal set** and report
  whether the advance numbers move at all. Whether they move is a finding either
  way — do not pre-commit. This item moves here from the closed predecessor.
- **Produce the achievable-cadence ceiling** the setpoint rule consumes. The
  rule itself (Arm D first; then ≥ 27 / 20–27 / < 20 Hz) is unchanged and lives
  in [`depth-qos-reliable-flip`](depth-qos-reliable-flip.md);
  only its ceiling's owner moved here. Note the good news it starts from: the
  host already receives ~28.5 Hz sim with the node stopped, **above the 27 Hz
  threshold**, so a successful fix keeps the setpoint at 30 and the `< 20 Hz`
  branch never opens.
- **`depth_age` at inference**, which the predecessor shipped the instrument for
  and never obtained — no inference ran, because SLAM cannot seed a map while
  depth is degraded, so `map→base_link` never published and the watchdog held
  every tick.

## Out of scope

- The depth QoS default. `reliable` stays set on the sim lanes and
  `best_effort` in `inference.yaml`; it is a one-line env flip once the
  multi-subscriber arms above give it a real test. Do not churn it on inference.
- The sim's duplicate depth content, and the bridge's own render rate — both are
  DGX-owned and neither is implicated by the arms above.
- Real-robot depth QoS, which stays an *Adjacent* item inside
  [`d555-depth-decode-validity`](../active/trained-policy/d555-depth-decode-validity.md):
  that lane is unobservable until the 16UC1 decode lands.

## Investigation pointers

- The probe method and its parameterised variant are described in the closed
  predecessor's rig-measurement section; the original lives with the 2026-08-01
  session artifacts.
- `cadence:` line fields that matter here: `depth rx`, `unconsumed`,
  `depth_age`, and **`timer_deadline_missed`** — the last separates a transport
  loss from host contention and was 0 in every offline replay while the rig
  logged 11 146 in one arm.
- Cyclone receive tuning already in place, so it is not the missing piece:
  [`cyclonedds.xml`](../../../source/strafer_ros/strafer_bringup/config/cyclonedds.xml)
  (16 MB socket buffer, defrag headroom 32) plus the `rmem_max` sysctl drop-in
  beside it.
