# Locate the depth receiver-host capacity ceiling

**Type:** investigation → task (deploy runtime)
**Owner:** Jetson
**Priority:** P1 — it costs 4–6× of the depth cadence on every mission, and it
is now the *only* surviving mechanism for the shortfall three briefs have chased.
**Estimate:** M (measurement first; levers scoped only after it names the fault)
**Branch:** `task/depth-receiver-host-capacity`

## Story

As the **operator running a depth policy against the sim bridge**, I want **the
depth frames the bridge already publishes to survive the trip to the node while
the rest of the stack is running**, so that **the policy sees the ~28.5 Hz sim
the host demonstrably receives when nothing else is attached, instead of the
5–8 Hz it sees under load.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

[`depth-qos-reliable-flip`](../../completed/depth-qos-reliable-flip.md) closed
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
[`depth-reception-reliability`](../../completed/depth-reception-reliability.md)
— the **frame-skip-0 whole-Jetson capacity ceiling** and the **`timestamp_fixer`
still-RELIABLE perception receive QoS** named there as its lever — plus items
2–4 of the 2026-08-03 cadence addendum's node-consumption list. Those were
findings inside a shipped brief with no active owner; this brief owns them.

## Acceptance criteria — measurement first, no fixes until it reads out

- [ ] **Which host loses the frames.** Run the probe **on the sim host**,
      alongside the bridge, while the Jetson node is attached. Sim-host probe at
      ~30 Hz sim ⇒ the loss is on the wire or the Jetson; sim-host probe also
      down ⇒ the publisher is throttling and the fault is bridge-side. Report
      both figures with their spread. **Nothing else in this brief is scoped
      until this reads out.**
- [ ] **Subscriber census + traffic accounting** on the raw 640×360 stream:
      enumerate every subscriber (inference node, rtabmap's aligned consumer,
      `timestamp_fixer`, any viz) and compute unicast fan-out × the ~68 Mbit/s
      per-stream requirement against measured link capacity. State whether the
      link is the binding constraint or is not.
- [ ] **Host CPU, node attached vs detached** (`tegrastats`), and with the
      node's decode and inference halves separated if the node permits it. The
      2026-08-06 arms ran with `inferences=0` throughout, so whatever the node
      costs there, it is *not* inference — say what it is.
- [ ] **Single- vs multi-subscriber arms at both reliabilities**, on the
      collapsed regime. This is where `reliable` vs `best_effort` gets its real
      test: the completed brief only ever compared them on an uncollapsed rig
      and on container loopback.
- [ ] Arms are **paired and interleaved**, with the warm-up trend controlled —
      the predecessor's first three arms read 26 → 77 → 246 frames per 90 s
      *while alternating QoS*. Single arms taken before the bridge settles
      measure the warm-up.
- [ ] Every arm records that the node under test is the intended build, by both
      tells: the startup `depth_reliability overridden to …` line and a
      `depth_age` field in the `cadence:` line. The `strafer-gpu` image bakes
      the node code, so a bind-mount alone changes nothing that runs, and
      `up -d --force-recreate` destroys an in-container `colcon build` — build
      *after* the final recreate, then `restart`.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports.

## Candidate levers — candidates, not conclusions

Recorded so they are not re-derived; none is adopted before the measurement
names the bottleneck.

- **CPU pinning / affinity between the inference container and the SLAM/Nav2
  containers.** The host-contention trigger both lanes pre-registered has
  effectively fired, so this lever is now *motivated* — but it lands after the
  measurement, not before.
- **Subscriber consolidation** — one host-side receiver of the raw stream that
  republishes locally, so unicast fan-out stops multiplying the wire cost.
- **Halve the wire cost: publish 16UC1 millimetres from the sim bridge**
  (~921 KB → ~461 KB per frame). This is a **cross-lane candidate**, flagged
  not scoped: the bridge half is DGX-owned. It couples with
  [`d555-depth-decode-validity`](../trained-policy/d555-depth-decode-validity.md)
  — the 16UC1 decode path the node already needs for real hardware would let the
  sim lane carry half the bytes *and* close the encoding disparity between the
  two lanes in one move. Raise it with the DGX lane rather than implementing it
  here.

## Post-fix plan (only once the ceiling moves)

- **Re-run at least one arm of the anchoring re-validation goal set** and report
  whether the advance numbers move at all. Whether they move is a finding either
  way — do not pre-commit. This item moves here from the closed predecessor.
- **Produce the achievable-cadence ceiling** the setpoint rule consumes. The
  rule itself (Arm D first; then ≥ 27 / 20–27 / < 20 Hz) is unchanged and lives
  in [`depth-qos-reliable-flip`](../../completed/depth-qos-reliable-flip.md);
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
  [`d555-depth-decode-validity`](../trained-policy/d555-depth-decode-validity.md):
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
  [`cyclonedds.xml`](../../../../source/strafer_ros/strafer_bringup/config/cyclonedds.xml)
  (16 MB socket buffer, defrag headroom 32) plus the `rmem_max` sysctl drop-in
  beside it.
