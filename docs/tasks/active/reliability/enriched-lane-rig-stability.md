# Stabilise the enriched sim-bridge lane — four failure modes that cost a session

**Type:** investigation (rig reliability)
**Owner:** Jetson + DGX
**Priority:** P1 — these did not degrade a measurement, they **prevented** one.
The 2026-08-02 enriched addendum session lost its second arm outright, and three
of the four modes fail *silently* or, worse, present as something else.
**Estimate:** M (four independent modes; each is separately actionable)
**Branch:** `task/enriched-lane-rig-stability`

## Story

As the **engineer running behavioural acceptance sessions on the sim-bridge
lane**, I want **the four failure modes observed on 2026-08-02 diagnosed and
either fixed or given documented detection/recovery procedures**, so that **a
multi-hour rig session is not lost to infrastructure, and so that a rig failure
is never mistaken for a policy result.**

## Context

The [2026-08-02 enriched-scene anchoring addendum](../../completed/enriched-scene-anchoring-addendum.md)
ran one of its two arms. Arm 1 completed and is clean. Arm 2 was started twice
and completed neither. The session record is
`~/strafer_v2_validation/ENRICHED_SESSION_LOG.md`.

**The unifying risk is misattribution.** Three of these four modes produce
symptoms that look like a policy or planner problem:

- a wedged WiFi adapter looks like "the machine crashed";
- inactive Nav2 lifecycle nodes look like "Nav2 can't plan there" *while
  `ComputePathToPose` keeps returning `PLANNABLE`*;
- a dead renderer looks like "the policy stopped commanding", because the node
  correctly watchdog-skips when depth is stale.

Arm 2's first two missions were in fact run against **zero camera input** before
this was noticed, and were discarded. Nothing in the mission JSON would have
revealed it — only the node's `depth rx=0` counter did.

## The four modes

### 1. rtabmap `Memory.cpp:3852::addLink()` — three deaths, one unloadable database

```
[FATAL] Memory.cpp:3852::addLink() Condition (fromS->getWeight() >= 0 && toS->getWeight() >= 0) not met!
terminate called after throwing an instance of 'UException'
```

Killed SLAM three times, at growing working-memory size (deaths near `WM≈262`,
`WM≈1908`). Preceded each time by rejected loop closures
(`Not enough inliers 0/15`). The third death left
`rtabmap_enrich_enrich1.db` (**1.25 GB**) **permanently unloadable** — rtabmap now
aborts on *load*, before producing a single iteration, so the recovery documented
in the cheatsheet (`docker restart strafer_slam`) no longer works on it.

The existing [`rtabmap-cold-start-determinism`](rtabmap-cold-start-determinism.md)
brief is adjacent but not the same failure; this is a mid-session
weight-bookkeeping assertion, not a cold-start issue.

**Ask:** determine whether this is the known upstream `addLink` weight bug and
whether a parameter (loop-closure/memory-management related) avoids it; and add a
db integrity pre-check so a poisoned database is detected at launch rather than
by an abort.

### 2. Nav2 lifecycle nodes drop to `inactive` after a network partition

After the WiFi adapter wedged, these did **not** recover on reconnect:

| node | state |
|---|---|
| `bt_navigator` | **inactive** |
| `velocity_smoother` | **inactive** |
| `waypoint_follower` | **inactive** |
| `planner_server`, `controller_server`, `behavior_server`, `smoother_server`, both costmaps | active |

This is the most dangerous of the four, because it fails in two *actively
misleading* stages:

1. With `bt_navigator` inactive, **every `/navigate_to_pose` goal is rejected**
   while `planner_server` stays active — so `ComputePathToPose` dry-runs keep
   returning `PLANNABLE`. **A plannability pre-check reports perfect health while
   no mission can start.**
2. After reactivating `bt_navigator` alone, goals are **accepted and planned**
   but the base never moves, because `velocity_smoother` — the final output
   stage — is still inactive. `/cmd_vel` carries nothing despite 7 advertised
   publishers.

Recovery was `ros2 lifecycle set /<node> activate` for the three; no container
restart was needed. **A restart would have been the wrong reflex** — container
`/tmp` is volatile and holds arm JSONs and staged tooling.

**Ask:** find out why `lifecycle_manager` does not re-activate them (bond
timeout during the partition is the suspicion); add a health check that asserts
**all** managed nodes are `active`, not just that the stack is up.

### 3. `mt7921u` USB WiFi adapter wedges under sustained load

NETGEAR `0846:9060` (MediaTek MT7921), USB 3.0 on `tegra-xusb`.

```
mt7921u: Message 00020002 (seq 8) timeout
  mt76u_stop_tx+0x2a8/0x370 [mt76_usb]        <- kernel stack trace
... then on replug:
mt7921u: vendor request req:… failed:-71      <- x51, EPROTO
```

A firmware command timed out in the TX path; the device stopped answering USB
control transfers entirely and was dead for **27 minutes** until physically
replugged. **The host never crashed** — `/proc/uptime` was continuous, no panic,
no OOM. Only the physical replug forced a teardown that surfaced the backlog.

Two enabling conditions worth recording:

- **Persistent journald was not enabled**, so two earlier events the same day
  were unrecoverable. It has since been enabled
  (`/var/log/journal`) and is what made this diagnosis possible.
- A **wired powerline path exists and is faster**: measured **251.6 Mbit/s**
  aggregate vs the dongle's 126.9, against a live traffic requirement of only
  **67.8 Mbit/s**. (Single-stream `nc` under-reports both at ~29 Mbit/s — that is
  a sender-pipeline artefact, not link capacity.) Latency is worse: 20.5 ms avg
  vs 6.4 ms.

**Ask:** move the DDS traffic to wired and re-measure; keep wired as the SSH
path regardless so an adapter wedge no longer costs rig control. **Do not switch
transports mid-session** — it changes latency and jitter and confounds any
arm-to-arm comparison.

### 4. The Isaac renderer stops producing frames while physics continues

The failure that ended the session. Physics kept stepping and DDS stayed healthy,
but **no rendered frame was ever produced again**:

| topic | state |
|---|---|
| `/clock` | 29 Hz — normal |
| `/d555/color/camera_info` | **3.0–4.5 Hz — flowing** |
| `/d555/color/image_raw` | **no data** |
| `/d555/depth/image_rect_raw` | **no data** |

`camera_info` comes from the **same OmniGraph node** as the images but carries no
rendered pixels. Its continuing to flow while both image topics are silent
isolates the fault to the **RTX render/annotator pipeline**. Corroborating:
**RTF rose 0.117 → 0.182** — the sim ran *faster* because it stopped paying
render cost.

**ROOT-CAUSED AND FIXED: a stale Omniverse/RTX shader cache.** Moving
`~/.cache/ov` (9.4 GB) aside and relaunching restored rendering —
**27.47 Hz sim depth, RTF 0.105, 0% consecutive-identical**. **No host reboot was
required.**

**The trap, and a large part of why this brief exists:** the fix was initially
recorded as having *failed*. With the cache cold, shaders recompile before the
first frame appears, so the topics stayed silent for minutes after relaunch —
indistinguishable from the fault itself. **A cold-cache relaunch must be given
several minutes before it is judged.**

Ruled out along the way, none of which fixed it:

- sim process state — **three** full relaunches, each reporting the correct task,
  `Environment seed : 42`, and `frame_skip=0 (derived, derived 0)` at 30 Hz;
- GPU contention — a single CUDA context (7.4 GB), no orphans, 70 GB RAM free,
  GPU 41% / 50 °C;
- DDS/discovery — `Publisher count: 1`, both sim nodes visible;
- Jetson subscribers — `slam` and `sim-perception` force-recreated;
- Kit diagnostics — **6706 lines, zero `[Error]`/`[Fatal]`**.

**Ask:** find why the cache goes stale (it had accumulated on a host up 28 days),
and add a **renderer health check on the bridge side** — the sim should not report
"async camera publisher up" and then publish `camera_info` into a void while
producing no frames. Detecting this costs one counter; not detecting it cost an
arm.

## Acceptance criteria

- [ ] Each of the four modes has either a fix or a documented
      detect-and-recover procedure in
      [`docs/sim_bridge_autonomy_cheatsheet.md`](../../../sim_bridge_autonomy_cheatsheet.md).
- [ ] **A pre-arm health gate** that a session can run in one command and that
      fails loudly on any of: a managed Nav2 node not `active`; zero depth
      arriving at the inference node; rtabmap not iterating; a database that
      fails an integrity pre-check.
- [ ] The gate is wired into the session protocol **before** the first mission of
      every arm, and re-checked between arms.
- [ ] rtabmap `addLink` triaged against upstream; poisoned-database detection at
      launch.
- [ ] Nav2 lifecycle non-recovery root-caused; `lifecycle_manager` behaviour on
      bond loss understood.
- [ ] Wired transport evaluated for the DDS path and a recommendation recorded.
- [x] Renderer stall root-caused (stale `~/.cache/ov`; cleared, rendering restored).
- [ ] Renderer-stall **detection** on the sim side, plus a documented cache-clear
      procedure that notes the cold-cache recompile delay.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).
- [ ] No regression in the workflows the touched code supports.

## Out of scope

- The depth QoS flip — [`depth-qos-reliable-flip`](depth-qos-reliable-flip.md)
  owns it. Note it targets a **different** shortfall: node-side consumption
  (the executor blocked during TensorRT inference; idle the node receives every
  frame with 1 missed deadline, mid-arm it missed **11 146**), not any of the
  four modes here.
- Any policy or training change.
