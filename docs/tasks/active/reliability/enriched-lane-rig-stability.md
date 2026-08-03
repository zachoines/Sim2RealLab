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

> **Update 2026-08-03:** the poisoned `enrich1` database was deleted in a
> volume-wide map cleanup and is **no longer available for forensics**. Triage
> must work from the captured logs (`~/strafer_v2_validation/logs/slam_*.log`,
> which contain all three FATALs and the rejected-loop-closure warnings that
> preceded them) and from reproducing the assertion fresh.

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

**ROOT CAUSE (coordinator DGX forensics, 2026-08-03 ruling): duplicate-instance
GPU contention — plus one still-open mid-run stall.** Two earlier diagnoses
recorded here are dead:

- the **shader-cache theory is CLOSED**, not merely retracted: the portable-mode
  kit tree's caches were warm and reused on every launch (46 GB
  DerivedDataCache with index-only writes; RTX shadercache untouched since
  April; driver GLCache opened 6 s after launch; zero CUDA JIT writes). A
  recompile storm is impossible to hide under
  `--/rtx/materialDb/syncLoads=True` with sar showing <10% CPU.
- the **"45–90 min first-frame latency" claim is also wrong.** Sar/sysstat +
  `/proc` forensics over every launch 08-01 → 08-03: warm-up **on an idle GPU
  is 10–20 min, vanilla and enriched alike** (08-01 vanilla ≤14 min; 08-02
  12:49 enriched ≤10–20 min; 08-02 22:17 enriched ≤12 min).

What actually produced the 45–90 min "stalls": **stacking a second bridge on a
GPU another bridge already owned.** All four 08-03 launches (10:15 / 11:11 /
11:30 / 12:04) started while the **08-02 22:17 bridge still held the GPU**;
each loaded ~17 GB, idled 45–75 min at <10% CPU, and died without a clean
shutdown. The depth "recovery" observed 08-03 12:40 was served by that
resident 08-02 bridge — healthy for ~14 h — once the stacked duplicates died
off, not by any relaunch coming good.

**Still open, as a distinct mode:** the original 2026-08-02 **mid-run** stall
(no restart involved; a bridge that had rendered for hours stopped producing
frames while physics continued, `camera_info` kept flowing, and the Kit log
stayed clean). If it recurs on an idle GPU: **`py-spy dump` the bridge before
killing it.**

**Operator rule (replaces the earlier "wait 60–90 min" rule):** before ANY
bridge launch, run
`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`;
if a bridge PID is resident, **adopt it or kill it — never stack a second
instance**. On an idle GPU expect 10–20 min to first frames; poll
`ros2 topic hz /d555/depth/image_rect_raw`.

**Asks (DGX-owned):**

1. **Bridge-side first-frame instrument** — a one-line
   `first image published, t=…` print from `run_sim_in_the_loop.py`. It must be
   bridge-side because Kit logs go silent ~10 s after env construction.
2. **Fail-loud launch guard** in `run_sim_in_the_loop.py`: refuse to start when
   a resident compute process already holds the GPU (env-var override for
   intentional coexistence). Mechanical enforcement beats discipline — the
   stacking mode cost the operator a full day.
3. Reproduce or rule out the **mid-run** stall as a distinct mode (py-spy
   capture protocol above).
4. **Document the DGX handback protocol** (coordinator-mandated, standing) in
   the cheatsheet: any session that launches a long-running DGX process over
   SSH ends with either explicit teardown (kill the PID, confirm the GPU is
   free) or an explicit handback line in the session report — "DGX bridge PID
   <n> left RUNNING — task <id>, scene token <t>, launched <time>." Silent
   orphans are a session-report defect.

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
- [x] Renderer/first-frame mode root-caused: duplicate-instance GPU contention
      (coordinator DGX forensics, 2026-08-03; shader-cache and first-frame-latency
      theories both closed — see mode 4).
- [ ] Bridge-side `first image published, t=…` print (Kit logs go silent ~10 s
      after env construction, so it cannot live there).
- [ ] Fail-loud GPU launch guard in `run_sim_in_the_loop.py` (refuse to start
      over a resident compute process; env-var override).
- [ ] The **mid-run** stall (2026-08-02 original event) reproduced or ruled out
      as a distinct mode; py-spy dump before killing on any recurrence.
- [ ] DGX handback protocol documented in the cheatsheet (explicit teardown or
      an explicit handback line; silent orphans are a session-report defect).
- [ ] **Promote `switch_arm.sh` into deploy tooling.** The session tool
      (`~/strafer_v2_validation/tools/switch_arm.sh`) is the only safe way to
      swap (model × anchoring): it force-recreates `inference` with the
      read-only YAML overlay and then **verifies from inside the container**
      (model path, `subgoal_anchoring:` line, `anchoring=` log line) — exactly
      the checks that catch the silent `docker compose restart` foot-gun.
      Generalize it (e.g. `deploy/tools/` or a make target), keep the
      in-container verification, and document it in the cheatsheet.
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
