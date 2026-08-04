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

**Ask (superseded by the triage below):** the original ask — chase an upstream
`addLink` weight bug and a loop-closure parameter — is **not** the fix. The
triage below shows the trigger is our own container stop grace period. What
remains: apply the grace-period fix, keep the db integrity pre-check as a
backstop, and only pursue upstream if the assertion survives the fix.

> **Update 2026-08-03:** the poisoned `enrich1` database was deleted in a
> volume-wide map cleanup and is **no longer available for forensics**. Triage
> must work from the captured logs (`~/strafer_v2_validation/logs/slam_*.log`,
> which contain all three FATALs and the rejected-loop-closure warnings that
> preceded them) and from reproducing the assertion fresh.

#### TRIAGE 2026-08-04 — this is OUR defect, not (primarily) an upstream bug

Done from `logs/slam_v2xmission.log` alone, as the coordinator scoped. The
framing above — "a mid-session weight-bookkeeping assertion" — is **wrong**, and
the fix is a one-line compose change.

**The assertion fires at DATABASE LOAD, not mid-session.** The evidence is
**the absence of any `rtabmap (N):` iteration line before the FATAL** — rtabmap
prints one per processed frame, and on the aborting launches it prints none:

```
17:53:46.969  rtabmap subscribed to (approx sync): /strafer/odom, /d555/... , /scan
17:53:47.434  [FATAL] Memory.cpp:3852::addLink() Condition (fromS->getWeight() >= 0 && toS->getWeight() >=0) not met!
```

(Elapsed time alone would prove nothing at 30 Hz — a frame could arrive inside
0.5 s. The iteration counter is what shows none was processed.) Every aborting
launch shows the same shape: subscribe, then FATAL, with no iteration in
between.

**rtabmap names the cause itself**, on the next launch that got far enough:

```
Memory.cpp:490::loadDataFromDb() The dictionary is empty or missing some words from nodes
  in WM, we will try to repair it. This can be caused by rtabmap closing before it has
  time to save the dictionary. Re-creating the dictionary from 261 nodes...
Memory.cpp:555::loadDataFromDb() Regenerated the dictionary with 15444 missing words (0 -> 15444)
```

**0 → 15444 words**: the visual-word table was *entirely absent* while 261 nodes
were present. rtabmap writes nodes incrementally but flushes the dictionary and
working-memory state **at close** — so the database was truncated mid-shutdown.

**Why it was truncated — the deployment defect.** `deploy/docker-compose.yml`
sets **no `stop_grace_period`**, so Docker's default applies: **SIGTERM, then
SIGKILL after 10 seconds.** The close-time write scales with the working set
being serialised, not with the 1.25 GB file itself, and on a map this size it
does not complete in 10 s. The
captured log contains **zero** rtabmap shutdown or save messages across every
stop — it never reached its save path. (`init: true` is also unset, so PID 1 is
the launch process with no dedicated reaper.)

**The accumulation, which explains the escalating severity:**

```
container stop / force-recreate
  → SIGKILL at 10 s, mid-flush
  → dictionary + WM state truncated a little further
  → next load: signature with weight < 0 reached by addLink() → abort
```

Damage accumulates across **stops**, not across aborts: a load-time abort writes
nothing, so it neither worsens nor repairs the database — it simply re-reads the
same truncated state, which is why the launch is deterministic once poisoned.

This predicts what was observed: harmless early (a small working set flushes
inside 10 s), fatal as the map grows, and finally a database that aborts on load
with zero iterations. It also explains why `docker restart strafer_slam` — the
cheatsheet's documented recovery — stopped working: it re-enters the same load
path.

**Fix, in priority order:**

1. **`stop_grace_period: 180s` on the `slam` service** (and any service owning a
   large on-disk store). This is the root-cause fix and is one line.
2. **`init: true`** on the compose services for correct signal delivery/reaping.
3. **Database integrity pre-check at launch** (already an acceptance item) —
   with the grace period fixed this becomes a backstop rather than the mitigation.
4. Verify rtabmap actually completes its save within the new window: on stop,
   expect the DB mtime to advance and the next load to show **no**
   `loadDataFromDb()` repair warning. That warning's absence is the regression
   test.

**Upstream relevance — secondary and weaker.** That `addLink()` hard-asserts
(aborts the process) on an inconsistent-but-repairable database, rather than
degrading like the dictionary path immediately above it does, is arguably an
upstream robustness gap worth reporting. But it is **not** why this happened to
us, and no upstream patch is needed to fix our rig: we were killing the process
mid-write. Version pinning and an upstream issue search are only worth doing if
the assertion recurs *after* the grace-period fix.

**Confidence:** high on the mechanism (rtabmap's own diagnostic names it, the
load-time timing is unambiguous, and the 0→15444 dictionary loss is direct
evidence of a truncated flush); untested on the fix, since it needs one
stop/start cycle on a large database to confirm — cheap, and it can ride the
next rig session rather than needing one of its own.

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
stayed clean). If it recurs: **capture stacks before killing it — `py-spy dump`
for Python frames AND `gdb`/`eu-stack` for the native ones** (the parked threads
are native, so py-spy alone is insufficient).**

> **ESCALATION 2026-08-04 — the stall now reproduces at LAUNCH, and it blocks
> the coordinator's sequencing.** During the cadence-profile capture attempt:
>
> - the **mid-run** mode recurred on the resident 08-02 22:17 bridge (~19 h of
>   healthy rendering, then depth dead while `camera_info` trickled);
> - then **two consecutive FRESH launches on a verified-idle GPU** (08-03
>   ~17:37 and 08-04 00:32:54, launched under the new pre-launch resident-PID
>   check) both stalled at first frame. Both logs went **silent ~14 s after
>   startup**, immediately after the `async camera publisher up` line; no
>   frame was ever produced (7 h observed / 35 min observed); `camera_info`
>   ~0.6 Hz wall ≈ RTF ~0.02.
> - All three instances share one signature: **every render-side thread
>   (`vkrt Analysis`, `vkps Update`, `UsdFrameComplete`, `rtx::streaming`,
>   tbb pool) parked in `futex_wait`**, single-digit CPU, zero Kit-log
>   errors. Onset age ranges from ~14 s to ~19 h, so this is one blocking
>   mechanism, not a warm-up phenomenon.
> - This **falsifies the 10–20 min idle-GPU warm-up expectation as a
>   sufficient operator model**: an idle GPU and a clean pre-launch check do
>   not guarantee first frames at all.
> - **The debugger precondition is unmet, and needs BOTH tools.** `py-spy`
>   could not attach at all (ptrace requires sudo on the DGX —
>   `Permission Denied`), and even with sudo it would be **insufficient**: the
>   parked threads are **native** (`vkrt`, `vkps`, tbb), and py-spy reports only
>   Python frames. The standing precondition is therefore scoped sudoers for
>   **`py-spy` AND `gdb`/`eu-stack`** (native stacks), or
>   `kernel.yama.ptrace_scope=0` for the debug window.
> - **Discriminant (coordinator, 2026-08-04):** the Isaac Lab **play-env** path
>   (no ROS bridge, tiled render) produced frames at **30 Hz on the same box in
>   the same window**. The stall has only ever been observed on the **bridge
>   camera path**, and both fresh stalls went silent immediately after
>   `async camera publisher up` — so the mechanism is plausibly specific to that
>   path rather than box-wide GPU state. Start debugging there.
> - Forensics on the DGX: `~/bridge_logs/midrun_stall_3361_forensics.txt`,
>   `firstframe_stall_42566_forensics.txt`,
>   `firstframe_stall_retry_forensics.txt`, stalled-launch logs
>   `cadence_capture_bridge{,2}.log`, Kit log `kit_20260803_173707.log`.
>
> **Consequence — scoped.** This mode is the critical path of **the cadence
> profile capture and of all future rig sessions**. It is **not** the critical
> path of the cadence adjudication itself: the emulation harness's synthetic
> cells (clean 30 Hz baseline, four-arm band, and the arm-1 profile
> reconstructed from logged telemetry) are **unblocked and proceed on the DGX**,
> carrying pre-registrations (i)/(ii)/(v) fully and (iii)/(iv) on the
> reconstructed profile. Only the **measured-profile replay** cell waits on the
> capture, as a later addition. A reader must not deprioritise the eval on the
> strength of this brief. What this mode does block is any further rig time
> spent on capture attempts — none should be spent until it is ruled.

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
3. Reproduce or rule out the **mid-run** stall as a distinct mode. Start from
   the **bridge camera path** — the play-env path renders clean at 30 Hz on the
   same box, and both fresh stalls went silent immediately after
   `async camera publisher up`. Native-stack capture protocol above.
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
- [x] rtabmap `addLink` **triaged** — load-time assertion caused by SIGKILL at
      the 10 s default stop grace period truncating rtabmap's dictionary flush;
      not primarily an upstream bug (see triage under mode 1).
- [x] **Grace-period fix applied**: `stop_grace_period: 180s` on `slam`. A
      survey of the compose found slam is the **only** service with a read-write
      persistent store (every other mount is read-only config; the TRT engine
      cache writes at build completion, not at stop), so no other service needs
      it. `docker restart` inherits StopTimeout, so the cheatsheet's recovery
      path is covered.
- [ ] **Runtime verification** of that fix: a stop/start cycle on a large
      database showing **no** `loadDataFromDb()` repair warning. Rides the next
      rig session.
- [ ] **`init: true` — deliberately deferred, not dropped.** It is correct for
      signal delivery and zombie reaping, but it is a behaviour change to every
      service and is not on the path to this failure: the flush is truncated by
      the grace period, not by PID 1 mishandling SIGTERM (rtabmap does receive
      it — it simply cannot finish in time). Landing it here would mix an
      untested global change into a targeted fix. File it separately.
- [ ] **Residual, out of this fix's reach:** dockerd's own `shutdown-timeout`
      (default 15 s) SIGKILLs containers on daemon shutdown and host reboot
      regardless of StopTimeout. A database truncated by a host reboot is
      therefore expected, not a regression of the grace-period fix. Raising
      `shutdown-timeout` in `daemon.json` is the lever if reboot-time corruption
      proves to matter.
- [ ] Poisoned-database detection at launch, as a backstop.
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
- [ ] The stall (both onsets — mid-run and at-launch) reproduced or ruled out;
      native + Python stacks captured before killing on any recurrence.
- [ ] **Debugger precondition provisioned on the DGX** (operator): scoped
      sudoers for `py-spy` and `gdb`/`eu-stack`, or
      `kernel.yama.ptrace_scope=0` for the debug window. Without this the
      capture protocol cannot execute.
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
