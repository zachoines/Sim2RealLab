# Flip the inference node's depth subscription to RELIABLE

**Type:** task (deploy runtime)
**Owner:** Jetson
**Priority:** P1 — it costs the policy ~20% of its training cadence on every
mission, and the loss is at the receiver, so no amount of node-side work
recovers it.
**Estimate:** S (one parameter, one QoS profile, plus a re-measure)
**Branch:** `task/depth-qos-reliable-flip`

## Story

As the **operator running the depth-subgoal policy against the sim bridge**, I
want **the inference node to receive every depth frame the bridge publishes**,
so that **the policy runs at the 30 Hz cadence it was trained at instead of the
22–25 Hz it currently sees.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

The 2026-08-01 four-arm re-validation measured the node's own cadence counters
per arm (straddler-free pooled rate, `depth_rx` against `inferences`):

| arm | achieved | depth arrival AT THE NODE | consumption |
|---|---|---|---|
| v2 × `rolling` | 23.70 Hz sim (79% of target) | 22.20 Hz sim | 97.9% |
| v2 × `mission` | 25.06 Hz sim (84% of target) | 25.54 Hz sim | 98.1% |

Consumption near 98% says the node converts almost every frame it *receives*
into an inference. The shortfall is therefore in arrival, not processing — the
node is starved, not slow.

**What that cadence costs is now measured rather than argued.** In closed-loop
emulation of the same temporal texture
([`cadence-emulation-eval`](../../completed/cadence-emulation-eval.md)), the
22–25 Hz band costs 3 points of completion against a 30 Hz baseline
(0.900 → 0.870) and the 12 Hz regime costs about a third (0.900 → 0.610).
Recovering arrival rate therefore recovers almost the whole temporal cost, and
this brief is the lever that recovers it: the executor split the shortfall used
to be attributed to already shipped twice, and the starvation it named was
bounded at 2.9% and refuted.

**The discriminating measurement.** A second subscriber was run **concurrently
with the inference node**, on exactly the node's own depth QoS (BEST_EFFORT,
`KEEP_LAST` depth 1), for 60 s during a live arm:

```
depth 134 frames, 2.25 Hz wall, RTF 0.075  ->  29.84 Hz SIM
consecutive-identical content 0/133
```

The bridge supplies the full ~30 Hz sim cadence and a second subscriber on the
same QoS receives it, at the moment the inference node is receiving 22–25 Hz.
That excludes a sim render shortfall (`sim-depth-render-rate-parity` is a
different defect) and excludes the transport itself.

**Mechanism.** With `history=KEEP_LAST, depth=1`, a frame arriving while the
node is busy — mid-inference, at 30 Hz, on a 3619-dim observation — overwrites
the queued frame, and the older one is lost at the receiver before any callback
runs. The probe does almost no per-frame work and so never overwrites. The
`inference-cadence-shortfall` session measured `reliable` at 30.15 Hz over
container loopback, consistent with this.

Do not be misled by the node's own shortfall attribution, which names
*"this node: frames arrived but did not become inferences"*. Its `gate` skips
are the expected steady-state skip on a depth variant (no new frame since the
last inference) and its `watchdog` skips are inter-mission idle. That line
attributes correctly for a *processing* shortfall and misattributes an *arrival*
one. Only the concurrent independent subscriber separates them.

## Why this matters beyond the Hz number

A `skip_gate` tick publishes **nothing** — it returns without a Twist, so the
previous command stays latched on `/cmd_vel` while the recurrent hidden state
stays frozen. At 22 Hz against a 30 Hz timer, roughly a quarter of ticks latch.
Training has no analogue: it steps the policy exactly once per environment step
with exactly one depth render per step, so a held action paired with a frozen
recurrent state is a deploy-only regime. This is a train↔deploy parity gap on
the control path, not merely a throughput loss.

## What landed (2026-08-05, code half)

- **The sim lanes subscribe RELIABLE.** `STRAFER_DEPTH_RELIABILITY` is a new
  per-lane env override on the node, read the same way `STRAFER_DEPTH_TIMEOUT_S`
  is, and set to `reliable` in both canonical sim env files
  (`env_sim_bridge.env`, `env_sim_in_the_loop.env`). The `inference.yaml`
  default stays `best_effort`, so the real-robot lane is untouched. Verified
  through the whole chain with `docker compose config`: the sim-bridge lane's
  `inference` service receives `STRAFER_DEPTH_RELIABILITY=reliable`, the
  real-robot lane receives no depth key at all.
- **The history depth stays 1**, and a test now pins it on *both* reliabilities
  — that is what keeps a reliable subscription from paying for arrivals with
  staleness.
- **The depth-age instrument AC3 needs.** `_on_depth` now carries the publisher
  stamp alongside the frame, and the age at the moment the policy consumes it
  is reported in the periodic `cadence:` line as
  `depth_age p50=… p95=… max=… s sim n=…`. Under `use_sim_time` both ends are
  sim seconds, so the figure is comparable across the rig's varying RTF. The
  brief's investigation pointer claiming the re-measure "needs no new
  instrumentation" was wrong on this criterion: the node discarded the header
  stamp and had no age counter.
- **The guard against the real-robot lane.** `check_env_sync`'s real-hardware
  guard now covers the new key, so it cannot leak into `env_autonomy.env`.
- **Two code comments that contradicted this repo's own verified finding** are
  corrected — see the QoS-compatibility note below. `_DEPTH_QOS` is replaced by
  `_DEPTH_HISTORY_DEPTH`: only its `.depth` was ever read, so its `reliability`
  field was dead and read as though it were the shipped setting.

## Decision and its basis

**Reliability, not history depth.** Both levers were on the table per the first
acceptance criterion; they fix different losses and only one is measured.

1. **Only `reliable` has been measured to recover the rate.** Replaying the
   preserved bag into the real node
   ([`inference-cadence-shortfall`](../../completed/inference-cadence-shortfall.md)):
   `best_effort` 25.44 / 21.07 / 24.88 / 27.37 Hz sim, `reliable` **30.15 Hz at
   98.6% consumption**, same harness. A raised history depth has never been
   measured at all, so choosing it would be the asserted answer this criterion
   rules out.
2. **The two settings address different links.** A ~921 KB depth frame
   fragments across many RTPS/UDP fragments; losing one drops the whole sample
   at a best-effort reader with no retransmit. Reliability recovers exactly
   that. History depth only prevents *overwrite in the reader's queue*, which
   requires the callback to be late — and `_on_depth` has had its own callback
   group and executor thread since `70323c8`, with `timer_deadline_missed`
   reading 0 in every fixed replay arm. Overwrite is the weaker-supported of
   the two mechanisms, so the brief's stated mechanism ("a frame arriving while
   the node is busy overwrites the queued frame") is not the one being fixed.
3. **Raising history depth trades directly against the third criterion.**
   Depth > 1 lets the reader hold older frames; with the tick driven by arrival
   and the gate keyed on a monotonic seq, the node would work through a backlog
   oldest-first — buying arrival count by spending staleness, which is the
   regression that criterion exists to catch. Depth 1 keeps "the frame the
   policy sees is the newest that arrived" structural on either reliability.

**The QoS-compatibility blocker named in the code is not one on these lanes.**
The node's comment claimed `best_effort` was "the only safe default" because a
RELIABLE subscriber receives nothing from a BEST_EFFORT publisher. The rule is
real; the conclusion was not, and this repo had already established why in
[`depth-reception-reliability`](../../completed/depth-reception-reliability.md):

| publisher | resolved QoS | reliable subscriber compatible? |
|---|---|---|
| sim bridge `async_camera_publisher._IMAGE_QOS` | RELIABLE, KEEP_LAST 10 | yes |
| real D555 via `realsense2_camera` (no `depth_qos` argument in `perception.launch.py`) | `SYSTEM_DEFAULT` → RELIABLE | yes |
| a camera brought up `depth_qos:=SENSOR_DATA` | BEST_EFFORT | **no** |

Nothing in the tree sets that override. The yaml default stays `best_effort`
for the unpinned case rather than because the sim lane needs it. The bridge
publisher being KEEP_LAST rather than KEEP_ALL also bounds the cost: a slow
reader never blocks it, it drops from its own history and sends a GAP.

**"Unpinned" is a real gap on the real lane, and pinning it belongs with the
work that first makes that lane functional.** The real lane's stream config is
pinned by explicit `launch_arguments` to `rs_launch.py` in
`strafer_perception/launch/perception.launch.py`; no QoS argument is among
them, so depth inherits the wrapper's `SYSTEM_DEFAULT`. (`d555_params.yaml` is
*not* the surface — its own header offers it as a `--params-file` and no launch
file loads it, so anything written there today is inert.) Pinning the argument
explicitly would remove the caveat and let the real lane subscribe RELIABLE at
parity with sim. It is deliberately **not** done here: on hardware the node
drops **100% of depth frames at the 16UC1-vs-32FC1 encoding gate**
([`d555-depth-decode-validity`](../trained-policy/d555-depth-decode-validity.md)),
so a QoS pin on that lane is unmeasurable until the decode lands — and an
argument name `rs_launch.py` does not declare fails the include outright, which
is not a change to make blind. It rides with that brief.

**Why the original best-effort rationale does not carry to this lane.**
`70323c8` chose BEST_EFFORT so a lost frame would skip a tick instead of
triggering a retransmit burst into a congested receiver. Two premises of that
have since moved: the same commit shipped the receive-side fix that removed the
drop point (16 MB socket buffer, defrag headroom, the `rmem_max` sysctl), and
`651fa13` made the tick fire on frame *arrival*, so a lost frame is now a lost
hidden-state step rather than a harmlessly skipped tick. The rationale still
stands wherever the publisher's QoS is unknown — which is the lane that keeps
the old default.

**The competing hypothesis is not settled by this change.**
`COORDINATOR_ADDENDUM_CADENCE_AND_RIG.md` (2026-08-03) attributes the arm-1
11.68 Hz to "the single-threaded rclpy executor blocked during TensorRT
inference". That description does not match the shipped node — the executor is
a `MultiThreadedExecutor` with 5 threads and depth has its own callback group —
but that arm also recorded **11 146 missed timer deadlines**, which the loopback
replay never reproduced (0 in every fixed arm). So a real host-load component
exists on the rig that no offline evidence can speak to. The QoS flip does not
address it, and the re-measure has to read both counters to tell them apart.

## What the rig session must run

1. **Bring the sim-bridge lane up** and confirm the override took:
   `docker compose … config` should show `STRAFER_DEPTH_RELIABILITY=reliable`
   on the `inference` service, and the node logs
   `depth_reliability overridden to 'reliable' via STRAFER_DEPTH_RELIABILITY`
   at startup. An `up -d --force-recreate inference` is required — `restart`
   reuses the old container env.
2. **Read the `cadence:` line** for `depth rx=`, the achieved Hz, the new
   `depth_age p50/p95/max`, **and `timer_deadline_missed`**. The last one is
   what separates a remaining transport loss from the host-load component
   above; if the flip lands short with deadline misses high, the next lever is
   the executor/affinity work in that addendum's §2, not more QoS.
3. **Run `tools/bridge_probe.py` concurrently**, on the node's *new* QoS.
4. **For the before/after that AC3 asks for**, take the `best_effort` reading
   with `STRAFER_DEPTH_RELIABILITY=best_effort` on the same lane — the
   instrument is reliability-agnostic, so the two arms differ in one variable.

**A caveat on the 95% bar.** The 2026-08-02 wire measurement in that addendum
put concurrent-subscriber delivery at **28.45 Hz sim**. AC2's ≥ 95% of 30 Hz is
28.5 Hz, at or just above that. If the re-measure reproduces a ~28.5 Hz wire,
the node cannot pass this criterion at any QoS setting, and the bar wants
re-scoping against the measured wire ceiling rather than the nominal 30. Record
the concurrent-probe figure before judging the node against the target.

**The setpoint question is sequenced and pre-registered, not open.** Two
questions were being conflated: *is the temporal gap the cause of the advance
failure* (closed — exonerated as sufficient, below) and *if the rig cannot
serve 30 Hz, what should training target* (a capability question). The second
is conditioned on two things, in order:

1. **Arm D runs first.** If timer-driven stale-reuse is adopted, inference runs
   at 30 Hz regardless of depth arrival and the setpoint question dissolves
   into the trained staleness axis — there is no setpoint to move.
2. **Only if Arm D is rejected** does this brief's post-flip ceiling matter,
   against a rule fixed in advance of the measurement:

| sustained achievable rate | outcome |
|---|---|
| **≥ 27 Hz** | setpoint stays 30 — the band costs ≤ 3% |
| **20–27 Hz** | judgment, reading the harness sensitivity curve at the measured point |
| **< 20 Hz** | setpoint moves via `POLICY_DECIMATION`, and training matches the measured **distribution** — interval histogram, duplicate run lengths, `depth_age` spread — not a mean |

Report the probe figure as a **number with its spread**, never as a verdict
against 30 Hz. The rule consumes the number; the measurement does not decide
the outcome by itself.

**Why a lower setpoint is an accepted outcome but not an expected one.**
[`cadence-emulation-eval`](../../completed/cadence-emulation-eval.md) shipped
2026-08-05 and swept exactly this axis in closed-loop sim, 100 episodes per
profile:

| profile | Hz | completion | ratio to clean |
|---|---|---|---|
| clean | 30.00 | 0.900 | 1.000 |
| band | 23.29 | 0.870 | **0.967** |
| degraded | 12.01 | 0.610 | 0.678 |

So the 22–25 Hz band costs **~3% of completion**, and that read-out
**licenses no cadence-targeted retrain — explicitly neither a fixed-20 Hz
retrain nor a temporal-texture-first augmentation.** A ~28.5 Hz wire ceiling
sits *above* the band point that costs 3%, so a ceiling in that region is close
to free and is not a reason to move the training setpoint. What that read-out
does endorse is this brief: recovering arrival rate "recovers almost the whole
temporal cost."

A flat lower setpoint would also have zero rate variance, which is the standing
argument against that particular shape: the deploy profile's problem is
variance and per-modality staleness skew, not the mean alone. That is why the
`< 20 Hz` branch above hands over a distribution rather than a number.

## Acceptance criteria

- [x] `depth_reliability` defaults to `reliable` for the sim-bridge lane, or the
      depth history depth is raised, with the choice justified against measured
      numbers rather than asserted. — **reliability flipped, history depth held
      at 1**; basis in [Decision and its basis](#decision-and-its-basis).
- [ ] **[operator-gated]** Re-measure with the same method: node `depth_rx` in
      sim Hz, and a **concurrent** independent subscriber. Both must read within
      10% of each other; the node must reach ≥ 95% of the 30 Hz target.
      See [What the rig session must run](#what-the-rig-session-must-run) —
      including why the 95% bar may not be reachable at any QoS.
- [ ] **[operator-gated]** Confirm no regression in end-to-end latency — a
      reliable subscription that buffers stale frames would trade a rate loss
      for a staleness loss. Report the depth age at inference time before and
      after. **The instrument this needs did not exist and now ships** (see
      below); the reading itself is a rig measurement.
- [ ] **[operator-gated]** Re-run at least one arm of the anchoring re-validation goal set and report
      whether the advance numbers move at all. **Whether they move is itself a
      finding either way** — do not pre-commit to an expectation here. The
      2026-08-01 session's attribution of the advance failure was bounded by a
      scene-class confound; that confound has since been discharged and the
      attribution reopened on other grounds
      ([`cadence-emulation-eval`](../../completed/cadence-emulation-eval.md)'s
      read-out), so this brief still must not assume an outcome.
- [x] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [x] No regression in the workflows the touched code supports. — `make test-ros`
      729 passed (`strafer_inference` 461 passed, 11 skipped); `make env-check`
      green.

## Investigation pointers

- The parameter is `depth_reliability`, declared in
  `strafer_inference/inference_node.py` and shipped as `"best_effort"` in
  `strafer_inference/config/inference.yaml`; the per-lane override is
  `STRAFER_DEPTH_RELIABILITY`.
- The node prints the QoS in its shortfall warning, so the rate half of the
  re-measure needs no new instrumentation. The **staleness** half did — the
  node discarded the depth header stamp — and the `depth_age` figures in the
  `cadence:` line are that instrument.
- The concurrent-probe method is `tools/bridge_probe.py` in the 2026-08-01
  session artifacts (`~/strafer_v2_validation/`); it subscribes on the node's
  exact QoS and reports both wall and sim-time rates plus RTF.

## Out of scope

- The sim's duplicate depth content (byte-identical consecutive frames;
  intermittent, 24.1% in one arm and 0% in another). A DGX brief owns it.
- The v2 advance failure and its attribution. Reopened on four candidates by
  [`cadence-emulation-eval`](../../completed/cadence-emulation-eval.md)'s
  read-out and taken up in
  [`cadence-harness-residual-arms`](../trained-policy/cadence-harness-residual-arms.md);
  this brief neither depends on nor prejudges it.
