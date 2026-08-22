# Window the cadence report so a multi-mission session stops reporting false shortfalls

**Type:** bug (instrumentation)
**Owner:** Jetson (`strafer_inference` lane)
**Priority:** P2 — it does not change what the node *does*, but the number it
reports is the one the setpoint rule consumes, and on any session with more than
one mission that number is wrong in the pessimistic direction. This session
emitted **220** `CADENCE SHORTFALL` warnings while the achieved cadence was
30.00 Hz sim.
**Estimate:** S (reset two fields where the counters already reset, or window
them)
**Branch:** `task/cadence-report-window-never-resets`

## Story

As the **engineer reading a rig session's cadence record**, I want **the
reported rate to describe the interval it is printed for**, so that **a
`CADENCE SHORTFALL` warning means the policy loop actually fell behind, and the
achievable-cadence ceiling can be read off the log instead of recomputed from
raw counters.**

## Context

`_maybe_log_cadence` prints `rate = inferences / span_sim` where

```
span_sim = self._cadence_t_last_sim - self._cadence_t0_sim
```

`_cadence_t0_sim` is assigned once, at the first inference, and never cleared
([`inference_node.py:1177-1179`](../../../../source/strafer_ros/strafer_inference/strafer_inference/inference_node.py#L1177));
`_cadence_t_last_sim` advances every inference. So `span_sim` is the node's
**lifetime** since its first inference, while the log line's own header says
`counters every 10 s`.

Between missions the node holds ticks at the watchdog because no goal is active
— correct behaviour — and those idle seconds land in `span_sim` while
contributing no inferences. `rate` therefore decays as soon as the first
inter-mission gap opens, and recovers only asymptotically once inference
resumes. The shortfall test at
[`:1267`](../../../../source/strafer_ros/strafer_inference/strafer_inference/inference_node.py#L1267)
(`rate < 0.9 * target`) is evaluated against that decayed value, so it latches on
for the rest of the session.

The line mixes window semantics, which is what makes this hard to spot:
`depth_age` is summarised over a bounded recent sample (`n=512` in every line
this session), while `inferences`, `ticks`, `depth rx`, `repeat_content` and the
skip counters are lifetime totals, and `rate` is a lifetime average. One line,
three different windows.

## Measured, 2026-08-17 rig gate

Six missions plus five unscored transits over ~96 min wall, one bridge, one node
process.

- **220** `CADENCE SHORTFALL` warnings, ramping smoothly from 11.46 Hz to
  26.0 Hz. The smooth monotone ramp is the signature: a real cadence fault does
  not walk linearly upward across an hour.
- Recomputed by differencing consecutive counter windows over the **complete**
  container log (886 counter lines, 948.8 s sim), `d(inferences)/d(span_sim)`
  across the 787 usable windows: **p05 28.18 / p50 30.00 / p95 31.19 Hz sim**,
  with only 2.3% (18/787) below 27 Hz, all of them straddling a mission
  boundary. Figures taken from a partial capture are smaller, which is why the
  span is stated.
- `timer_deadline_missed = 0`, `reuse = 0`, `obs_none = 0`, `gate = 0`,
  `bad_encoding = 0`, `bad_shape = 0` for the whole session. Nothing else
  corroborated a shortfall.
- The **attribution** in the warning is correct and is what prevents a
  misreading: it names `watchdog` skips rather than the depth transport. A
  reader who trusts the attribution reaches the right conclusion; a reader who
  trusts the rate does not.

## Acceptance

- [ ] The rate printed with a counter block describes that block's interval.
      Either reset `_cadence_t0_sim`/`_cadence_t_last_sim` wherever the other
      counters reset, or carry a separate windowed accumulator for the rate.
- [ ] `CADENCE SHORTFALL` fires on the windowed rate, so it does not latch for
      the remainder of a session after one idle gap.
- [ ] Idle intervals with no active goal are excluded from the denominator, or
      the line states that it includes them.
- [ ] Window semantics on the line are uniform, or each field names its window.
- [ ] A regression test drives two missions separated by an idle gap and asserts
      no shortfall warning is emitted when every active window ran at target.

## Scope

Reporting only — no change to tick scheduling, the watchdog, reuse semantics, or
any threshold. The achievable-cadence figures already recorded from earlier
sessions were taken from this same field; any that came from a multi-mission run
should be recomputed by differencing before being compared to a new one.
