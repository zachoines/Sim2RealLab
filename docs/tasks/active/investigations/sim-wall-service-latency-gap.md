# Measure the sim-time discount on wall-clock service latency (VLM / planner)

**Type:** investigation
**Owner:** Either
**Priority:** P3
**Estimate:** S–M
**Branch:** task/sim-wall-service-latency-gap

## Story

As a **mission operator preparing for real-robot deployment**, I want
**the wall-clock cost of VLM / planner service calls measured against
sim-time mission budgets**, so that **deadlines and recovery loops tuned
in sim don't silently assume services that are ~10× faster than the
real robot will experience**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`nav2-sim-real-promotion-architecture`](../tooling/nav2-sim-real-promotion-architecture.md)
  — the sim→real knob-promotion discipline this gap belongs to.

## Context

Under `use_sim_time`, everything in the sim-time domain (physics, policy
tick, Nav2, executor deadlines) is invariant to the sim's real-time
factor. External HTTP services are not: VLM grounding and planner calls
cost **wall** seconds while the robot moves in **sim** seconds. Measured
on the 2026-07-02 sim-in-the-loop runs, the bridge RTF is ~0.095
(derived from RTAB-Map's stamp-throttled iteration cadence), so a 3 s
VLM call costs only ~0.3 s of mission time in sim — **the services are
effectively ~10× faster in sim than they will ever be on the real
robot**, where wall = sim.

This is a flattering-direction sim-to-real gap: mid-mission re-grounding
(`scan_for_target`, staging-loop re-projection, arrival verification)
and any recovery loop that interleaves service calls with motion will
consume dramatically more real mission time on deployment than sim
validation suggests. Budgets that pass in sim (e.g. the 180 s sim-time
navigation timeout, scan/rotate budgets that span grounding calls) may
be materially tighter in real terms.

## Acceptance criteria

- [ ] Instrument (or extract from executor logs) wall-clock latency
      distributions for the VLM `/ground`, `/describe`, `/detect_objects`
      and planner endpoints during a representative sim mission set —
      p50 / p95 per endpoint.
- [ ] Enumerate the executor code paths whose sim-time deadlines span
      service calls (scan_for_target, staging re-ground loops,
      `_verify_arrival`, plan compilation), and compute each budget's
      effective real-robot equivalent at RTF = 1 (i.e. re-add the
      service wall cost that sim discounts).
- [ ] Disposition per path: budget is fine at RTF 1 / needs a widened
      default / needs the service call moved outside the deadline.
      File follow-up briefs for any needed changes rather than fixing
      in this investigation.
- [ ] Record the method for deriving RTF from logs (the RTAB-Map
      detection-cadence trick or a /clock probe) so future runs can
      re-measure cheaply.

## Out of scope

- Speeding up the sim or the services (the DGX hosts both; see
  [`bridge-publish-rate-decouple`](../sim-performance/bridge-publish-rate-decouple.md)
  for the bridge-side throughput work).
- Changing any deadline in this brief — measurement and disposition
  only.
