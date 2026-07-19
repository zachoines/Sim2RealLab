# Derive the bridge camera publish cadence from decimation

**Status:** Shipped 2026-07-18 in `69cb3b0` (DGX). `--camera-frame-skip` default
flips from a hardcoded 3 to `None` = derived
(`round(POLICY_PERIOD_S / (sim.dt x decimation)) - 1`, floored at 0) so the bridge
camera publishes once per policy period at any decimation; explicit values still
win verbatim (escape hatch, floored at 0) with an off-policy startup warning, and
a startup contract print plus a duplicate-frame warning make the cadence
impossible to mis-read. Behavior at the default `decimation 1` is unchanged
(derived skip = 3). New Kit-free unit suite 18/18; full lab pure suite 788 passed
/ 1 skipped.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/154

**Type:** task / bug (cadence correctness)
**Owner:** DGX (`source/strafer_lab/scripts/run_sim_in_the_loop.py`, `source/strafer_lab/strafer_lab/bridge/`)
**Priority:** P2
**Estimate:** S — a pure derivation + startup contract print + unit tests; no mechanics change.
**Branch:** task/bridge-frame-skip-derive

## The fault (measured, session-confounding)

`run_sim_in_the_loop.py`'s `--camera-frame-skip` defaulted to a hardcoded
**3**, calibrated for the old bridge default `decimation=1`: a bridge tick
was one physics tick, and skip-3 throttled 120 Hz ticks down to the 30 Hz
depth contract. At `--decimation 4` a bridge tick is one `env.step` = one
fresh render already at 30 Hz, so the same default **discarded 3 fresh
frames of every 4** — depth published at 7.5 Hz sim, one quarter the
training/policy rate. The freshness gate then correctly ticked the policy at
a rate its training never saw. This silently confounded decimation-4
deployment runs until the cadence was pinned by hand.

## The invariant now encoded

The bridge publishes one camera frame per **policy period** — one frame per
`1/30` sim-s, the same contract the real D555 (30 Hz) and training (a fresh
render per `env.step`) both satisfy. In bridge ticks:

    frame_skip = round(POLICY_PERIOD_S / (sim.dt x decimation)) - 1   (floored at 0)

This yields the correct skip for both decimation configs (`sim.dt` 1/120):
`decimation 1 -> 3`, `decimation 4 -> 0` — which is the regression argument
(the old fixed 3 was only ever the `decimation 1` answer).

## The change

- `--camera-frame-skip` default is `None` = derived at env-cfg-resolution
  time (after `--decimation` / the env cfg settle), from the live
  `env_cfg.sim.dt x decimation`. An explicit value still wins verbatim
  (floored at 0 to match the publisher), with a startup warning if it implies
  a publish rate off the policy period.
- `POLICY_PERIOD_S` comes from `strafer_shared.constants` (the same constant
  the obs-parity join tolerance uses); no `30` literal.
- **Startup contract print**: effective publish rate (sim Hz), the policy
  period, the resolved/derived frame_skip, bridge tick rate, and renders per
  tick — so this class of fault can never hide again. A `render_interval` too
  coarse for the cadence (a fresh render doesn't advance between publishes)
  warns about duplicate frames.
- Help text, the `bridge/config.py` `camera_frame_skip` comment, the async
  publisher's `frame_skip` docstring, and `bridge-runtime-invariants.md` all
  updated off the stale "matches `sim.render_interval`" framing.

## Out of scope

The async publisher's `frame_skip` mechanics, the OmniGraph path, the
freshness gate, and the Jetson side are untouched — this only fixes which
value the runner hands the publisher by default.

## Relationship to `bridge-publish-rate-decouple`

This delivers the **camera-cadence slice** of
[`bridge-publish-rate-decouple`](../active/sim-performance/bridge-publish-rate-decouple.md):
the default publish cadence is now correct for any decimation. The deeper
**action-hold / sub-step decouple** — publishing *more often* than the
control step so `decimation 4` dynamics and a ~30 Hz publish rate can coexist
architecturally — stays parked in that brief.

## Tests + gates

- Kit-free unit (`test_sim_in_the_loop_camera_cadence.py`, 18/18): the
  derivation matrix (`(1/120,1)->3`, `(dec 4)->0`, `dec 6/8` floor, a
  non-1/120 `sim.dt` proving it isn't hardcoded, a policy-period override),
  explicit passthrough / explicit-zero-vs-unset / negative-floor, the
  `--camera-frame-skip` `default=None` sentinel asserted against the real
  parser (a silent revert to a fixed default fails here), and both warning
  conditions (off-policy explicit skip, duplicate frames, and their
  independent co-occurrence).
- Lab pure suite green via `make test-lab-pure`: 788 passed / 1 skipped.
- Kit smoke [operator-timed]: a `--decimation 4` bridge run WITHOUT the flag —
  the startup contract print shows `publish 30.00 Hz sim` and the node-side
  cadence report shows the 33.3 ms sim spacing (the corrected-cadence re-run
  that already used `--camera-frame-skip 0` explicitly doubles as this gate;
  this just confirms the derived default matches).
