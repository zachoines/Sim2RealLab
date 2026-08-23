# Stale junit XMLs — recovered from the fixed output paths, produced by the 2026-08-14 baseline

`run_tests.py` writes each suite's junit XML to a fixed path under `source/strafer_lab/` and
**does not unlink it first**. When a suite times out, `_run_subprocess` kills the child before
pytest writes results, so no new XML appears — and whatever file the *previous* session left at
that path survives untouched.

Both files here were picked up that way while preserving this session's failing artifacts. Their
internal `timestamp` attributes give them away:

| file | timestamp | tests | what it actually is |
|---|---|---|---|
| `kit-suite-env--STALE-2026-08-14-baseline.xml` | `2026-08-14T15:48:18` | 268 | the pre-bump baseline's env run |
| `kit-suite-depth_noise_test_holes--STALE-2026-08-14-baseline.xml` | `2026-08-14T15:52:57` | 2 | the pre-bump baseline's holes run |

They are held here, out of the `-FAILRUN` set, so that set sums only to what this session ran.
Had they been left in place they would have contributed 270 passing tests to a run that
collected neither suite.

This generalises the hazard recorded at §11.8 of the scoping report. That note covers a *rerun*
overwriting a failing artifact; this is the mirror case — a *timeout* leaving the previous
session's artifact in place, where the XML set then reports an older run's result under this
run's name. The durable fix is for `run_tests.py` to unlink a suite's XML before launching it,
so an absent result can never read as a stale pass. That is an instrument change and belongs to
a later PR, not to this measurement record.
