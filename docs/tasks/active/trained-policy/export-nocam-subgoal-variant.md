# Enable `NOCAM_SUBGOAL` export through `export_policy.py`

**Type:** task / feature
**Owner:** DGX (`strafer_lab` lane)
**Priority:** P1 — unblocks the deployable `NOCAM_SUBGOAL` artifact the parked Jetson-lane hybrid runtime consumes.
**Estimate:** S
**Branch:** task/export-nocam-subgoal-variant

## Story

As an **operator promoting the converged `NOCAM_SUBGOAL` checkpoint to robot
deployment**, I want **`export_policy.py --variant NOCAM_SUBGOAL` to emit a
correctly-labeled `.pt`/`.onnx` + sidecar**, so that **the Jetson inference
node configured for `NOCAM_SUBGOAL` loads the artifact instead of refusing a
mislabeled sidecar**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md) — this
  is a DGX-only change; the variant is already first-class in the shared
  `strafer_shared` boundary, which stays untouched.
- [context/branching-and-prs.md](../context/branching-and-prs.md)

## Context

`PolicyVariant.NOCAM_SUBGOAL` is a first-class 19-dim, **camera-less** variant
that mirrors `NOCAM`'s shapes and scales with renamed `subgoal_*` observation
keys (the goal-shaped fields refer to a rolling subgoal pose on a planner path
rather than a final goal). A converged checkpoint exists. The variant — and its
sidecar-label contract — already live in `strafer_shared.policy_interface`; only
the **export CLI** could not reach it:

- `export_policy.py` derives its `--variant` argparse `choices` live from
  `sorted(_DEFAULT_ENV_BY_VARIANT.keys())`, and that map held only `NOCAM` and
  `DEPTH`, so argparse rejected `--variant NOCAM_SUBGOAL`.
- The naive workaround (`--variant NOCAM` + an `--env` override) writes a
  sidecar stamped `policy_variant=NOCAM`. That label is load-bearing:
  `strafer_shared.policy_interface.load_policy()` raises `ValueError` when the
  sidecar variant disagrees with the requested variant, so the Jetson node
  configured for `NOCAM_SUBGOAL` would refuse the artifact. The export must
  stamp `policy_variant=NOCAM_SUBGOAL`, which requires real CLI support.
- The camera-enable heuristic in `main()` keyed on the literal string `"NOCAM"`
  (`if args.variant != "NOCAM" or args.num_envs > 1`). Because
  `NOCAM_SUBGOAL != "NOCAM"`, it would wrongly force cameras on for a
  camera-less env (19-dim obs, no camera/depth obs term, no camera prim in the
  scene). It is factored into a module-level `_should_enable_cameras` predicate
  keyed on a camera-less variant **set**, so it is unit-testable without a Kit
  boot (`main()` imports `isaaclab.app` before parsing args) and a future
  camera-less variant can't silently regress.

The default env id `Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-v0` is the
realistic-DR play env, registered character-for-character in
`source/strafer_lab/strafer_lab/tasks/navigation/__init__.py`. This rides the
same export chain as the sibling [`policy-export-deprecation-migration`](../active/trained-policy/policy-export-deprecation-migration.md)
(precedent: [`export-onnx-depth`](export-onnx-depth.md)); the produced artifact
is the input the parked Jetson-lane [`hybrid-mode`](../parked/trained-policy/hybrid-mode.md)
+ [`strafer-hybrid-sim-validation`](../parked/trained-policy/strafer-hybrid-sim-validation.md)
briefs consume (the `NOCAM_SUBGOAL` checkpoint itself came from
[`subgoal-env`](subgoal-env.md); the loader/runtime from
[`inference-package`](inference-package.md)).

## Acceptance criteria

- [ ] `_DEFAULT_ENV_BY_VARIANT["NOCAM_SUBGOAL"] == "Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-v0"`,
      which auto-widens the parser's `--variant choices` (derived from the map
      keys) so the CLI accepts `--variant NOCAM_SUBGOAL`.
- [ ] `_should_enable_cameras(...)` truth table pins both halves of the
      boolean: `("NOCAM_SUBGOAL", 1) -> False` (the fix), `("NOCAM_SUBGOAL", 2)
      -> True`, `("DEPTH", 1) -> True`, `("NOCAM", 1) -> False` and
      `("NOCAM", 2) -> True` (existing behavior preserved).
- [ ] Sidecar label flow-through: a sidecar written with
      `policy_variant="NOCAM_SUBGOAL"` reads back `policy_variant ==
      "NOCAM_SUBGOAL"`, `obs_dim == 19`, `is_recurrent is False` — closing the
      silent-mislabel gap without a Kit boot.
- [ ] `make test-lab-pure` is green (the new coverage lives in
      `source/strafer_lab/tests/policy_tooling/test_export_policy.py`).
- [ ] Operator-run, after the GPU/Kit export: the sidecar at
      `models/strafer_nocam_subgoal_v0.json` shows `policy_variant ==
      "NOCAM_SUBGOAL"`, `obs_dim == 19`, `is_recurrent == false`, `env_id ==
      "Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-v0"`, and
      `load_policy(..., variant=PolicyVariant.NOCAM_SUBGOAL)` round-trips
      without the mismatch `ValueError`.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See [`conventions.md`'s user-facing documentation
      maintenance section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Out of scope

- The entire Jetson runtime: `inference_node.py` variant-awareness, the
  deploy-side rolling-subgoal generator, the third `hybrid_nav2_strafer` backend
  in `ros_client.py`, the costmap-freshness watchdog. These are downstream
  Jetson-lane work behind [`hybrid-mode`](../parked/trained-policy/hybrid-mode.md).
- Any edit to `source/strafer_shared/` — `PolicyVariant.NOCAM_SUBGOAL` and the
  obs contract already support the variant there (Jetson-owned, append-only
  boundary).
- The stale Kit-only `source/strafer_lab/test_sim/env/test_env_registration.py`
  (`EXPECTED_ENVS` omits the Subgoal ids — pre-existing test-vs-code drift,
  independent of this change).
- Actually running the GPU/Kit export — operator-run.
