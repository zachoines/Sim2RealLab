# Make the deployed-artifact tests find their artifact instead of skipping

**Type:** bug
**Owner:** Jetson agent
**Priority:** P3
**Estimate:** S–M
**Branch:** task/deployed-artifact-test-discovery

## Story

As **the inference runtime suite**, I want **the tests that load a real deployed
policy to resolve the artifact wherever it actually lives**, so that **the only
tests exercising a genuine ONNX artifact stop silently skipping on the machines
that have one and the checkouts that do not.**

## Context bundle

- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/deploy-env-config.md](../../context/deploy-env-config.md)
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md)

## Context

Three tests in
[`source/strafer_ros/strafer_inference/test/test_inference_runtime.py`](../../../../source/strafer_ros/strafer_inference/test/test_inference_runtime.py)
derive `<repo-root>/models/<artifact>.onnx` from `__file__` and call `skipTest`
when it is absent:

- `TestOnnxThreadPinning::test_default_pins_to_one_thread` (`:380`)
- `TestOnnxThreadPinning::test_explicit_thread_count_is_applied` (`:394`)
- `TestDepthSubgoalArtifactLoads::test_loads_recurrent_and_infers` (`:421`)

These are the only tests that exercise a real deployed artifact rather than a
synthetic one — the recurrent-load path, the observation-dimension contract, and
the ORT intra-op thread pin that keeps a ~50 µs MLP from starving RTAB-Map and
Nav2 on the Jetson.

`models/` is **untracked but not gitignored**, so whether they run is a property
of the checkout: on a checkout where artifacts have been placed they execute; in
a git worktree or a fresh clone they skip. Nothing reports the difference, so a
green run means either "the contract holds" or "nothing was checked", and the
two are indistinguishable from the output.

This is the same defect class as the scene-corpus leak fixed for the sim lane
(PR #204), but it needs a different remedy. Mocking the artifact would defeat
tests whose entire purpose is loading a real one, so the answer is making the
artifact path discoverable — an env var, a conftest-resolved fixture, a search
over known locations — plus a deliberate decision about what should happen when
no artifact is found anywhere.

That decision is the design question this brief exists to settle, and it belongs
to the ROS lane: a skip is right for a contributor with no artifacts, and wrong
for a Jetson release gate that must not report green without having loaded one.

## Acceptance criteria

- [ ] The artifact path is discoverable rather than derived from `__file__`
      alone — the mechanism is the lane's call, but it must resolve for a
      checkout, a worktree, and a Jetson deployment without editing the test.
- [ ] A run that skips these tests says so distinguishably from a run that
      executed them; a release-gate invocation can require them to have run.
- [ ] `make test-ros` passes on a host that has the artifacts and on one that
      does not, with the difference visible in the output.
- [ ] No production module changes to accommodate the tests.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit.
- [ ] No regression in `make test-ros`.

## Investigation pointers

- `source/strafer_ros/strafer_inference/test/test_inference_runtime.py:374-434` —
  the three `_model_path` helpers and their `skipTest` guards.
- `models/strafer_nocam_subgoal_v0.onnx`, `models/strafer_depth_subgoal_v0.onnx` —
  the two artifacts named.
- `source/strafer_shared/policy_interface.py` — `load_policy` / `PolicyVariant`,
  what the tests exercise.
- `source/strafer_lab/test_sim/common/scenes.py` — the sim lane's answer to the
  same defect class, for contrast. It mocks; this one must not.

## Out of scope

- Any change to `source/strafer_lab/` test hermeticity — done in PR #204.
- Committing model artifacts to the repository.
- Rewriting these tests to use a synthetic artifact, which would delete the
  coverage this brief exists to restore.
