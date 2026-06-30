# Auto-launch `strafer_inference` + subgoal generator from the bringup

**Status:** Shipped 2026-06-29 in `23abcea` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/126

**Type:** task
**Owner:** Jetson agent
**Priority:** P2
**Estimate:** S
**Branch:** task/bringup-inference-autolaunch

## Story

As a **Jetson operator**, I want **the inference node (and, for hybrid,
the subgoal generator) to come up automatically inside `make launch-sim`
/ `make launch-autonomy`, gated on the selected nav backend**, so that
**I run one command instead of hand-launching a second `ros2 launch`
shell and hand-building a per-run config overlay**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)
- [context/conventions.md](../context/conventions.md)

## Context

The two node launch files (`strafer_inference/launch/inference.launch.py`,
`subgoal_generator.launch.py`) exist and are clean, but no bringup
includes them today — the operator has to open a separate shell and
`ros2 launch` them by hand, then hand-build a per-run config overlay for
`model_path` / `policy_variant` / `use_sim_time`.

`STRAFER_NAV_BACKEND` already drives the executor dispatch
(`strafer_autonomy/clients/ros_client.py`): `nav2` (default when unset),
`strafer_direct`, `hybrid_nav2_strafer`. The hybrid dispatch builds its
`ActionClient` against the absolute name `/strafer_inference/navigate_to_pose`.
No launch file reads the backend today. This task makes the bringup read
the **same** env var and auto-launch the matching server under it, so the
client and server stay in lockstep from one source of truth.

Silent-fallback chain to make loud: if `model_path` is the empty sentinel,
`inference_node` logs an error and returns **without advertising** the
action server; the dispatcher's `wait_for_server` then times out and the
mission silently falls back to nav2. An auto-launched policy backend with
an empty `model_path` boots every node yet silently degrades — so the
launch must emit a visible ERROR for that case.

The two validation briefs
([`strafer-direct-sim-validation`](../active/trained-policy/strafer-direct-sim-validation.md),
parked [`strafer-hybrid-sim-validation`](../parked/trained-policy/strafer-hybrid-sim-validation.md))
assume this wiring exists and hand-launch the nodes; this brief is the
wiring they depend on.

## Acceptance criteria

- [ ] Operator runs only `make launch-sim` (HIL) or `make launch-autonomy`
      (real); the inference node and (for hybrid) the subgoal generator
      come up automatically inside it — no separate manual `ros2 launch`
      shell, no hand-built per-run config overlay.
- [ ] `model_path` / `policy_variant` / `use_sim_time` are set **once**
      via env-defaulted launch args (env file feeds the arg default),
      never edited per run; the override applies last-wins over the YAML.
- [ ] Backend gating: `strafer_direct` and `hybrid_nav2_strafer` launch
      `strafer_inference`; only `hybrid_nav2_strafer` also launches
      `strafer_subgoal_generator`; `nav2` / unset launches **neither**
      (byte-identical bringup — zero behavior change).
- [ ] `use_sim_time` reaches each node as a real **bool** (bool-coerced
      inside the inference launch files), never a raw `"false"` string.
- [ ] A policy backend with an empty `STRAFER_INFERENCE_MODEL_PATH`
      produces a launch-time **ERROR** (not a silent boot); the Makefile
      targets fail-fast on the same misconfiguration.
- [ ] Inference node namespace/name stays `/strafer_inference` (the
      hybrid dispatch targets `/strafer_inference/navigate_to_pose`); the
      subgoal generator stays in the global namespace.
- [ ] Lane fence honored: no edits to `strafer_shared`, `strafer_lab`,
      the DGX planner service, or any `strafer_autonomy` dispatch
      (`ros_client.py` is reference-only).
- [ ] Gating is covered by a test for each backend on both top-level
      launch files.
- [ ] If your work invalidates a fact in any referenced context module,
      package README, top-level `Readme.md`, or guide under `docs/`,
      update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance).
- [ ] No regression in plain nav2 bringup or `make launch-nav` (unset
      backend → neither inference node launches).

## Investigation pointers

- `source/strafer_ros/strafer_inference/launch/inference.launch.py` —
  declares only `config_file` + `log_level` today; namespace
  `strafer_inference` is load-bearing.
- `source/strafer_ros/strafer_inference/launch/subgoal_generator.launch.py`
  — global namespace, policy-free.
- `source/strafer_ros/strafer_bringup/launch/bringup_sim_in_the_loop.launch.py`
  — body built imperatively in the `_launch_setup` OpaqueFunction.
- `source/strafer_ros/strafer_bringup/launch/autonomy.launch.py` — plain
  `LaunchDescription([...])`, already reads `os.environ.get(...)` at build
  time for `VLM_URL` / `PLANNER_URL`.
- `strafer_description/launch/description.launch.py` — the repo's
  `use_sim_time` bool-coercion precedent (`PythonExpression`).
- `inference_node.py` — `model_path` empty → no action server advertised.

## Out of scope

- The validation run itself (rig smoke, goal-(a) validation) — owned by
  the operator/coordinator on the validation briefs.
- The `SUBGOAL_GOAL_A_RUNBOOK.md` simplification (dropping the manual
  "Shell B") — the coordinator owns that runbook; it is not in this
  worktree.
- Any change to the executor dispatch or the backend value set.
</content>
</invoke>
