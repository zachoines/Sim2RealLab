# Cut the sim-bridge cheatsheet to commands and gotchas

**Type:** docs
**Owner:** Either (docs only; no hardware)
**Priority:** P2 — the cheatsheet is the sim-bridge lane's runbook and is read at every rig
session. About 80 of its 240 lines explain a mechanism, retell an incident or quote a
measurement, so the step and its pitfall are buried in prose that other docs already hold. It is
P2 rather than P3 because one of those passages misstated which service the lane's canon env
reaches, and that is the kind of error a runbook reader acts on.
**Estimate:** M (the inventory below exists; each passage needs its home checked before it goes)
**Branch:** `task/sim-bridge-cheatsheet-trim`

## Story

As **someone bringing up the sim-bridge stack at the terminal**, I want **the cheatsheet to be
short descriptions, gotchas, and commands that paste as they are**, so that **I find the step and
its pitfall without reading the mechanism behind it.**

## Context bundle

- [context/README.md](../../context/README.md) — what belongs in a context module and what
  belongs in the cheatsheet (operational facts at the keyboard).
- [context/conventions.md](../../context/conventions.md) — the user-facing documentation surface
  table, which lists `example_commands_cheatsheet.md` but not this file.
- [context/deploy-env-config.md](../../context/deploy-env-config.md) — the env chain the
  cheatsheet's config bullet summarises.
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

The shape to reach: every item is a command block that pastes as it is, or a description or
gotcha of at most three lines. Mechanism, history and measurements live in the doc that owns the
subject, linked once. Review of
[#235](https://github.com/zachoines/Sim2RealLab/pull/235) asked for this, and #235 started on the
policy-goal bullet:
- the budgets were deleted rather than moved, since `bridge-runtime-invariants` already states
  them, and the bullet now links it;
- the sim-time requirement became a gotcha with a `printenv` check;
- the canon bullet was corrected. It had said `sim_bridge.env` loads "on top of `autonomy.env`",
  which reads as reaching the executor. Only `inference` gets it.

**Passages to cut, at `docs/sim_bridge_autonomy_cheatsheet.md` after #235** (line numbers will
move as the work proceeds):

| Lines | Passage | Already covered at, or proposed home |
|---|---|---|
| 17-23 | What the Kit boot watchdog does | `tools/kit_boot_watchdog.sh` header; `docs/example_commands_cheatsheet.md` |
| 25-27 | `PYTHONUNBUFFERED=1` is load-bearing when redirected | Possibly obsolete: the launch line exports it, and the watchdog sets it for its child (`tools/kit_boot_watchdog.sh`:191-195). Keep the cadence-print check (`frame_skip=3 (derived, derived 3)` / `publish 30.00 Hz sim`) that precedes it |
| 29-39 | Why not `--decimation 4 --render-interval 4`, with rates and quotes | [`enriched-lane-rig-stability`](../reliability/enriched-lane-rig-stability.md) mode 4; keep the one-line warning |
| 84-93 | The rtabmap `addLink()` FATAL mechanism and `stop_grace_period` history, in the command block's comments | `enriched-lane-rig-stability` mode 1; keep "aborts again within ~1 s → bump the token" and the `loadDataFromDb() ... repair` check |
| 116-121 | Why both hosts must be wired | [`sim-bridge-link-transport-capacity`](../reliability/sim-bridge-link-transport-capacity.md); keep the first line and the check |
| 135-160 | What `configure_inference.sh` does; recreating `inference` by hand; swapping the model by hand | `source/strafer_ros/deploy/tools/configure_inference.sh` header; `source/strafer_ros/deploy/README.md`; keep "recreating `inference` by hand reverts the anchoring" |
| 161-165 | Bump the SLAM scene token | Duplicates the NX block's step 1b (65-70); keep one |
| 170-181 | Where the policy config comes from | `context/deploy-env-config.md`; `source/strafer_ros/deploy/README.md`. Both say the lane loads `[autonomy.env, sim_bridge.env]` without saying it is `inference` only |
| 198-203, 218-221, 226-228 | The costmap-halo escapes on both lanes, and the measured 0.20 / 0.15 m | Hybrid lane: the `subgoal_generator_node.py` module docstring and `strafer_inference/config/subgoal_generator.yaml`. Nav2 lane: `source/strafer_ros/README.md`; [`nav2-lane-inflated-start-recovery`](../../completed/nav2-lane-inflated-start-recovery.md). Keep the log lines, the probe command and the manual-strafe recovery, and keep the hold defined while the kept lines refer to it |

Inbound links: six docs link the file, none by anchor.
- `jetson-untether.md`:124 cites it by line number (`:56-61, :74-79`).
- `enriched-lane-rig-stability.md`:425 has an open criterion that each of its four failure modes
  has "a documented detect-and-recover procedure" in this file.

## Acceptance criteria

- [ ] The cheatsheet states its shape at the top: commands, and descriptions or gotchas of at
      most three lines, each linking its long form once. `conventions.md`'s user-facing
      surface table gains a row for this file.
- [ ] Each passage in the table is reduced to its gotcha and command, or deleted because its home
      already says it or it no longer applies. This brief records the outcome for each row.
- [ ] Every item that is not a command block is at most three lines, including items the table
      does not list. This brief records any item kept longer, and why.
- [ ] Prose that has no other home moves to the doc that owns its subject, not to a new doc.
- [ ] Every command block is unchanged, or this brief records why it changed.
- [ ] `context/deploy-env-config.md`, `source/strafer_ros/deploy/README.md`, the
      `deploy/tests/gen_env.py` docstring (:19-22) and the comments in
      `strafer_bringup/config/env_sim_bridge.env` (:5-6, :13-14) say that `sim_bridge.env`
      reaches the inference service only.
- [ ] `jetson-untether.md`:124 points at the passages it means by name, not by line number.
- [ ] `enriched-lane-rig-stability`'s detect-and-recover procedures are still in the cheatsheet,
      as commands.
- [ ] `tools/check_brief_links.py` reports no broken link that `main` does not.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Out of scope

- `docs/example_commands_cheatsheet.md`.
- Changing a procedure, a tracked config's values or a script's behaviour. Comment and docstring
  corrections are in scope.
- Giving the executor this lane's sim-time setting from a tracked file. That would retire the
  overlay gotcha. [`jetson-untether`](../reliability/jetson-untether.md) records the overlay's
  keys.
- The context modules' own length.
