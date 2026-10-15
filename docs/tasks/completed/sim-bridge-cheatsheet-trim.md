# Cut the sim-bridge cheatsheet to commands and gotchas

**Status:** Shipped 2026-10-14 in `cf0554a` and `ad01630` (Either; shipped from Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/244

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

- [context/README.md](../context/README.md) — what belongs in a context module and what
  belongs in the cheatsheet (operational facts at the keyboard).
- [context/conventions.md](../context/conventions.md) — the user-facing documentation surface
  table, which lists `example_commands_cheatsheet.md` but not this file.
- [context/deploy-env-config.md](../context/deploy-env-config.md) — the env chain the
  cheatsheet's config bullet summarises.
- [context/branching-and-prs.md](../context/branching-and-prs.md)

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
| 29-39 | Why not `--decimation 4 --render-interval 4`, with rates and quotes | [`enriched-lane-rig-stability`](../active/reliability/enriched-lane-rig-stability.md) mode 4; keep the one-line warning |
| 84-93 | The rtabmap `addLink()` FATAL mechanism and `stop_grace_period` history, in the command block's comments | `enriched-lane-rig-stability` mode 1; keep "aborts again within ~1 s → bump the token" and the `loadDataFromDb() ... repair` check |
| 116-121 | Why both hosts must be wired | [`sim-bridge-link-transport-capacity`](../active/reliability/sim-bridge-link-transport-capacity.md); keep the first line and the check |
| 135-160 | What `configure_inference.sh` does; recreating `inference` by hand; swapping the model by hand | `source/strafer_ros/deploy/tools/configure_inference.sh` header; `source/strafer_ros/deploy/README.md`; keep "recreating `inference` by hand reverts the anchoring" |
| 161-165 | Bump the SLAM scene token | Duplicates the NX block's step 1b (65-70); keep one |
| 170-181 | Where the policy config comes from | `context/deploy-env-config.md`; `source/strafer_ros/deploy/README.md`. Both say the lane loads `[autonomy.env, sim_bridge.env]` without saying it is `inference` only |
| 198-203, 218-221, 226-228 | The costmap-halo escapes on both lanes, and the measured 0.20 / 0.15 m | Hybrid lane: the `subgoal_generator_node.py` module docstring and `strafer_inference/config/subgoal_generator.yaml`. Nav2 lane: `source/strafer_ros/README.md`; [`nav2-lane-inflated-start-recovery`](nav2-lane-inflated-start-recovery.md). Keep the log lines, the probe command and the manual-strafe recovery, and keep the hold defined while the kept lines refer to it |

**Outcome, 2026-10-14** (PR #244; the cheatsheet went from 240 lines to 193):

| Lines | Outcome |
|---|---|
| 17-23 | A three-line gotcha linking `tools/kit_boot_watchdog.sh`: it relaunches a stalled Kit boot, prints one line per attempt, and never touches a bridge once booted. It adds why the raw launch is used: `make sim-bridge` passes no `--task`. The other wrapped targets are in `docs/example_commands_cheatsheet.md`:14-15. |
| 25-27 | Deleted as no longer applying. The watchdog redirects its child and sets `PYTHONUNBUFFERED=1` itself (`kit_boot_watchdog.sh`:191-195). The cadence-print check is kept, and so is the export in the launch block. |
| 29-39 | A two-line warning: depth starves to ~0.3 Hz sim. It links mode 4, which holds the rates and the parked threads. The help-text quotes are in `run_sim_in_the_loop.py`. |
| 84-93 | Three comment lines around the unchanged `docker restart strafer_slam`: a load abort, restart once, bump the token if it aborts again within ~1 s. The `loadDataFromDb() ... repair` check is now a grep in step 3. The mechanism and the grace-period history are left to mode 1, linked once below the block. |
| 116-121 | The first line is kept, with a two-line reason linking `sim-bridge-link-transport-capacity`. The check block is kept, and its comment is corrected for an idle node. |
| 135-160 | Three items: the tool, whose header now lists its checks; the kept gotcha that recreating `inference` by hand reverts the anchoring; and the by-hand swap, linking the deploy README's "One key, one home". That section gains the untracked-overlay shadow caveat. |
| 161-165 | Deleted as a duplicate. Step 1b is the one token step, and it carries the "every sim restart / new layout / corrupts `/plan`" reason. Refusal of a mismatched db is in the deploy README's scene-keying section, which the block links. |
| 170-181 | A three-line gotcha: canon `env_sim_bridge.env` reaches `inference` only; edit it, `make env-sync`, recreate; not `compose/sim.env`. It links `deploy-env-config#lanes`. |
| 198-203, 218-221, 226-228 | One item per lane, each linking its long form. The log-line blocks and the plannability probe are kept. The hold is defined beside `Planner starvation:`. The 0.20/0.15 m measurement is left to `nav2-lane-inflated-start-recovery#found-while-verifying`, and the "neither escape plans from deep in the halo" gotcha keeps its link. The unsupported "~0.4 m" cue became "~0.2 m, own cell inscribed (≥ 253)": the footprint's inscribed radius is 0.214 m, and the BT's log line names cost 253. |

**Command blocks that changed** (all others are byte-identical):
- **The NX block.**
  - Its comments are cut as above.
  - The example token is `sim$(date +%Y%m%d-%H%M)`, not `run5`: a literal token pasted at a later sim restart would reuse the old layout's db.
  - Step 3 gains a read-only `grep 'loadDataFromDb().*repair'`, the old prose check.
- **The config-change bullet's verification** is now a block. `printenv` is unchanged, and "md5sum the artifact" became an in-container `md5sum`, run once in `strafer-gpu:humble`.
- **The pre-flight block.**
  - `<sim-host>` became the DGX's address from the Prereqs heading.
  - Its comment had read `inferences=0` with watchdog skips as "depth not arriving". When no goal is active that pair is the normal state (`watchdog.py`:95-96), so the comment now says a flat `depth rx` is the sign.
- **The tool block** calls the script by full path, once with no arguments to list models and once with an example, because `<model> <mission|rolling>` does not paste.
- **The probe and direct-goal blocks** use `docker exec strafer_navigation` in place of `docker compose -f docker-compose.yml exec navigation`. The file's last `cd` leaves the shell at the repo root, where that `-f` did not resolve.

Inbound links: six docs link the file, none by anchor.
- `jetson-untether.md`:124 cites it by line number (`:56-61, :74-79`).
- `enriched-lane-rig-stability.md`:425 has an open criterion that each of its four failure modes
  has "a documented detect-and-recover procedure" in this file.

## Acceptance criteria

- [x] The cheatsheet states its shape at the top: commands, and descriptions or gotchas of at
      most three lines, each linking its long form once. `conventions.md`'s user-facing
      surface table gains a row for this file.
      *2026-10-14: both, and `context/README.md`'s line on the cheatsheet states the shape too.*
- [x] Each passage in the table is reduced to its gotcha and command, or deleted because its home
      already says it or it no longer applies. This brief records the outcome for each row.
      *2026-10-14: the outcome table under Context.*
- [x] Every item that is not a command block is at most three lines, including items the table
      does not list. This brief records any item kept longer, and why.
      *2026-10-14: none is longer, and no prose line is over 100 columns. Checked
      mechanically.*
- [x] Prose that has no other home moves to the doc that owns its subject, not to a new doc.
      *2026-10-14: the shadow caveat to the deploy README, the tool's checks to its header, and
      restart for crash recovery to the deploy README's config-change section.*
- [x] Every command block is unchanged, or this brief records why it changed.
      *2026-10-14: seven of twelve changed, each listed under the outcome table with its
      reason; none changes a procedure.*
- [x] `context/deploy-env-config.md`, `source/strafer_ros/deploy/README.md`, the
      `deploy/tests/gen_env.py` docstring (:19-22) and the comments in
      `strafer_bringup/config/env_sim_bridge.env` (:5-6, :13-14) say that `sim_bridge.env`
      reaches the inference service only.
      *2026-10-14: all four, and `env_sim_bridge.env`'s sim-time comment as well.*
- [x] `jetson-untether.md`:124 points at the passages it means by name, not by line number.
      *2026-10-14: and its docs box's second line-number citation (`:105`, `:116-128`).*
- [x] `enriched-lane-rig-stability`'s detect-and-recover procedures are still in the cheatsheet,
      as commands.
      *2026-10-14: every one the cheatsheet had is kept: mode 1's FATAL grep, restart and token
      bump, now with the repair grep; modes 3 and 4's cadence and pre-flight checks; and mode 4's
      flag warning. The cheatsheet never had a mode 2 lifecycle check or mode 4's "one Kit
      instance" `nvidia-smi` rule. Adding them would be a procedure change, so that brief's
      criterion stays open for them.*
- [x] `tools/check_brief_links.py` reports no broken link that `main` does not.
      *2026-10-14: the same 591 as `main`; the cheatsheet's own links and anchors all resolve.*
- [x] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
      *2026-10-14: `deploy-env-config`, the deploy README, the sim-bridge overlay's model-path
      comment and `configure_inference.sh`'s header.*

## Out of scope

- `docs/example_commands_cheatsheet.md`.
- Changing a procedure, a tracked config's values or a script's behaviour. Comment and docstring
  corrections are in scope.
- Giving the executor this lane's sim-time setting from a tracked file. That would retire the
  overlay gotcha. [`jetson-untether`](../active/reliability/jetson-untether.md) records the overlay's
  keys.
- The context modules' own length.
