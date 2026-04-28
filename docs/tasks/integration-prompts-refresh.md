# Refresh integration prompts + sim-in-the-loop runbook

**Type:** docs / refresh
**Owner:** Operator-driven; could delegate to either a DGX or Jetson agent
to do the half they own and reconcile in a single PR
**Priority:** P1 (blocks the next end-to-end integration round; not blocking
ongoing bug-fix work)
**Estimate:** M (~half-day per integration prompt + the runbook)
**Branch:** task/integration-prompts-refresh

## Story

As an **operator preparing to spin up a fresh end-to-end integration test
(bridge + CLIP/VLM data collection)**, I want **the three INTEGRATION_*
documents to reflect the current state of the codebase, not the state at
the original integration round**, so that **handing them to a fresh
agent yields correct context the agent can execute against without me
correcting drift mid-run**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](context/bridge-runtime-invariants.md)

The integration docs being refreshed are themselves *prompts that
duplicate what context modules now own*. A reasonable shape for the
refreshed prompts is to read those modules in turn (rather than re-state
their contents inline), making the prompts shorter and self-correcting
as modules update.

## Context

The three integration docs are the prompts that bootstrap fresh DGX
and Jetson agents into an integration test:

- `docs/INTEGRATION_PROMPT_DGX.md` — DGX-side agent context.
- `docs/INTEGRATION_PROMPT_JETSON.md` — Jetson-side agent context.
- `docs/INTEGRATION_SIM_IN_THE_LOOP.md` — the runbook both agents
  follow, with stage-by-stage commands.

They've drifted since their original write-up because the system
moved on:

- **Conda env name**: 5+ references to `env_phase15`; current name is
  `env_isaaclab3` (Isaac Lab 3.0 / Isaac Sim 6 migration).
- **Bridge runtime invariants**:
  - `--profile` harness on `run_sim_in_the_loop.py` (commit `70c4ba9`)
    for phase-level perf attribution.
  - cmd_vel normalization in both bridge paths (commits `d642bff`,
    `70c4ba9`) — Twist values now divided by `MAX_LINEAR_VEL` /
    `MAX_ANGULAR_VEL` before writing to the action tensor.
  - Sim-time-aware navigation timeout in the executor (`f60456e`) +
    `STRAFER_NAVIGATION_TIMEOUT_S` env var.
- **Bridge-mode default for missions**:
  - Phase-level profiling found `--viz kit` editor viewport adds
    ~96 ms / loop. `make sim-bridge` (headless) is now the
    recommended daily-driver for missions; `make sim-bridge-gui` is
    for visual debugging only. The runbook's stage-by-stage commands
    should reflect that. Visual debugging from the headless mode
    needs a Jetson-side viewer (see
    [jetson-headless-viewer.md](jetson-headless-viewer.md)).
- **Scene-side fixes that affect bridge runs**:
  - Infinigen floor mesh colliders are stripped at bake time
    (`6f9976c`); the env's `lift_ground_plane_to_floor` startup event
    raises `/World/ground` to floor height. Operators no longer see
    the "robot bounces / sinks into the floor" failure mode the
    original integration runbook documented.
  - D555 renderer frustum clip decoupled from depth sensor limit
    (`0101232`) — RGB now extends through the full room, not just
    6 m.
- **CLIP/VLM data collection track**:
  - The integration round this refresh enables is not just
    bridge mission-execution — it ALSO covers VLM grounding +
    perception data collection through the same bridge. Stage-list
    likely needs a new stage (or an explicit branch in stage 4)
    covering: VLM grounding flow over the bridge, scene metadata
    extraction integration, and the
    `prepare_vlm_finetune_data.py` / `finetune_clip.py` end-to-end
    path. Today's runbook only covers bridge-driven Nav2 missions.
- **New artifacts the prompts should reference**:
  - [`docs/example_commands_cheatsheet.md`](../example_commands_cheatsheet.md)
    for the exact one-liners operators run on the DGX.
  - [`docs/PERF_INVESTIGATION_SIM_IN_THE_LOOP.md`](../PERF_INVESTIGATION_SIM_IN_THE_LOOP.md)
    Findings 8-10 — current bridge perf attribution + recommended
    knobs.
  - [`docs/tasks/`](.) — the queue of follow-up work the agents
    should know about (so they don't independently rediscover the
    same issues).

## Acceptance criteria

- [ ] Conda env references updated repo-wide: `env_phase15` →
      `env_isaaclab3`. Sanity-grep returns no `env_phase15` outside
      historical perf-doc text.
- [ ] `INTEGRATION_PROMPT_DGX.md` ownership boundaries match the
      current state of the lanes. Source of truth for the lanes is
      [`context/ownership-boundaries.md`](context/ownership-boundaries.md);
      either link out to it from the prompt, or replicate inline and
      keep both in sync.
- [ ] `INTEGRATION_SIM_IN_THE_LOOP.md` stage-by-stage commands point
      at `make sim-bridge` (headless) by default, with a clear
      callout for when to use `--viz kit` (debugging only). Include
      the new `--profile` flag at least once in the perf-debug
      section.
- [ ] Runbook's "Stage 4 — Manual mission submission" or follow-on
      stage exercises the cmd_vel normalization explicitly: a Nav2
      `cmd_vel` of e.g. 0.5 m/s should produce ~0.5 m/s body velocity
      in sim, not the old 0.78 m/s. Add an `ros2 topic echo /cmd_vel`
      check + an `unwrapped.scene["robot"].data.root_lin_vel_b`
      cross-check.
- [ ] New stage (or explicit Stage 4 / Stage 5 branch) covers the
      CLIP/VLM data-collection path through the bridge:
      `extract_scene_metadata.py` → `generate_scenes_metadata.py`
      → bridge harness → `prepare_vlm_finetune_data.py` →
      `finetune_clip.py`. Reference the same stage-by-stage
      "operator runs / agent verifies" cadence as the existing
      stages.
- [ ] Cross-references to deleted docs are gone:
      `VALIDATE_ISAAC_SIM_AND_INFINIGEN.md` and the `archive/`
      directory no longer exist; previous links to them have been
      replaced with self-contained prereqs (already done in commit
      `ded56ea`, but verify nothing was missed).
- [ ] Each prompt links to `docs/example_commands_cheatsheet.md` as
      the canonical place for commands operators copy-paste.
- [ ] Each prompt links to `docs/tasks/` and explicitly tells the
      agent: "don't rediscover known issues — check this directory
      first." Include the current task list (kit-pump-redundancy,
      async-camera-publishers, jetson-headless-viewer,
      integration-prompts-refresh itself can be omitted as the
      authoring task) so agents have the live queue.
- [ ] First-action sections at the bottom of the two PROMPT docs
      tell the agent the actual current branch state (last few
      commits on `phase_15-isaaclab3`, what's known-good vs
      in-progress).
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- Recent perf attribution: `docs/PERF_INVESTIGATION_SIM_IN_THE_LOOP.md`
  Findings 8-10 (the camera-bridge-on/off + headed/headless decomposition).
- Recent cmd_vel fix rationale: captured in
  [`context/bridge-runtime-invariants.md`](context/bridge-runtime-invariants.md#cmd_vel-normalization-contract-both-paths).
  The original session brief (`dgx_scratch_pad.md`) has been deleted;
  the invariant lives in context now.
- Task-brief format + composition guidance: see
  [`docs/tasks/README.md`](README.md). Match that voice / structure
  for any new stage briefs you fold into the runbook.
- Source of truth for the cheatsheet content:
  [`docs/example_commands_cheatsheet.md`](../example_commands_cheatsheet.md)
  — gets updated as commands evolve, mirror anything in the runbook
  against it.

## Out of scope

- Modifying the integration prompts to cover **real-robot** bringup.
  These three docs are sim-in-the-loop-specific. Real-robot bringup
  has its own launch files (`autonomy.launch.py`, `base.launch.py`,
  etc.) and would need a separate runbook if/when that integration
  round happens.
- Refactoring the integration runbook itself (stage structure,
  shell-by-shell command layout). Just refresh the contents of the
  existing structure unless drift is severe enough to warrant a
  larger change.
- Editing the underlying scripts (`run_sim_in_the_loop.py`,
  `bringup_sim_in_the_loop.launch.py`, etc.) — this is a docs-only
  task. If a script needs to change to make the runbook work, that's
  a separate ticket.
