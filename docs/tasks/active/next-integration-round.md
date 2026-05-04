# Run the next end-to-end sim-in-the-loop integration round

**Type:** task / integration test
**Owner:** Either (one round, two parallel agents — one DGX, one
Jetson — with the operator relaying observations between them)
**Priority:** P1 (gating signal that the bridge + autonomy + VLM/CLIP
pipelines compose end-to-end)
**Estimate:** M-L (~1 full day with both hosts ready; longer if the
round surfaces bugs that need filing as follow-up briefs)
**Branch:** task/next-integration-round

## Story

As an **operator who has just landed a batch of bridge / autonomy /
VLM-CLIP infrastructure work**, I want **a fresh end-to-end run of
[`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md)
across both hosts**, so that **integration regressions are caught
against the runbook's known-good baseline before the next deployable
round of work starts** — and so that any bugs surfaced get filed as
their own follow-up briefs in the same session, rather than
rediscovered later.

## Context bundle

Read these before starting:
- [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md)
  — the runbook itself; this brief is the "schedule a run" wrapper.
- [`context/repo-topology.md`](../context/repo-topology.md) — hosts,
  IPs, repo paths, conda envs, ROS distro / DDS / domain.
- [`context/ownership-boundaries.md`](../context/ownership-boundaries.md)
  — what each agent may edit. Critical because two agents are active
  at the same time on adjacent code.
- [`context/bridge-runtime-invariants.md`](../context/bridge-runtime-invariants.md)
  — cmd_vel normalization, telemetry/camera split, headless vs
  `--viz kit`, `--profile` harness + reference per-phase numbers,
  scene-side prerequisites.
- [`context/branching-and-prs.md`](../context/branching-and-prs.md)
  — branch off `main`, one brief → one branch → one PR.
- [`docs/example_commands_cheatsheet.md`](../../../docs/example_commands_cheatsheet.md)
  — canonical operator one-liners (copy-paste from here, don't
  re-derive).
- [`BOARD.md`](../BOARD.md) — current active queue. **Don't
  rediscover known issues** — if a symptom you see during the run
  matches an open brief, leave it alone and report; don't fold a
  separate fix into this round's PR.

## Context

This brief is **scheduling work**, not implementation work. Its
output is:

1. A clean pass through Stages 1-6 of the runbook on the current
   tip of `main`.
2. A short report from each agent (DGX-side + Jetson-side) covering
   what passed, what failed, what was changed (if anything), and
   what the operator should ask the peer agent to verify.
3. New follow-up briefs filed under
   [`docs/tasks/active/`](.) for every bug surfaced that wasn't
   already in the queue.

### Agent-role split

The runbook's stages are owned by hosts, not by this brief. When the
operator launches two parallel agents against this brief, each
should treat the runbook as authoritative and act within their
ownership lane:

- **DGX agent** drives Stage 2 (bridge alone), Stage 5 (harness
  mode), and Stage 6 (VLM/CLIP data-collection sweep). Mostly
  passive on Stage 3.
- **Jetson agent** drives Stage 3 (bringup consumes the bridge),
  Stage 3.5 (Foxglove-over-SSH), and is the operator-facing
  half of Stage 4 (mission CLI). Mostly passive on Stage 2 and 5.
- **Both** participate in Stage 1 (DDS discovery) and Stage 4
  (manual mission); the cmd_vel normalization cross-check in
  Stage 4 is DGX-side.

If a fix needs to cross the lane line, **stop and report** rather
than reaching across. The operator relays. The convention exists
because two agents working the same files at once is the failure
mode that motivated the lane split in the first place.

### Prerequisite infra

This brief assumes the following are in place at the time it's
picked up:

- **Bridge** — `make sim-bridge` boots on the DGX without errors
  and reaches the env-step loop (Stage 2 prerequisite).
- **VLM service** — `make serve-vlm` reachable from the Jetson at
  `http://192.168.50.196:8100/health` (Stage 4 prerequisite).
- **Planner service** — `make serve-planner` reachable from the
  Jetson at `http://192.168.50.196:8200/health` (Stage 4
  prerequisite).
- **Jetson bringup** — `colcon build` of `strafer_ros` is current,
  `bringup_sim_in_the_loop.launch.py` boots without errors (Stage 3
  prerequisite).
- **Generated scene with metadata** — at least one Infinigen scene
  under `Assets/generated/scenes/<scene_name>/` with both
  per-scene `scene_metadata.json` and the combined
  `scenes_metadata.json` present (Stage 5 + 6 prerequisite). If
  not, run the metadata authoring commands in Stage 6 first.

If any of those are red, fix the underlying issue (or file a
prerequisite brief) before starting Stage 1.

## Acceptance criteria

- [ ] **Prerequisites green.** All four checklist sections at the
      top of the runbook (`Both hosts`, `DGX only`, `Jetson only`,
      and the prerequisite-infra list above) pass before Stage 1.
- [ ] **Stages 1-5 pass.** Each go/no-go check in
      [`INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md)
      Stages 1, 2, 3, 3.5, 4, 5 succeeds on the current `main`.
- [ ] **cmd_vel normalization cross-check holds in Stage 4.** Body
      `root_lin_vel_b` magnitude tracks the published
      `/cmd_vel.linear.x` within ~10 % at steady state. A ~1.57×
      regression mode is filed as a P0 follow-up brief immediately.
- [ ] **Stage 6 produces a populated SFT JSONL + CLIP ONNX pair.**
      `data/vlm_sft/<scene_name>/grounding.jsonl` has ≥ one example
      per object label; `models/clip_<scene_name>/clip_visual.onnx`
      and `clip_text.onnx` are written; the `finetune_clip.py`
      MLflow run records non-NaN contrastive loss decreasing across
      epochs.
- [ ] **Bug reports filed.** Any new symptoms surfaced during the
      run that don't already match an open brief are filed as new
      `docs/tasks/active/<slug>.md` briefs in the same PR, per the
      format in [`README.md`](../README.md). Link them from this
      brief's `Follow-ups:` stamp on ship.
- [ ] **No work that crosses lane lines lands in this PR.** Any
      cross-lane fix discovered during the run is filed as a
      separate brief; the integration round's own PR contains only
      the bug reports and any explicit "no-fix-yet" notes.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- The runbook's per-stage troubleshooting tables are the first
  place to look for any failure. They map symptoms to the most
  likely cause and pointer to the relevant code site.
- Bridge perf regressions: re-run with `--profile` and compare to
  the per-phase reference numbers in
  [`bridge-runtime-invariants.md`](../context/bridge-runtime-invariants.md#phase-level-profiler---profile).
- VLM/CLIP pipeline: the Stage 6 troubleshooting table covers the
  common shapes (per-scene metadata empty, frames at wrong
  resolution, MLflow loss = NaN).

## Out of scope

- **Real-robot bringup.** This runbook is sim-in-the-loop only;
  real-robot bringup uses a different launch chain
  (`autonomy.launch.py`, `base.launch.py`) and is a separate
  integration round.
- **Refactoring the runbook.** If the run surfaces drift in the
  runbook itself (e.g., a stage's command is now wrong), file a
  separate `docs/tasks/active/integration-runbook-refresh-<NN>.md`
  brief — don't fold runbook edits into this round's PR.
- **Bug fixes for issues surfaced during the run.** The PR for
  this brief contains the run's report + new briefs only. Each
  surfaced bug gets its own one-brief-one-branch-one-PR life cycle.
- **Tuning passes** (RL training reruns, planner-prompt iteration,
  VLM fine-tunes against newly-collected data). Those are
  downstream of this brief's data-collection step, not part of it.
