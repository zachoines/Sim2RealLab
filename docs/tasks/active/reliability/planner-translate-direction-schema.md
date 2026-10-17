# Planner reports a translate as ordered direction moves, not a signed vector

**Type:** task (planner schema)
**Owner:** maintainer + DGX agent, paired session (the sweep and its record autonomous)
**Priority:** P1 — goal-b's translate correctness. On a command that names two axes the planner
returns a component with the wrong sign on about half of the measured commands, under every prompt
tried, and the executor dispatches it faithfully: the robot goes to the mirror-image goal. The
compile-first dry-check for real-robot submissions is the fence until this ships.
**Estimate:** M (schema, parser, compiler table, prompt, tests, then the sweep and its record)
**Branch:** `task/planner-translate-direction-schema`

## Story

As the **executor dispatching a translate the planner compiled**, I want **the planner to report a
translate as the moves the command names, in the command's own direction words, and the compiler to
turn them into the robot frame**, so that **no step asks a small language model to apply a frame
sign it does not apply reliably.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`planner-translate-sign-2026-10-11`](../../../measurements/planner-translate-sign-2026-10-11/README.md):
  the sweep, the arms, the held-out set and their harness; §5c says why the compiler guard was
  withdrawn.
- [`planner-translate-two-axis-sign`](../../completed/planner-translate-two-axis-sign.md): the
  prompt examples that shipped, and the guard that did not.

## Context

- **The schema is the defect.** The planner asks the model for `translation_xy`, a signed
  `[forward, left]` vector in metres (`planner/prompt_builder.py`:38 and :51). `intent_parser.py`
  parses it (:85-117) into `MissionIntent.translation_xy` (`schemas/mission.py`:33), and
  `plan_compiler._compile_translate` (:156-175) passes it into the one `translate` step that the
  executor runs (`executor/mission_runner.py`, `_translate`, :1883).
- **No prompt tried carries the sign.** On the record's 48 two-axis commands, Qwen3-4B at greedy
  decoding mis-signs 31 at `main`'s prompt, 24 with three two-axis examples, and 22 with a rule
  sentence on top. The prompt states the convention ("Right = negative left") in every arm. None of
  the examples is backward-left, and that class stays at 10 of 12 mis-signed under both prompt
  arms; a backward-left example was not tried.
- **A text guard is not the answer either.** A compiler guard that read the sign from the command's
  direction words fixed the measured commands. It was withdrawn before merge: a regex over the
  operator's phrasing is brittle, and opaque to an operator who does not know the protected
  phrasing (the record's §5c).
- **The proposal.** Replace the vector with ordered moves the model can copy from the command:
  `"moves": [{"direction": "<forward|backward|left|right>", "distance_m": <float>}, …]`, in command
  order.
  - The compiler maps each direction to an axis and a sign with one fixed table: forward +x,
    backward −x, left +y, right −y.
  - It composes the single `translate` step the executor already consumes, summing per axis. The
    executor's plan format does not change.
- **The CLI's phrasing is one among many.** The CLI sets' `format_command` builds "move X meters
  forward|backward and Y meters left|right" (`tools/cli_mission.py` in the evidence repository's
  `goal-a-cli-confirmation-2026-10-03` deposit). It stays as it is.
- **The acceptance set.** The record's 48-command sweep (`sweep_commands.json`) and 12 held-out
  commands (`heldout_commands.json`), with its harness (`translate_sign_sweep.py`,
  `planner_log_config.json`), all in the record's deposit.
  - *2026-10-14, a mis-signed command outside the sweep.* In the speed-cap A/B on the sim-bridge lane
    ([`lane-term-speedcap-ab-2026-10-14`](../../../measurements/lane-term-speedcap-ab-2026-10-14/README.md)
    §6), the shipped planner compiled `move 3.077 meters forward and 0.035 meters left` to
    dy −0.035, and the compile-first dry-check refused it.
    - The sweep measured this class, a two-axis command with a tiny lateral component, at 0.021 m.
      On the shipped tree the anchor in this same wording (`move 3.107 meters forward and 0.021
      meters left`) compiled exactly, and `drive 3.107 meters ahead then 0.021 meters left` was
      mis-signed (`runs/shipped_pass1.stdout` in the record's deposit).
    - It did not measure 0.035 m. An exact compile at one pair of numbers does not cover the template
      at another, so this command joins the acceptance set.
- **Today's state.** The tree carries the prompt examples, and the compiler passes the model's
  values through. On the record's commands that is 24 of 48 and 2 of 12 mis-signed, measured live
  on the shipped tree. The live fixture for backward-left is a strict `xfail` naming this brief
  (`source/strafer_autonomy/tests/test_planner_live_translate.py`).

## The dropped axis

Folded in from `planner-translate-dropped-axis` (filed 2026-10-12): one brief owns translate
correctness.

- **The four backward commands.** In the record's sweep the model returned `dx = 0` on all four
  "strafe S Y meters while moving backward X meters" commands. It did so in every arm: before the
  prompt change, with the two-axis examples, with the extra rule sentence, and with the guard.
  - The four:
    - "strafe right 3.533 meters while moving backward 0.032 meters";
    - "strafe right 0.9 meters while moving backward 0.6 meters";
    - "strafe left 3.533 meters while moving backward 0.032 meters";
    - "strafe left 0.9 meters while moving backward 0.6 meters".
  - The forward forms of the same wording keep `dx`.
- **H3.** In the held-out set, "head forward 2.25 m, then 0.35 m to the right" compiled to
  `[2.25, 0.0]`: the lateral component was dropped.
- **A wrong sign on top.** With the examples only, the two "strafe left … while moving backward …"
  commands also come back with the lateral sign wrong, `[0.0, -3.533]` and `[0.0, -0.9]`: a pure
  strafe to the right (the record's `runs/shipped_pass1`).
- **The counts.** 4 of the sweep's 48 and 1 of its 12 held-out commands. The rate over other
  wordings is not measured.
- **Expected to improve, not assumed to.** Two named moves are a more natural extraction than a
  filled vector. Measure it.

## Acceptance criteria

- [ ] **The change.** The schema, the parser, the compiler and the prompt change together.
      `translation_xy` leaves the model's output, kept at most as a compiler-internal value. The
      tests are mutation-proven: they fail on `main`.
- [ ] **The sweep.** The 48, the 12 and the 2026-10-14 tiny-lateral command run again on the live
      planner with the record's harness: 0 mis-signs on all three, and the dropped-axis count
      reported on the 48 and the 12.
- [ ] **The record.** `planner-translate-direction-schema-<date>` under `docs/measurements/`,
      deposited per [`docs/measurements/README.md`](../../../measurements/README.md), with two arms:
      `main` (the examples only) and the schema.
- [ ] **The dry-check.** The compile-first dry-check for real-robot submissions is revisited on that
      record. It stays unless the sweep reads 0 mis-signs and 0 dropped axes on both sets.
- [ ] **The marker.** The live module's `xfail` comes off (it turns red the day backward-left
      passes), and its cases pass against the branch's planner.
- [ ] **Regression.** The autonomy suite (`-m "not requires_ros"`) passes.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation steps (paired session)

1. **Schema and parser** (`intent_parser.py`, `schemas/mission.py`): parse and validate `moves`,
   reject an unknown direction or a negative distance.
2. **The compiler table** (`plan_compiler._compile_translate`): direction to axis and sign, summed
   per axis, one `translate` step.
3. **The prompt** (`prompt_builder.py`): the schema text and the translate examples, rewritten as
   moves.
4. **Tests**: parser, table and prompt examples, each shown to fail on `main`.
5. **Then the sweep, autonomous.** Serve the branch's planner with the record's log config and run
   its harness on the 48 and the 12. Pass the venv to `make` itself
   (`make serve-planner VENV_VLM=<main checkout>/.venv_vlm`): the Makefile assigns `VENV_VLM` with
   `:=`, so an environment variable does not reach it.

## Out of scope

- Rotation sign: [`planner-rotate-direction-prompt`](planner-rotate-direction-prompt.md).
- The executor's translate skill and the plan format it consumes.
- The CLI sets' `format_command`.
- Any intent other than `translate`.
