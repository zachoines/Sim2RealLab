# The planner's sign on a two-axis translate, 2026-10-11

On 2026-10-03 the planner compiled `move 1.712 meters forward and 1.372 meters right` to a move
1.372 m to the robot's left
([`goal-a-cli-confirmation-2026-10-03`](../goal-a-cli-confirmation-2026-10-03/README.md), R1).
This record measures how often the planner returns a wrong sign on a two-axis translate. It sends 52
commands to the live Qwen3-4B planner in the executor's request shape, before the change and under
three candidate changes. Each is scored against a rule committed before its first request.

**On these commands, the prompt change alone does not fix the sign. The prompt change plus a sign
guard in the plan compiler does.**
- **Before the change**, 31 of 48 balanced two-axis commands come back with a component of the wrong
  sign, and 12 are exact. R1 reproduces. Its messages, rebuilt locally, hash to the 2026-10-03
  value (`f809e9804bbf…`), and the model returns `[1.712, 1.372]`.
- **Three two-axis examples in the prompt** (arm A) leave 24 of 48 scored mis-sign. **A rule
  sentence on top of them** (arm B) leaves 22. Counting the dropped-axis plans that also carry a
  wrong sign, 26 and 23 (§3). Neither meets the rule.
- **Arm A's prompt plus a guard in `_compile_translate`** (arm C) gives 0 mis-signs of 48, 44 exact,
  and all five anchors exact. It meets the rule as registered, and it is what the branch ships. The
  guard gives each non-zero component the sign that the command's own words name, when they name
  exactly one direction on that axis. On these commands, and on the held-out 12, that zero follows
  from the guard's construction (§5).
- **The 4 inexact commands under arm C are not sign errors.** "strafe S Y meters while moving
  backward X meters" comes back with the backward component at 0 (`dropped_axis`), in every arm.
- **Held out.** On 12 commands written after arms A and B, arm C compiles 0 mis-signs and 11 exact.
  The prompt alone mis-signed 2 of them, both backward and left, and the guard corrected both. The
  twelfth drops an axis: "head forward 2.25 m, then 0.35 m to the right" gives `[2.25, 0.0]`.
- Every arm's two passes agree request by request, 52 of 52.

**What follows.**
- The shipped fix is the prompt examples plus the compiler guard.
  [`planner-translate-two-axis-sign`](../../tasks/completed/planner-translate-two-axis-sign.md)
  expected the prompt change alone, because the compiler passed the sign through. Arm A measured
  that change at 24 of 48.
- The operating rule is that every real-robot submission is compiled through the planner first and
  dispatched only if the plan matches. This record does not decide whether to keep it. It reports
  what that dry-check would still catch on these commands: a dropped axis, 4 of 48 and 1 of 12.

## 1. Setup

**The planner.**

| | |
|---|---|
| model | `Qwen/Qwen3-4B` in HF transformers, in process: `device_map=auto, dtype=auto, 4bit=False` (the load line of each serve log) |
| decoding | greedy: `do_sample=False`, thinking off. Transformers warns that `temperature`, `top_p` and `top_k` "are not valid and may be ignored" (`runs/pre_serve_planner.log`, line 16) |
| service | `make serve-planner`, port 8200, `.venv_vlm`, one launch per arm |
| logging | `UVICORN_LOG_CONFIG=planner_log_config.json`, which uvicorn reads as `--log-config`. It routes the `strafer_autonomy` logger to stderr at INFO, so each request's raw `LLM output` line reaches the service log. The Makefile and the tree are unchanged for it |
| host | the DGX, one planner service at a time (`provenance.md`) |

**Requests.** Each command is planned the way the executor plans it, in the shape the CLI set's
`plan_check.py` reproduces (`translate_sign_sweep.py`):
- `HttpPlannerClient` with its default config;
- the command verbatim, `robot_state` None, the executor's fresh-mission summary and
  `DEFAULT_AVAILABLE_SKILLS`.

The tool rebuilds the model's messages locally and records their sha256. It also records the path
and the `SYSTEM_PROMPT` sha256 of the `prompt_builder` it imported, under the service's
`PYTHONPATH`. All 436 `/plan` requests across the five service launches returned 200 (the serve
logs).

**Commands** (`sweep_commands.json`, 52).
- 48 balanced commands in four classes: forward-right (FR), forward-left (FL), backward-right (BR)
  and backward-left (BL). Each class has six wordings × two magnitude pairs.
  - The classes share magnitudes and differ only in their direction words.
  - Each wording has one 3-decimal pair and one short pair.
- 4 anchors: the CLI set's four forward-left commands (FX, G1, L1, L2), which compiled exactly on
  2026-10-03.
- R1 is FR/W1's first command. "The five anchors" are R1 and those four.
- None of the 52 magnitudes appears in the examples the change adds (1.5/0.5 and 0.8/0.3).

| wording | forward form, X forward and Y to side S | pairs (X, Y) |
|---|---|---|
| W1 | "move X meters forward and Y meters S" (R1's template) | (1.712, 1.372), (2.0, 1.0) |
| W2 | "go forward X m and Y m to the S" | (0.845, 2.316), (1.0, 2.0) |
| W3 | "drive X meters ahead then Y meters S" | (3.107, 0.021), (1.2, 0.4) |
| W4 | "strafe S Y meters while moving forward X meters" | (0.032, 3.533), (0.6, 0.9) |
| W5 | "move Y meters to the S and X meters forward" | (2.468, 0.913), (3.0, 0.25) |
| W6 | "translate X m forward, Y m S" | (1.354, 1.529), (0.7, 0.7) |

The backward forms say "backward" in place of "forward", and W3 says "back" in place of "ahead".

| anchor | command |
|---|---|
| R1 | "move 1.712 meters forward and 1.372 meters right" |
| FX | "move 3.074 meters forward and 0.191 meters left" |
| G1 | "move 3.107 meters forward and 0.021 meters left" |
| L1 | "move 1.354 meters forward and 1.529 meters left" |
| L2 | "move 0.032 meters forward and 3.533 meters left" |

**Outcome per command.** The first that applies, at a tolerance of 0.001 m (`classify` in
`translate_sign_sweep.py`):
1. `error`: the call raised.
2. `wrong_intent`: not exactly one step, or the mission or its step is not `translate`.
3. `axis_swap`: |dx| and |dy| match |Y| and |X|, with |X| ≠ |Y|.
4. `dropped_axis`: a component below 0.0005 m where a non-zero one was expected.
5. `mis_sign`: a component of the wrong sign, as `dy_flip`, `dx_flip` or `both_flip`.
6. `magnitude`: the signs are right, but a component is off by more than the tolerance.
7. `exact`: both components within the tolerance.

"Mis-sign of 48" counts the balanced commands scored `mis_sign`. A command scored `dropped_axis` or
`wrong_intent` is not counted there, even when its other component also has the wrong sign.

**What was written down first.**
- The question, the commands, the scoring tool, the log config and the acceptance rule were committed
  to the evidence repository at `d2fb7cc` (2026-10-11 18:03:55 CDT). The first scored pass started
  at 18:05:13 CDT (23:05:13Z, `started_utc` in `runs/pre_pass1/summary.json`).
- **The rule.** The change is accepted if, after it, there are 0 mis-signs of 48, the five anchors
  are all exact, and no class's exact count is lower than before. If the prompt change was not
  accepted, the translate rule was to gain one sentence and the sweep was to be repeated (arm B),
  with both results reported.
- Arms A and B both failed the rule. An addendum then registered arm C under the same rule, with a
  consistency check and a held-out set. It was committed with the before, A and B runs at `c9745b4`
  (18:28:26). Arm C's service started at 18:28:38 (`runs/postC_serve_planner.log`, line 4), and
  its first pass at 18:29:36 CDT (23:29:36Z, `started_utc` in `runs/postC_pass1/summary.json`).

## 2. The arms

| arm | change against `9f7d207` | file | `SYSTEM_PROMPT` sha256 | served from |
|---|---|---|---|---|
| before | none | | `abef3f984f4a…` | the main checkout at `9f7d207` |
| A | three two-axis examples: forward+right `[1.5, -0.5]`, forward+left `[1.5, 0.5]`, backward+right `[-0.8, -0.3]` | `armA_prompt.patch` | `5f8d805d3ad0…` | a worktree of `task/planner-translate-two-axis-sign` |
| B | A, plus "Each axis keeps its own sign: a component to the right is negative even after a positive forward component." in the translate rule | `armB_prompt.diff`; the served file is `runs/postB_prompt_builder.py` | `f83cc796294a…` | the same worktree |
| C | A's prompt, plus the guard in `plan_compiler._compile_translate` | `armC.patch` | `5f8d805d3ad0…` | the same worktree |

- Before the change, `prompt_builder.py` is blob `ae5b1482`, the same blob as at `ed0d5af`, the
  brief's baseline.
- `armC.patch` applied to `9f7d207` gives the branch's `prompt_builder.py`, `plan_compiler.py` and
  `test_plan_compiler.py` at `1031643` byte for byte. The branch's other changes are its other tests
  and the `live_planner` marker.
- The sha256 column is the `prompt_builder` that the sweep tool imported, from each run's
  `summary.json`. Two things in the deposit bear on what the service itself served:
  - The serve logs start `.venv_vlm/bin/uvicorn` by a relative path for the before arm, which only
    the main checkout has. For A, B and C they start it by the main checkout's absolute path, as
    `make` does in the worktree with `VENV_VLM` set.
  - In arm C, the service's raw outputs and its compiled plans differ only in sign (§5).

## 3. Results

From `sweep_tables.txt`. Every arm's two passes are identical request by request, so pass 1 stands
for both.

| arm | mis-sign of 48 | exact of 48 | anchors exact of 5 | mis-sign sub-types | passes identical | R1 plan | R1 message sha256 |
|---|---:|---:|---:|---|---|---|---|
| before | 31 | 12 | 4 | {'dy_flip': 27, 'dx_flip': 2, 'both_flip': 2} | 52/52 | [1.712, 1.372] | f809e9804bbf… (the deposit's) |
| A | 24 | 20 | 5 | {'dy_flip': 20, 'dx_flip': 1, 'both_flip': 3} | 52/52 | [1.712, -1.372] | 9c6cc2296821… |
| B | 22 | 22 | 3 | {'dy_flip': 18, 'dx_flip': 1, 'both_flip': 3} | 52/52 | [1.712, -1.372] | e68f14de5385… |
| C | 0 | 44 | 5 | {} | 52/52 | [1.712, -1.372] | 9c6cc2296821… |

"The deposit's" is `llm_messages_sha256_local_prompt_builder` in
`goal-a-cli-confirmation-2026-10-03`'s `logs/runs/R1.plancheck_failed`. Both values hash messages
rebuilt locally from the `prompt_builder` each tool imported, not the service's own input. Under the
before arm, FX, G1, L1 and L2 also hash to that deposit's values, in its
`logs/runs/<anchor>.plancheck.out` (`runs/pre_pass1/results.jsonl`). Arms A and C share a message
sha256 because they share a prompt.

**Against the rule.**
- **A: not accepted.** 24 mis-signs, and forward-left's exact count falls from 9 to 8.
- **B: not accepted.** 22 mis-signs, two forward-left anchors flip (FX and G1), so 3 of 5, and
  forward-left's exact count falls from 9 to 8.
- **C: accepted.** 0 mis-signs, 5 of 5 anchors, and every class's exact count at or above before
  (FR 12 against 2, FL 12 against 9, BR 10 against 1, BL 10 against 0).

**Per class** (pass 1, tolerance 0.001 m):

| arm | class | n | error | wrong_intent | axis_swap | dropped_axis | mis_sign | magnitude | exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| before | FR | 12 | 0 | 1 | 0 | 0 | 9 | 0 | 2 |
| before | FL | 12 | 0 | 0 | 0 | 0 | 3 | 0 | 9 |
| before | BR | 12 | 0 | 0 | 0 | 2 | 9 | 0 | 1 |
| before | BL | 12 | 0 | 0 | 0 | 2 | 10 | 0 | 0 |
| before | anchor_FL | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| A | FR | 12 | 0 | 0 | 0 | 0 | 7 | 0 | 5 |
| A | FL | 12 | 0 | 0 | 0 | 0 | 4 | 0 | 8 |
| A | BR | 12 | 0 | 0 | 0 | 2 | 3 | 0 | 7 |
| A | BL | 12 | 0 | 0 | 0 | 2 | 10 | 0 | 0 |
| A | anchor_FL | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| B | FR | 12 | 0 | 0 | 0 | 0 | 6 | 0 | 6 |
| B | FL | 12 | 0 | 0 | 0 | 0 | 4 | 0 | 8 |
| B | BR | 12 | 0 | 0 | 0 | 2 | 2 | 0 | 8 |
| B | BL | 12 | 0 | 0 | 0 | 2 | 10 | 0 | 0 |
| B | anchor_FL | 4 | 0 | 0 | 0 | 0 | 2 | 0 | 2 |
| C | FR | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |
| C | FL | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |
| C | BR | 12 | 0 | 0 | 0 | 2 | 0 | 0 | 10 |
| C | BL | 12 | 0 | 0 | 0 | 2 | 0 | 0 | 10 |
| C | anchor_FL | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |

- Backward-left stays at 10 of 12 mis-signed under both prompt arms. The three examples include no
  backward-left case (an observation; a backward-left example was not tried).
- The `dropped_axis` rows are the same four commands in every arm: W4's backward forms, BR and BL
  (`runs/<arm>_pass1/results.jsonl`).
- Under A and B, BL/W4's `dropped_axis` plans also carry a wrong lateral sign (A both, B one). So 26
  (A) and 23 (B) of 48 have some wrong-signed component. Before and C have none hidden this way
  (`runs/<arm>_pass1/results.jsonl`).

## 4. Before the change, by wording

The balanced 48, pass 1:

| wording | error | wrong_intent | axis_swap | dropped_axis | mis_sign | magnitude | exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1 | 0 | 0 | 0 | 0 | 6 | 0 | 2 |
| W2 | 0 | 0 | 0 | 0 | 6 | 0 | 2 |
| W3 | 0 | 1 | 0 | 0 | 5 | 0 | 2 |
| W4 | 0 | 0 | 0 | 4 | 2 | 0 | 2 |
| W5 | 0 | 0 | 0 | 0 | 7 | 0 | 1 |
| W6 | 0 | 0 | 0 | 0 | 5 | 0 | 3 |

These rows are read from `runs/pre_pass1/results.jsonl`:
- **Every wording mis-signs.** W1, R1's own template, gets 2 of 8 right: both forward-left ones.
- **The lateral sign is most of it.** 29 of the 31 mis-signs flip the lateral component: 27 flip it
  alone (`dy_flip`), and two flip both (`both_flip`, BL's W5 forms). The other two flip only the
  backward component (`dx_flip`, BR's W5 forms). In all four W5 backward commands, "move Y meters to
  the S and X meters backward", the backward component comes back positive.
- **Forward-left is the least affected class**, 3 of 12. Its two W4 commands ("strafe left …") come
  back with a negative lateral component, and so does W5's short pair.
- **The `wrong_intent`** is FR/W3's "drive 1.2 meters ahead then 0.4 meters right". It came back as
  `go_to_targets`. In arms A, B and C the same command is a `translate`.
- **The four `dropped_axis`** are "strafe S Y meters while moving backward X meters", with the
  backward component returned as 0.

## 5. The guard (arm C)

**What it does** (`armC.patch`; `plan_compiler.py` at `1031643`).
- It lower-cases the command and splits it into words.
- For x it reads "forward", "forwards" and "ahead" (positive) against "backward", "backwards" and
  "back" (negative). For y it reads "left" (positive) against "right" (negative).
- When the words name exactly one of an axis's two directions, a non-zero component takes that
  sign. Otherwise the LLM's value passes through, and a zero stays zero.
- It changes signs only, never a magnitude or an axis, and only in `translate` intents.

**Consistency.** Arm C's prompt is arm A's. The arm C service log holds 120 `LLM output` lines: the
two passes of 52, the held-out 12 and the 4 live fixtures.
- Arm C's raw output equals arm A's compiled plan on 52 of 52 commands, across two service launches.
- Arm C's raw output is identical across its own two passes, 52 of 52.

**Effect.**
- The guard changed 26 of the 52 compiled values, every one by sign alone.
- Scored without the guard, the raw outputs mis-sign 24 of 48, which is arm A's count. Compiled, they
  mis-sign 0 of 48.
- The 26 changes are the 24 mis-signs and the two BL/W4 commands. Those two are `dropped_axis` with
  or without the guard. Their raw lateral component was also negative, for "left", and the guard
  corrected it.
- **The zero is by construction.** Every command in the sweep and the held-out set names each
  non-zero axis with exactly one of the guard's words (`sweep_commands.json`,
  `heldout_commands.json`). On these commands the guard rules out a compiled mis-sign: the 0 of 48
  and 0 of 12 follow from the code, not the model. What arm C's runs measure is the rest: 44 of 48
  exact, 5 of 5 anchors, the dropped axis, and the raw outputs (24 of 48 and 2 of 12 mis-signed).

| class | wording | raw LLM | compiled | command |
|---|---|---|---|---|
| FR | W2 | [0.845, 2.316] | [0.845, -2.316] | go forward 0.845 m and 2.316 m to the right |
| FR | W2 | [1.0, 2.0] | [1.0, -2.0] | go forward 1.0 m and 2.0 m to the right |
| FR | W3 | [3.107, 0.021] | [3.107, -0.021] | drive 3.107 meters ahead then 0.021 meters right |
| FR | W3 | [1.2, 0.4] | [1.2, -0.4] | drive 1.2 meters ahead then 0.4 meters right |
| FR | W5 | [2.468, 0.913] | [2.468, -0.913] | move 0.913 meters to the right and 2.468 meters forward |
| FR | W5 | [3.0, 0.25] | [3.0, -0.25] | move 0.25 meters to the right and 3.0 meters forward |
| FR | W6 | [1.354, 1.529] | [1.354, -1.529] | translate 1.354 m forward, 1.529 m right |
| FL | W3 | [3.107, -0.021] | [3.107, 0.021] | drive 3.107 meters ahead then 0.021 meters left |
| FL | W4 | [0.032, -3.533] | [0.032, 3.533] | strafe left 3.533 meters while moving forward 0.032 meters |
| FL | W4 | [0.6, -0.9] | [0.6, 0.9] | strafe left 0.9 meters while moving forward 0.6 meters |
| FL | W6 | [0.7, -0.7] | [0.7, 0.7] | translate 0.7 m forward, 0.7 m left |
| BR | W2 | [-1.0, 2.0] | [-1.0, -2.0] | go backward 1.0 m and 2.0 m to the right |
| BR | W5 | [2.468, -0.913] | [-2.468, -0.913] | move 0.913 meters to the right and 2.468 meters backward |
| BR | W5 | [3.0, 0.25] | [-3.0, -0.25] | move 0.25 meters to the right and 3.0 meters backward |
| BL | W1 | [-1.712, -1.372] | [-1.712, 1.372] | move 1.712 meters backward and 1.372 meters left |
| BL | W1 | [-2.0, -1.0] | [-2.0, 1.0] | move 2.0 meters backward and 1.0 meters left |
| BL | W2 | [-0.845, -2.316] | [-0.845, 2.316] | go backward 0.845 m and 2.316 m to the left |
| BL | W2 | [-1.0, -2.0] | [-1.0, 2.0] | go backward 1.0 m and 2.0 m to the left |
| BL | W3 | [-3.107, -0.021] | [-3.107, 0.021] | drive 3.107 meters back then 0.021 meters left |
| BL | W3 | [-1.2, -0.4] | [-1.2, 0.4] | drive 1.2 meters back then 0.4 meters left |
| BL | W4 | [0.0, -3.533] | [0.0, 3.533] | strafe left 3.533 meters while moving backward 0.032 meters |
| BL | W4 | [0.0, -0.9] | [0.0, 0.9] | strafe left 0.9 meters while moving backward 0.6 meters |
| BL | W5 | [2.468, -0.913] | [-2.468, 0.913] | move 0.913 meters to the left and 2.468 meters backward |
| BL | W5 | [3.0, -0.25] | [-3.0, 0.25] | move 0.25 meters to the left and 3.0 meters backward |
| BL | W6 | [-1.354, -1.529] | [-1.354, 1.529] | translate 1.354 m backward, 1.529 m left |
| BL | W6 | [-0.7, -0.7] | [-0.7, 0.7] | translate 0.7 m backward, 0.7 m left |

## 6. Held out

`heldout_commands.json` holds 12 commands, written after arms A and B. They are not part of the rule.
They ran once, on arm C:
- 8 two-axis commands in wordings the sweep does not use;
- 3 single-axis commands;
- one command where "right" is not a direction ("move forward 1.3 meters right away").

Raw LLM mis-signs: 2 of 12. Compiled mis-signs: 0 of 12. Compiled exact: 11 of 12.

| wording | expected | raw LLM | compiled | outcome | command |
|---|---|---|---|---|---|
| H1 | [1.1, -0.6] | [1.1, -0.6] | [1.1, -0.6] | exact | slide 0.6 meters right and go 1.1 meters forward |
| H2 | [-0.75, 0.4] | [-0.75, -0.4] | [-0.75, 0.4] | exact | back up 0.75 meters and shift 0.4 meters left |
| H3 | [2.25, -0.35] | [2.25, 0.0] | [2.25, 0.0] | dropped_axis | head forward 2.25 m, then 0.35 m to the right |
| H4 | [1.8, 0.65] | [1.8, 0.65] | [1.8, 0.65] | exact | go 1.8 meters ahead and 0.65 meters to your left |
| H5 | [-0.9, -1.15] | [-0.9, -1.15] | [-0.9, -1.15] | exact | move backwards 0.9 m and 1.15 m right |
| H6 | [-0.5, -0.5] | [-0.5, -0.5] | [-0.5, -0.5] | exact | step back 0.5 meters and 0.5 meters to the right |
| H7 | [2.6, -0.3] | [2.6, -0.3] | [2.6, -0.3] | exact | move 2.6 meters forward while drifting 0.3 meters right |
| H8 | [-1.25, 0.85] | [-1.25, -0.85] | [-1.25, 0.85] | exact | go 1.25 meters backward and 0.85 meters to the left |
| H9 | [0.0, 1.4] | [0.0, 1.4] | [0.0, 1.4] | exact | move left 1.4 meters |
| H10 | [-0.6, 0.0] | [-0.6, 0.0] | [-0.6, 0.0] | exact | go back 0.6 meters |
| H11 | [0.0, -0.75] | [0.0, -0.75] | [0.0, -0.75] | exact | strafe right 0.75 meters |
| H12 | [1.3, 0.0] | [1.3, 0.0] | [1.3, 0.0] | exact | move forward 1.3 meters right away |

- **H2 and H8** are the raw mis-signs. Both are backward and left, and the guard corrects both.
- **H3** is the residual. The model drops the 0.35 m to the right, and the guard cannot restore a
  zero.
- **H12** compiles exactly because the model returned a lateral 0, which the guard leaves alone.

## 7. Tests

**The autonomy suite**, `-m "not requires_ros"`, with
`source/strafer_autonomy/tests/test_sim_in_the_loop_runtime_env.py` left out:

| tree | tests | passed | skipped | failures | errors | file |
|---|---:|---:|---:|---:|---:|---|
| `main` at `9f7d207` | 692 | 692 | 0 | 0 | 0 | `tests/autonomy_not_ros_main.xml` |
| the branch, arm C's source | 705 | 701 | 4 | 0 | 0 | `tests/autonomy_not_ros_armC.xml` |

- **13 new tests.** Two are in `test_prompt_builder.py` (`TestTranslateExamples`) and seven in
  `test_plan_compiler.py`: the guard's direction-word cases and its pass-through cases. The other
  four are the live fixtures, which skip without `STRAFER_LIVE_PLANNER_URL`.
- **The 705 test ids** in the branch's file are the ones the branch collects at `1031643`
  (`tests/branch_collection_ids.txt`).
- **The module left out** fails to collect in `.venv_vlm`, on `main` and on the branch alike:
  `ModuleNotFoundError: No module named 'isaaclab_tasks'`, reached through `strafer_lab`. It is
  environmental, predates this change, and is not touched. Its collection output on both is
  `tests/collection_error.txt`.

**The live fixtures**, `source/strafer_autonomy/tests/test_planner_live_translate.py`, with
`STRAFER_LIVE_PLANNER_URL=http://127.0.0.1:8200`. Each case is 1.712 m and 1.372 m, asserted to
1e-3 per component:

| planner served from | forward-right (R1) | forward-left | backward-right | backward-left | file |
|---|---|---|---|---|---|
| the branch (arm C's service) | pass | pass | pass | pass | `tests/live_translate_armC.xml` |
| `main` at `9f7d207` | fail, (1.712, 1.372) | pass | fail, (-1.712, 1.372) | fail, (-1.712, -1.372) | `tests/live_translate_main.xml` |

The `main` row ran the branch's test module against a service started from the main checkout
(`runs/main_fixture_serve_planner.log`).

**The prompt tests, mutated.** `test_prompt_builder.py` fails one test in each of two cases:
- on the branch's prompt with the forward-right example's sign flipped, it fails
  `test_signs_follow_each_example_command`;
- on `main`'s prompt, which has no two-axis example, it fails
  `test_two_axis_examples_cover_both_sides_and_backward`.

Both runs are in `tests/mutation_runs.txt`, re-run on the branch's test module after the sweeps.

## What this record does not show

- **Other models or settings.** It covers one model, greedy decoding, on one GPU. No other model,
  decoding or quantization was run.
- **A rate for commands in general.** The 52 and 12 commands come from hand-written templates. The
  counts hold for these wordings only.
- **Robot motion.** Nothing was dispatched. The record scores plans, and the executor is unchanged.
- **That every sign is now right.** The guard reads words, not meaning. These are properties of the
  code, not measured cases:
  - a command that names both directions on an axis ("move 1 meter left, then 1 meter right")
    passes the model's sign through;
  - a direction word outside the guard's lists passes the model's sign through;
  - a "right" or "back" that is not a direction still sets the sign of a non-zero component. H12
    compiled exactly only because the model returned a lateral 0.
- **A fix for the dropped axis.** "strafe S Y meters while moving backward X meters" drops the
  backward component in every arm, 4 of 48. H3 drops the lateral one. The guard leaves a zero at
  zero.
- **Any other intent.** Rotation sign is
  [`planner-rotate-direction-prompt`](../../tasks/active/reliability/planner-rotate-direction-prompt.md).
  No other intent was sent.
- **Whether the dry-check rule can be retired.** The residual above is what it would still catch.

## Deviations from the protocol as first registered

- **Arm C was added after arms A and B.** The first registration named the prompt change and one
  fallback sentence, and both failed the rule. The addendum that registered arm C, its consistency
  check and the held-out set was committed before arm C's service started. The rule did not change.
- **The guard was written after arms A and B's results** on the same 48 commands. The held-out 12
  are the commands it was not written against.
- **The served code's check.** The pre-registration names two checks of the served code: a read of
  the uvicorn process's environment, and the sweep tool's own record of its import. The import
  records are deposited (`summary.json`). The environment was read from `/proc` before each
  service's sweeps — `PYTHONPATH` set to the worktree's `source/strafer_autonomy` for arms A, B and
  C, unset for the before arm and the `main` fixture run — but the reads were not saved, so no file
  here carries them.
- **By wording after the change.** The pre-registration lists every outcome by class and wording.
  By-wording counts are tabled for the before arm only (§4). Arms A, B and C have them per command
  in `runs/<arm>_pass1/results.jsonl`, and arm C's four non-exact commands are all W4 (§3).
- **The dtype.** The pre-registration says bf16. The serve logs show only `dtype=auto`, and no
  deposited file records the dtype it resolved to.

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `planner-translate-sign-2026-10-11/record-files/` |
| deposit commit | `43d7178cca26bb92bac78c0a9e72225f9e3eec33` |

The deposit was built in three commits:
- **`d2fb7cc7ec090277d5ef22354d73e2c23469efcc`**, before the first request: the pre-registration,
  the commands, the sweep tool and the log config.
- **`c9745b49304356e975cfdd565a27835c3043fd7d`**, before arm C's service started: the addendum, the
  arm patches, the held-out commands, and the before, A and B runs.
- **The deposit commit above**, which adds the rest and replaces `runs/postB_serve_planner.log`
  with its complete copy (four shutdown lines added, 18:28:37).

The deposit holds:
- the inputs:
  - `PREREGISTRATION.md`, the registration (committed at `d2fb7cc`, before the first request) and
    its addendum (committed at `c9745b4`, before arm C's service started);
  - `sweep_commands.json` and `heldout_commands.json`;
  - `translate_sign_sweep.py`, the request and scoring tool;
  - `planner_log_config.json`, the service's log config;
- the arms: `armA_prompt.patch`, `armB_prompt.diff` with `runs/postB_prompt_builder.py`, and
  `armC.patch`;
- the runs:
  - `runs/<arm>_pass1/` and `runs/<arm>_pass2/` for `pre`, `postA`, `postB` and `postC`, each with
    `results.jsonl` and `summary.json`, and its console in `runs/<arm>_pass<n>.stdout`;
  - `runs/postC_heldout/`, with its console in `runs/postC_heldout.stdout`;
  - each service's log, with every request's raw `LLM output`: `runs/<arm>_serve_planner.log` and
    `runs/main_fixture_serve_planner.log`;
- the test runs, `tests/*.xml`;
- `sweep_tables.py` with `sweep_tables.txt`, the tables in §3–§6.

Each command block below runs from the Sim2RealLab root, with the evidence repository cloned there.

Restore into this record's directory with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/planner-translate-sign-2026-10-11/record-files/. \
      docs/measurements/planner-translate-sign-2026-10-11/
```

Re-derive the tables with:

```
cd docs/measurements/planner-translate-sign-2026-10-11
PYTHONDONTWRITEBYTECODE=1 ../../../.venv_vlm/bin/python sweep_tables.py | diff - sweep_tables.txt
```

Verify digests with:

```
cd Sim2RealLab-Artifacts/planner-translate-sign-2026-10-11/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

sha256 of every file in the deposit:

```
a13c65e395256f0ed5d4ef08b96aa61e6e93788502855cee8e2143fe5df5f550  PREREGISTRATION.md
f4f4fcede13c4f36abf40574c90d151e09fd6f82435d7690f5a02925cc1bf740  armA_prompt.patch
4c045987498a97327ba72a48ae3a4f27936151b2de75a28c214a3b7f9828b950  armB_prompt.diff
3b377bd2c8fd9b4495065ae96c6306e3c6add2a6c15116868bf63c67818261b0  armC.patch
7cc1b8fc517ec12078cf8a024327892f326e8a58bc5e0066b4f77bb0a90eb6cb  heldout_commands.json
15db559cd09153a5aa9d61d64f18711e5721404a3728f5365c33700338dd8260  planner_log_config.json
1fdfc04d959e026213dd009846a7c7a5d46691ba35f4ad1138d0f16d7759e516  runs/main_fixture_serve_planner.log
b7534c2c456e0d2bfc054d76ca83c6f871ee8377a555f5525e09cf9bd9fe74e2  runs/postA_pass1.stdout
9f81b69a53412b18a3f8d5a1fcdabc84526d265731632946f768fb7d0c9373ee  runs/postA_pass1/results.jsonl
674b2ac3c368296cf14372190f287727b881fe5001eff147ba96b9b768352947  runs/postA_pass1/summary.json
b7534c2c456e0d2bfc054d76ca83c6f871ee8377a555f5525e09cf9bd9fe74e2  runs/postA_pass2.stdout
01f2c8947610f781d2d6a2aff868571fff3b37c5935c64f2eef5aa0769e71d13  runs/postA_pass2/results.jsonl
6b859cca2b5d35142058fcdc23026ad4f6463ee5641f0b23fbf508732d1adf23  runs/postA_pass2/summary.json
0915b9cabba82f609cfbca196078a475bdb9f415d50c790a165c1a6866d544d9  runs/postA_serve_planner.log
0a25b44ff88f56be4534b42ab36a11938ad1d5fe6792b8e9b3d5b1e4693af6cf  runs/postB_pass1.stdout
747e2bae113506bb25a2dbfaa8e6c65c395af2c292853b4c797606cd9a61159f  runs/postB_pass1/results.jsonl
442bba55dc9f8cd1c5070ad4d79c023e0aeb2f7ea2ad95079dcab5d34948857d  runs/postB_pass1/summary.json
0a25b44ff88f56be4534b42ab36a11938ad1d5fe6792b8e9b3d5b1e4693af6cf  runs/postB_pass2.stdout
b6987776b64f4d46215f3f2237d7278cfec214c30ee5778873210fb56b4d9bb8  runs/postB_pass2/results.jsonl
6492d6eebc7e8718e5568f0266a5d1a6b7236a02b0c12b100a37b57f861be18e  runs/postB_pass2/summary.json
21e12f23ae333ec50e814770181f87c0788aa5b74fffe6936a653f08e45aada1  runs/postB_prompt_builder.py
8a09947b4f1eadfb04816f5bc562c1125c35450483b0e9db56e25a8e6bc73049  runs/postB_serve_planner.log
601bd45fa3f7537423d8dd09f5d34a576713a5522fa1f8dbaa0a1484558908ad  runs/postC_heldout.stdout
87aecad60a3a300e85e86b5f45ebd0c3534c80867c9eadaf3fa79cfbfff7769b  runs/postC_heldout/results.jsonl
18d68589fa70def380b5a6e461536106c23b0b83bbe8d00250d76aa80f9fa7a3  runs/postC_heldout/summary.json
c4ccee6ae3a05f31aa0909b28606c38dcfcc5255a9c0cfc56b8e4430d261c4d3  runs/postC_pass1.stdout
21871d797bfa356d551620e45440a4395e3f0331b9bc23972cbf0638414219a5  runs/postC_pass1/results.jsonl
7045b8794f0a6eef907dbbb48f54d23c59387b2b74c72a5b12a3b07316f21999  runs/postC_pass1/summary.json
c4ccee6ae3a05f31aa0909b28606c38dcfcc5255a9c0cfc56b8e4430d261c4d3  runs/postC_pass2.stdout
396061d5c37bf58f20dc7ebc05f89a9653c87a0b82f5091f3955970cd13477ae  runs/postC_pass2/results.jsonl
6cb0f5c1bd3cd6fb60e70d849b2f58706d361cc47397f779e53db3af33836310  runs/postC_pass2/summary.json
9c4bce7ec40d3d34a37b8c6520dddb5346b67c0762f077b5c88cd342a74e6473  runs/postC_serve_planner.log
5f0cc5626d958a8382e76e5f3b92d69a208c337cc10203c6b4069b2eb6165497  runs/pre_pass1.stdout
bf0b9e33cf62283d9e7897c6f4c17b9ffb90ed5bf1680b299fc94f03e4cb6437  runs/pre_pass1/results.jsonl
3560f2c64cdc93af311fb4ece86a55b7f9d056942a7904ff514740471f177e01  runs/pre_pass1/summary.json
5f0cc5626d958a8382e76e5f3b92d69a208c337cc10203c6b4069b2eb6165497  runs/pre_pass2.stdout
c3338dbcb90fdb3183c2bebbc53d7cc4133134df49bd4ae9b9fed041ba461cfb  runs/pre_pass2/results.jsonl
6b4ddb5f9752ee00a604ea158fe3d8ffa87bd137e8a5302dae08380f04f68c68  runs/pre_pass2/summary.json
1fc100f41b1f3d5228633a81d086ea90952a3c382b5520b7c8011dd79becd9c0  runs/pre_serve_planner.log
64a59789a83d279a0c69e8c3ec04c43b98f507814e4c347eb5c57224c88f64d6  sweep_commands.json
18592d66705a4a744cac2d053911fecdcf9c08703da6e4a23cd6e53425595d79  sweep_tables.py
5574e494cd0f5b04c852a27c921cc3988c35ad5b987e1dda559f7fc6e7429b45  sweep_tables.txt
e5dbae221035d51187ba29f116b5d108c39620ac563fa5c1ff9e37350a067b5c  tests/autonomy_not_ros_armC.xml
99c2da8f565ccd3130dbba5603c5d814fe0ef94c5808b326b77f3b156a6a992b  tests/autonomy_not_ros_main.xml
3adfdeb7389061107ddf01e616dbbfe8fa7f4b5ec3ef4fd8cf0ca06c2fdf3533  tests/branch_collection_ids.txt
ea27e7e6e50eed8bb97824daa20af726bcc8a9d461e51da430a683566374fd4a  tests/collection_error.txt
f24612520735ec2b87060ca6aba913c557042a48fb05c3a2b179cb2e9c7ba937  tests/live_translate_armC.xml
8574a3ec5a8b7b1c359cef8e85ee6bdcf6e69886da584c4be9b58ab5369bfb3d  tests/live_translate_main.xml
d7a44097d6d301850ca7978024d954d066c195ccc638fc6b5898fc20b84fe8fb  tests/mutation_runs.txt
c7ec1b20aa1f4230b5be6a05e560e440fcadaa7ba207c6ae85361d996050cff5  translate_sign_sweep.py
```
