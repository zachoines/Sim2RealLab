# Provenance — the planner's sign on a two-axis translate, 2026-10-11

captured_date: 2026-10-11

## Host

| role | host | kernel | driver |
|---|---|---|---|
| planner service, sweeps and tests | gx10-d1d8 (ASUS GX10, a GB10 DGX Spark system, aarch64) | 7.0.0-1019-nvidia | 580.173.02 |

The four JUnit files carry `hostname="gx10-d1d8"`.

## Tree and stack

| runs | Sim2RealLab tree |
|---|---|
| before the change; the `main` suite; the `main` live-fixture service (the test module is the branch's) | the main checkout at `9f7d207` (the #238 merge), with `source/strafer_autonomy` unmodified |
| arms A, B and C; the branch suite and live-fixture runs | a worktree of `task/planner-translate-two-axis-sign` off `9f7d207`, each arm's change applied in the working tree (`armA_prompt.patch`, `armB_prompt.diff`, `armC.patch`) |

- **The branch's commits came after the runs.** `38ce883`, `e568e4a` and `1031643` were all made at
  18:41:26 CDT. `armC.patch` applied to `9f7d207` reproduces their `prompt_builder.py`,
  `plan_compiler.py` and `test_plan_compiler.py` byte for byte.
- **The worktree ran with `PYTHONPATH`** naming its own `source/strafer_autonomy`, because
  `.venv_vlm`'s editable install points `strafer_autonomy` at the main checkout. The sweep tool ran
  under the same `PYTHONPATH`, and each run's `summary.json` records the `prompt_builder` it
  imported. That is the worktree's for A, B and C, and the main checkout's for the before arm.

| | |
|---|---|
| interpreter | `.venv_vlm`: Python 3.12.3, torch 2.11.0+cu128, transformers 5.5.3, fastapi 0.135.3, uvicorn 0.44.0, pydantic 2.12.5 |
| NVRTC | `/usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvrtc.so.13.0.88` (line 1 of each serve log) |
| model | `Qwen/Qwen3-4B` from the host's Hugging Face cache, snapshot `1cfa9a7208912126459214e8b04321603b3df60c`, the only Qwen3-4B snapshot there. The serve logs do not print a revision |
| service | `make serve-planner` (port 8200) with `UVICORN_LOG_CONFIG` naming `planner_log_config.json`; worktree launches add `VENV_VLM=<main checkout>/.venv_vlm` |

## GPU

One planner service ran at a time. Each service's log ends with its shutdown before the next one's
log starts.

## Timeline (2026-10-11, CDT)

The service times are the serve logs' own stamps. The pass times are each run's `summary.json`
(`started_utc` and `finished_utc`, UTC − 5 h). The test times are the JUnit `timestamp`s.

| service | started | startup complete | what ran against it | `/plan` requests, all 200 | shutdown |
|---|---|---|---|---:|---|
| before (main checkout) | 18:04:06 | 18:05:04 | passes 1 and 2, 18:05:13–18:10:50 | 104 | 18:11:26 |
| arm A (worktree) | 18:11:43 | 18:12:34 | passes 1 and 2, 18:12:44–18:18:27 | 104 | 18:18:51 |
| arm B (worktree) | 18:18:52 | 18:19:51 | passes 1 and 2, 18:19:55–18:25:43 | 104 | 18:28:37 |
| arm C (worktree) | 18:28:38 | 18:29:33 | passes 1 and 2, 18:29:36–18:35:19; held-out pass, 18:35:19–18:35:58; live fixtures, 18:37:25 | 120 | 18:38:33 |
| `main`, for the live fixtures (main checkout) | 18:38:35 | 18:39:32 | live fixtures, 18:39:33 | 4 | 18:39:47 |

| other events | time |
|---|---|
| pre-registration committed, `d2fb7cc` | 18:03:55 |
| addendum and the before, A and B runs committed, `c9745b4` | 18:28:26 |
| autonomy suite, `main` | 18:38:07 |
| autonomy suite, branch | 18:38:14 |
| branch commits `38ce883`, `e568e4a`, `1031643` | 18:41:26 |
