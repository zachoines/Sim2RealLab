# Provenance — the planner's sign on a two-axis translate, 2026-10-11

captured_date: 2026-10-11, with the narrowed guard's runs on 2026-10-12 and the shipped tree's on
2026-10-13

## Host

| role | host | kernel | driver |
|---|---|---|---|
| planner service, sweeps and tests | gx10-d1d8 (ASUS GX10, a GB10 DGX Spark system, aarch64) | 7.0.0-1019-nvidia | 580.173.02 |

All eleven JUnit files carry `hostname="gx10-d1d8"`.

## Tree and stack

| runs | Sim2RealLab tree |
|---|---|
| before the change; the `main` suite; the `main` live-fixture service (the test module is the branch's) | the main checkout at `9f7d207` (the #238 merge), with `source/strafer_autonomy` unmodified |
| arms A, B and C; arm C's suite and live-fixture runs (2026-10-11) | a worktree of `task/planner-translate-two-axis-sign` off `9f7d207`, each arm's change applied in the working tree (`armA_prompt.patch`, `armB_prompt.diff`, `armC.patch`) |
| the narrowed guard (2026-10-12) | the same worktree. The suite's JUnit (20:29:36) is stamped 19 s before `14477af` and `a80c30f` were committed (20:29:55), so it ran on the uncommitted tree those commits recorded. The service, its passes and the live fixtures ran after the commits, at `a80c30f` |
| the shipped tree (2026-10-13) | the same worktree. The service, the live fixtures before the marker, and the sweep and held-out passes ran at `d40cdca` (committed 20:14:09), before `b3be9b0` existed. The marked fixtures, the suite and the marker's mutation ran on the clean tree at `b3be9b0`, and then on a clean worktree at `2841f74` with a service started from it (`tests/shipped_run_stamps.txt`, `tests/final_run_stamps.txt`). The package source is the same at all three commits |

- **The branch's commits came after the runs.** `38ce883`, `e568e4a` and `1031643` were all made at
  18:41:26 CDT. `armC.patch` applied to `9f7d207` reproduces their `prompt_builder.py`,
  `plan_compiler.py` and `test_plan_compiler.py` byte for byte.
- **The worktree ran with `PYTHONPATH`** naming its own `source/strafer_autonomy`, because
  `.venv_vlm`'s editable install points `strafer_autonomy` at the main checkout. The sweep tool ran
  under the same `PYTHONPATH`, and each run's `summary.json` records the `prompt_builder` it
  imported. That is the worktree's for A, B and C and for the narrowed and shipped passes, and
  the main checkout's for the before arm. The shipped and final services' environments are
  deposited (`runs/*_serve_planner.environ.txt`).

| | |
|---|---|
| interpreter | `.venv_vlm`: Python 3.12.3, torch 2.11.0+cu128, transformers 5.5.3, fastapi 0.135.3, uvicorn 0.44.0, pydantic 2.12.5 |
| NVRTC | `/usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvrtc.so.13.0.88` (line 1 of each serve log) |
| model | `Qwen/Qwen3-4B` from the host's Hugging Face cache, snapshot `1cfa9a7208912126459214e8b04321603b3df60c`, the only Qwen3-4B snapshot there. The serve logs do not print a revision |
| service | `make serve-planner` (port 8200) with `UVICORN_LOG_CONFIG` naming `planner_log_config.json`; worktree launches add `VENV_VLM=<main checkout>/.venv_vlm` |

## GPU

One planner service ran at a time. Each service's log ends with its shutdown before the next one's
log starts. On 2026-10-12 and 2026-10-13 no Kit process ran beside the services.

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

## Timeline (2026-10-12, CDT): the narrowed guard

| service | started | startup complete | what ran against it | `/plan` requests, all 200 | shutdown |
|---|---|---|---|---:|---|
| narrowed guard (worktree, `a80c30f`) | 20:47:01 | 20:47:58 | live fixtures, 20:48:08; pass 1, 20:48:22–20:51:13; held-out pass, 20:51:14–20:51:52 | 68 | 20:52:02 |

| other events | time |
|---|---|
| autonomy suite, the uncommitted narrowed tree | 20:29:36 |
| commits `14477af`, `a80c30f` | 20:29:55 |

## Timeline (2026-10-13, CDT): the shipped tree

| service | started | startup complete | what ran against it | `/plan` requests, all 200 | shutdown |
|---|---|---|---|---:|---|
| shipped (worktree, `d40cdca`) | 20:17:00 | 20:17:56 | live fixtures before the marker, 20:18:12; pass 1, 20:18:32–20:21:22; held-out pass, 20:21:22–20:22:00; live fixtures with the marker, 20:22:39; the marker's mutation, 20:23:11 | 76 | 20:23:30 |
| final (clean worktree, `2841f74`) | 20:53:27 | 20:54:22 | live fixtures, 20:54:39; the mutated module four times, requests 20:55:04–20:56:25 | 20 | 20:56:32 |

| other events | time |
|---|---|
| commit `d40cdca` (the guard withdrawn) | 20:14:09 |
| commit `b3be9b0` (tests) | 20:22:24 |
| autonomy suite, `b3be9b0` | 20:22:53 |
| commit `2841f74` (the live module's failure reporting; it also removes the dropped-axis brief) | 20:53:14 |
| autonomy suite, `2841f74` | 20:54:54 |

The shipped service's first launch attempt stopped at `make`'s NVRTC check, because `VENV_VLM` was
given in the environment rather than to `make` (`DEPOSIT.md`). It started no service and sent no
request.
