# Consolidate the workstation-side script / tooling layout

**Status:** Shipped 2026-06-07 in `aba26ab` (DGX).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/77

**Type:** refactor / tooling (non-test script + tool reorganization)
**Owner:** DGX agent (lane: top-level `Scripts/`, `source/strafer_lab/scripts/`, `source/strafer_lab/strafer_lab/tools/`, `source/strafer_vlm/`, top-level `Makefile`, the workstation-facing `docs/`)
**Priority:** P3 — tooling polish; doesn't block features. The non-harness majority is pickable now; the harness-owned cluster is sequenced (see [Coordination & sequencing](#coordination--sequencing)).
**Estimate:** L (~multi-day: a mechanical `git mv` sweep + import/subprocess/Makefile fixups + a wide-but-mostly-path-only doc pass + the placement-rule convention. The retirement calls and the harness sequencing are the only judgement parts.)
**Branch:** task/script-tooling-layout-consolidation (exists, already on)

## Story

As **an agent adding or running a workstation-side script**, I want **one
obvious home for runnable scripts and one for importable tool modules,
with a written rule that decides where a new file goes**, so that **I stop
guessing between three locations (top-level `Scripts/`, `source/strafer_lab/scripts/`,
`source/strafer_lab/strafer_lab/tools/`), every entry point is reachable at a
stable path, and the next person who adds an `eval_*.py` or a `build_*.py`
doesn't have to re-derive the convention by reading the existing mess.**

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md) — hosts, the `pip install -e source/<pkg>` contract, the **"Key entry-point scripts"** table this brief rewrites (lines ~79–91), and the **"Workspace layout"** block (line ~58 documents `Scripts/`).
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md) — `strafer_ros` is the Jetson/ROS lane and is **out of scope**; this brief is DGX/workstation-side only.
- [`context/conventions.md`](../../context/conventions.md) — the comment-style + **no-transient-references** rules (the executor will trip a `Phase 6` reference in `Scripts/scenegen/__init__.py`), and the **user-facing-documentation-maintenance** section that drives the doc sweep. This brief *adds a new section* to this file (the placement rule).
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- **Sibling briefs — coordinate, do not overlap** (see [Coordination & sequencing](#coordination--sequencing) for the full contract):
  - [`harness-architecture`](../harness/harness-architecture.md) — **owns `capture.py`, `teleop_capture.py`, `run_sim_in_the_loop.py`, `collect_demos.py`, and the retirement of four downstream scripts.** In flight.
  - [`strafer-lab-test-tree-unification`](strafer-lab-test-tree-unification.md) — owns the `test/` + `tests/` *file* layout.
  - [`unify-test-targets-and-ci`](unify-test-targets-and-ci.md) — owns `make test-*` + CI; explicitly **punts `make lint`/`format` rework**.
  - [`windows-workstation-bringup`](windows-workstation-bringup.md) — lane-claims `Scripts/` + `env_setup.sh` + `Makefile`; owns the `.ps1` launchers' future.
  - [`install-docs-consolidation`](../../parked/tooling/install-docs-consolidation.md) — will rewrite each package README's Install/Run sections. Parked.

## The problem (measured against `main` @ `ac8253f`)

Non-test workstation Python lives in **three** locations with no written
rule deciding which one a file belongs in. Verified counts (excluding
`strafer_ros`, which is out of scope):

| Location | What's there | Kind | Count |
|---|---|---|---|
| top-level `Scripts/` | training / inference / export / smoke + robot-asset authoring + `scenegen/` + `type_stubs/` + `*.ps1` | runnable (mostly) | 10 `.py` + `scenegen/` (10) + `type_stubs/` (2) + 3 `.ps1` |
| `source/strafer_lab/scripts/` | harness drivers, scene-gen, capture, demo collection, retired VLM/CLIP prep | runnable | 11 |
| `source/strafer_lab/strafer_lab/tools/` | importable helpers (writers, readers, scene accessors, gamepad, profiler) | importable library | 13 |
| `source/strafer_vlm/strafer_vlm/tools/` | Qwen2.5-VL grounding probes | runnable-as-module | 2 |

Six concrete defects, not just an eyesore:

1. **`Scripts/` (capital-S) is vestigial but still load-bearing.** It
   predates the `source/<pkg>/` package structure. Five genuinely-active
   entry points (`train_strafer_navigation.py`, `play_strafer_navigation.py`,
   `export_policy.py`, `benchmark_policy.py`, `test_strafer_env.py`) live
   there while their *exact siblings* (same package, same `$ISAACLAB -p`
   invocation) live in `source/strafer_lab/scripts/`.

2. **The README inventory is split-brained.** `source/strafer_lab/README.md`'s
   "Scripts and tools inventory" (lines ~113–160) groups entries **by
   category, not by location**, and lists both `Scripts/` (capital) and
   `scripts/` (lowercase) in the *same table* — including the anomalous
   `Scripts/capture.py` sitting in a row of `scripts/*` siblings (line 138).
   The doc has already given up on encoding *where* things live.

3. **A duplicated `prep_room_usds.py`.** `Scripts/scenegen/prep_room_usds.py`
   (901 lines) is an **older Windows-oriented** pipeline (`isaaclab.bat`,
   `C:\` paths, Omniverse Scene Optimizer decimation) with **no callers**.
   The **active** `source/strafer_lab/scripts/prep_room_usds.py` (634 lines,
   Infinigen `generate`/`ingest`/`presets`) is the one tests + the Makefile
   path + the active harness/multi-room briefs use. Same filename, two
   different pipelines — a foot-gun.

4. **An orphaned scene-gen pipeline.** All of `Scripts/scenegen/` (a
   Replicator-SDG pipeline: `generate_infinigen_rooms.sh` → `build_asset_manifest.py`
   → `validate_asset_pool.py` → `compose_scenes_replicator.py` →
   `split_scene_dataset.py` → `export_scene_stats.py`) has **no external
   callers** and its `__init__.py` is labelled with a transient
   `Phase 6` reference (banned by `conventions.md`). It appears superseded
   by the Infinigen pipeline in `source/strafer_lab/scripts/`, but a whole
   pipeline is not a safe auto-delete — flagged for human confirmation in
   [Retirement candidates](#retirement-candidates).

5. **`make lint`/`format` mis-scope.** The four lint/format targets
   (`Makefile:45–56`) run on `source/strafer_ros/ source/strafer_shared/ Scripts/`
   — they **never lint `strafer_lab`** and they point at `Scripts/`, the
   directory this brief empties. After the move they lint a near-empty dir.

6. **No rule for what's coming.** The planned epics already disagree with
   *themselves* on placement: `validator-evaluation` wants
   `Scripts/eval_transit_monitor.py`; `room-state-eval-harness` wants
   `scripts/eval_room_state.py` + `strafer_lab/tools/geodesic.py`;
   `mission-generator` wants **both** `strafer_lab/tools/build_mission_queue.py`
   *and* `Scripts/build_mission_corpus.py`; `backbone-bakeoff` wants
   `scripts/export_backbone_onnx.py`; `vla-v2-architecture` wants
   `strafer_lab/tools/build_vla_dataset.py`. Every one of these is a coin
   flip today. The rule this brief writes makes the choice for all of them.

## Current layout analysis

Classified on three axes: **runnable vs importable**, **needs-Kit vs
needs-IsaacLab(no-Kit) vs pure-python**, and **logical package**. Caller
columns are condensed — the full caller map (every `file:line`) was
gathered during authoring; the load-bearing callers are listed.

### A. top-level `Scripts/` (Python)

| File | Runnable? | Kit? | Real callers (non-doc) | Disposition |
|---|---|---|---|---|
| `train_strafer_navigation.py` | yes | IsaacLab, no-Kit | none in code (operator-run via `$ISAACLAB -p`); cheatsheet/README/DGX_SPARK_SETUP | → `scripts/` |
| `play_strafer_navigation.py` | yes | IsaacLab, no-Kit | operator-run; README/cheatsheet | → `scripts/` |
| `export_policy.py` | yes | IsaacLab, no-Kit | **`tests/test_export_policy.py`** (test-tree brief's orphan), cheatsheet | → `scripts/` |
| `benchmark_policy.py` | yes | **pure-python** | wraps `strafer_shared.policy_interface.benchmark_policy()`; operator-run | → `scripts/` |
| `test_strafer_env.py` | yes | IsaacLab, no-Kit | operator-run; cheatsheet | → `scripts/` |
| `capture.py` | yes | pure-python (dispatcher) | **subprocesses `teleop_capture.py`**; `tests/harness/test_capture_cli.py`; **harness-architecture's entry point** | → `scripts/` **(harness-coordinate)** |
| `setup_physics.py` | yes | needs-Kit | none (manual robot-USD authoring; 1383 lines; referenced by `mujoco-warp` spike) | → `scripts/asset_authoring/` |
| `collapse_redundant_xforms.py` | yes | pure-python (USD) | none (manual robot-USD authoring) | → `scripts/asset_authoring/` |
| `inspect_robot_prim_layout.py` | yes | needs-Kit | none (manual robot-USD inspection) | → `scripts/asset_authoring/` |
| `run_empty_lab.py` | yes | IsaacLab, no-Kit | `launch_isaac_lab.ps1:12` | → `scripts/asset_authoring/` (update the `.ps1` path) |

### B. `Scripts/scenegen/` and `Scripts/type_stubs/`

| File / dir | Runnable? | Kit? | Callers | Disposition |
|---|---|---|---|---|
| `scenegen/` (whole pipeline) | mixed | mixed | **none external**; `__init__.py` has `Phase 6` transient label | **flag for human → retire or move** (see Retirement) |
| `scenegen/prep_room_usds.py` | yes | needs-Kit | none (stale Windows dup of the active script) | **retire** (superseded) |
| `scenegen/infinigen_sdg_utils.py` | no (lib) | needs-Kit | imported only inside `scenegen/` | follows the scenegen decision |
| `type_stubs/{generate_stubs,test_stubs}.py` + `README.md` | yes | needs-Kit | none external (dev tooling) | → `scripts/type_stubs/` (absorb root `TYPE_STUBS_README.md`) |

### C. `Scripts/*.ps1` (Windows launchers)

`convert_step_to_usd.ps1`, `launch_isaac_lab.ps1`, `launch_isaac_sim.ps1`
— **leave in place** (Windows is a future target; see Out of scope).
`launch_isaac_lab.ps1` references `run_empty_lab.py`; update that one path
if `run_empty_lab.py` moves.

### D. `source/strafer_lab/scripts/` (already in the right *kind* of home)

All 11 are runnable. Seven are **live and stay put**; three are
**harness-owned retirement-candidates that relocate to `scripts/retired/`**
(see [Retirement candidates](#retirement-candidates)). The deletion of the
retirees stays harness's — this brief only segregates them so the live dir is
clean. The live harness drivers (`run_sim_in_the_loop.py`, `teleop_capture.py`,
`collect_demos.py`) are **not** moved or deleted here:

| File | Runnable | Kit? | Notes |
|---|---|---|---|
| `run_sim_in_the_loop.py` | yes | needs-Kit | **harness-owned**; Makefile `sim-bridge`/`-gui`/`-harness` invoke it (`Makefile:163,171,182`) |
| `teleop_capture.py` | yes | needs-Kit | **harness-owned**; imports 6 `tools/` modules; called by `capture.py` subprocess |
| `collect_demos.py` | yes | needs-Kit | **harness-owned** (gamepad mapping reused by teleop) |
| `roller_bounce_probe.py` | yes | IsaacLab, no-Kit | physics regression probe; referenced by `mujoco-warp` spike |
| `prep_room_usds.py` | yes | IsaacLab, no-Kit | active Infinigen pipeline; `tests/harness/test_prep_room_usds.py` + `strafer_autonomy/tests/test_prep_room_usds.py` |
| `postprocess_scene_usd.py` | yes | IsaacLab, no-Kit | called by `prep_room_usds.py`; tests |
| `generate_scenes_metadata.py` | yes | IsaacLab, no-Kit | scene-gen |
| `extract_scene_metadata.py` | yes | IsaacLab, no-Kit | Blender subprocess; tests |
| `generate_descriptions.py` | yes | pure-python | **harness retirement-candidate** → `scripts/retired/` (leaf script, no live importers) |
| `prepare_vlm_finetune_data.py` | yes | pure-python | **harness retirement-candidate** → `scripts/retired/` (imports `dataset_export`, which also retires) |
| `finetune_clip.py` | yes | pure-python | **harness retirement-candidate** → `scripts/retired/` (leaf script) |

The five non-harness, non-retirement ones (`roller_bounce_probe`,
`prep_room_usds`, `postprocess_scene_usd`, `generate_scenes_metadata`,
`extract_scene_metadata`) are already correctly placed — no move. The three
harness retirement-candidates above move into `scripts/retired/` rather than
staying mixed with live entry points — see [Retirement candidates](#retirement-candidates)
for the `retired/` mechanism and the harness coordination.

### E. `source/strafer_lab/strafer_lab/tools/` (importable library — already correct)

All 13 are importable modules (no `__main__`), pure-python. They are the
canonical "imported by scripts" home and **stay put**. Load-bearing
imports: `teleop_capture.py` imports `gamepad_reader`, `phase_profiler`,
`lerobot_writer`, `lerobot_depth`, `scene_paths`, `teleop_buttons`,
`teleop_mission_picker`; `generate_descriptions.py` imports `scene_labels`,
`spatial_description`; `prepare_vlm_finetune_data.py` imports `dataset_export`;
`run_sim_in_the_loop.py` imports `perception_writer`. **Cross-package
coupling to watch:** `source/strafer_autonomy/tests/` imports
`strafer_lab.tools.{bbox_extractor,dataset_export,infinigen_label_parser,perception_writer,scene_labels,spatial_description}`
— tool moves would break the autonomy test lane too (these stay put, so no
break — but any future tool move must check this).

- `dataset_export.py` is a **harness retirement-candidate** → moves to
  `strafer_lab/tools/retired/`. It is the one retiree with **live couplings**
  that must follow it: its importer `prepare_vlm_finetune_data.py` (also a
  retiree → `scripts/retired/`, so update that import to
  `strafer_lab.tools.retired.dataset_export`), and the live test
  `strafer_autonomy/tests/test_dataset_export.py` (update its import path so
  `make test-dgx` stays green — or retire the test alongside; **coordinate
  with the autonomy test lane**, default to repointing the import).
- `bbox_extractor.py` — **stays live in `tools/` (NOT `retired/`)**. It is the
  *producer* for the parked
  [`vlm-grounding-finetune`](../../parked/clip-validation/vlm-grounding-finetune.md)
  — owned for future use, not deletion. The opposite of a retiree.

### F. `source/strafer_vlm/strafer_vlm/tools/` (minor)

`live_qwen25vl_grounding.py` and `test_qwen25vl_grounding.py` are
**runnable** (`python -m strafer_vlm.tools.<x>`), pure-python, no importers.
By the rule (runnable → `scripts/`) they belong in a new
`source/strafer_vlm/scripts/`. They are part of the orphaned Qwen grounding
tooling (the [`vlm-grounding-finetune`](../../parked/clip-validation/vlm-grounding-finetune.md)
brief is parked); small and low-risk. **Recommended:** move both to
`source/strafer_vlm/scripts/`; **acceptable alternative:** leave them and
note them as "runnable-as-module, retained in tools/ pending the parked
grounding brief." Pick one and record it.

### G. `source/strafer_autonomy/` (no action — already compliant)

No `scripts/` or `tools/` scatter. Entry points are `[project.scripts]`
console entries (`strafer-autonomy-cli`, `strafer-executor`) mapping to
package modules. This is the cleanest pattern in the repo; named here only
to record that it needs nothing. (Console-script entry points are **not**
an option for the `strafer_lab` Kit scripts, which must launch via
`$ISAACLAB -p <path>` — hence `strafer_lab` keeps a `scripts/` dir rather
than adopting console entries.)

## Future script needs

The placement rule has to house work that's coming, not just what exists.
From the active/parked epics (read their briefs):

| Brief | New file it names | Kind | Rule says |
|---|---|---|---|
| [`mission-generator`](../harness/mission-generator.md) | `build_mission_queue.py` | importable | `strafer_lab/tools/` |
| [`mission-generator`](../harness/mission-generator.md) | `build_mission_corpus.py` | runnable CLI | `strafer_lab/scripts/` |
| [`validator-evaluation`](../clip-validation/validator-evaluation.md) | `eval_transit_monitor.py` | runnable | `strafer_lab/scripts/` |
| [`room-state-eval-harness`](../multi-room/room-state-eval-harness.md) | `eval_room_state.py` | runnable | `strafer_lab/scripts/` |
| [`room-state-eval-harness`](../multi-room/room-state-eval-harness.md) | `geodesic.py` | importable | `strafer_lab/tools/` |
| [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md) | `export_backbone_onnx.py` | runnable | `strafer_lab/scripts/` |
| [`vla-v2-architecture`](../../parked/experimental/vla-v2-architecture.md) | `build_vla_dataset.py` | importable | `strafer_lab/tools/` |
| [`harness-architecture`](../harness/harness-architecture.md) | `loader_helpers.py`, `export_*.py` converters | importable | `strafer_lab/tools/` |

The point is not to pre-create these — it's that after this brief there is
**one defensible answer** for each, and the briefs that currently name a
`Scripts/`-capital path get reconciled to the rule when they're picked up
(noted in their cross-references rather than rewritten preemptively here).

## Retirement candidates

The goal is a **fully consolidated layout at the end of this brief** — no
deprecated files left mixed in with live entry points, and no fate left
"deferred to a PR that may not land for weeks." There are two kinds of
retirement, handled differently. **Ambiguous calls are flagged for human
confirmation — never auto-delete.**

### Kind 1 — owner-retired (deprecated, another brief owns the deletion) → `retired/`

[`harness-architecture`](../harness/harness-architecture.md) has marked four
downstream scripts for retirement and will **delete them in its
implementation PRs as it supersedes each** (and in at least one case *mine*
them first — it says the captioner's prompt is "recoverable from
[`generate_descriptions.py`]'s Stage 2"). Rather than leave them sitting in
`scripts/` / `tools/` looking like first-class entry points until then, this
brief relocates them into a sibling **`retired/`** folder. This keeps the
live dirs clean **and** keeps the code physically present and discoverable
for the owner to mine, instead of forcing a `git show` of a deleted file.

`retired/` semantics (state this in the placement rule): *deprecated,
retained for reference / mining by its owning brief, pending that brief's
deletion; **not imported or invoked by any live entry point**.* It is a
holding area, not a permanent home — the owning brief empties it on ship.

| Candidate | Current path | → | Owner / deletion |
|---|---|---|---|
| `generate_descriptions.py` | `scripts/` | `scripts/retired/` | harness (captioner mission-source supersedes; mines the Stage-2 prompt) |
| `prepare_vlm_finetune_data.py` | `scripts/` | `scripts/retired/` | harness (direct LeRobot v3 loading supersedes) |
| `finetune_clip.py` | `scripts/` | `scripts/retired/` | harness (the `cotrained-retrieval-augmented` recipe supersedes) |
| `dataset_export.py` | `tools/` | `tools/retired/` | harness (direct LeRobot v3 loading supersedes) |

Mechanics:
- `scripts/retired/` needs no `__init__.py` (scripts aren't imported).
  `tools/retired/` **does** — add `strafer_lab/tools/retired/__init__.py` so
  `dataset_export` is importable as `strafer_lab.tools.retired.dataset_export`.
- **Drag the live couplings.** `prepare_vlm_finetune_data.py` imports
  `dataset_export` (both retire together; update the import to the
  `tools.retired` path). The live test
  `strafer_autonomy/tests/test_dataset_export.py` imports
  `strafer_lab.tools.dataset_export` — repoint it to the `retired` path so
  `make test-dgx` stays green (or retire the test alongside; coordinate with
  the autonomy lane).
- **Notify the owner in the same PR:** repoint `harness-architecture.md`'s
  "Retired downstream scripts" list (and its `loader_helpers.py` /
  `export_*.py` future-converter homes if affected) to the `retired/` paths,
  so the harness deletion PRs delete from the right place. The deletion
  obligation stays harness's — this brief only relocates.
- **`bbox_extractor.py` is explicitly NOT a retiree** — it's the live
  producer for the parked grounding brief; it stays in `tools/`.

### Kind 2 — this-brief-retired (no future consumer) → delete or human-flag

These have no owning brief that needs them later, so the clean end-state is
deletion (git history is the record); the orphaned pipeline is the one
human-gated call.

| Candidate | Evidence | Recommendation |
|---|---|---|
| `Scripts/scenegen/prep_room_usds.py` | Older Windows pipeline (`isaaclab.bat`, `C:\` paths); **no callers**; superseded by the active `source/strafer_lab/scripts/prep_room_usds.py` (different, current pipeline; the one tests + briefs use). | **Delete** (low risk; git history holds it; the active script is unambiguous). |
| `Scripts/scenegen/` (rest of the pipeline) | **No external callers** anywhere; `__init__.py` carries a `Phase 6` transient label; the active scene-gen path is the Infinigen pipeline in `scripts/`. But it's a *complete* Replicator-SDG pipeline (asset-manifest → compose → split → stats) — possibly a parked capability, not dead. | **Flag for human.** Three options: (a) delete the dir, (b) move to `source/strafer_lab/scripts/scenegen/` and fix the `Phase 6` label (if it's a live capability), or (c) park it under `scripts/retired/scenegen/` if you want it discoverable but inert. Do **not** delete without confirmation. |
| root `TYPE_STUBS_README.md` | Duplicates `Scripts/type_stubs/README.md`. | **Merge** into the moved `scripts/type_stubs/README.md`; delete the root copy (it's a pointer). |
| `Scripts/*.ps1` launchers | No DGX-side callers; Windows-only. | **Keep, leave in place** (scope: Windows future; `windows-workstation-bringup` owns them). |
| `strafer_vlm/.../{live,test}_qwen25vl_grounding.py` | Orphaned grounding probes; parked brief. | **Do not retire** — relocate per §F or leave; the parked grounding brief may adopt them. |

Scripts with **no callers but a legitimate manual purpose** (NOT retirement):
`setup_physics.py`, `collapse_redundant_xforms.py`, `inspect_robot_prim_layout.py`,
`run_empty_lab.py` — these are one-shot robot-asset authoring/inspection
utilities (run by hand to (re)build the robot USD; `setup_physics.py` was
touched as recently as the teleop-perf physics work). They get a home, not a
grave (`scripts/asset_authoring/`).

## Target layout + placement rule

### Layout

```text
source/strafer_lab/
  scripts/                          # ALL runnable strafer_lab entry points (flat)
    train_strafer_navigation.py     # ← moved from Scripts/
    play_strafer_navigation.py      # ← moved from Scripts/
    export_policy.py                # ← moved from Scripts/
    benchmark_policy.py             # ← moved from Scripts/
    test_strafer_env.py             # ← moved from Scripts/
    capture.py                      # ← moved from Scripts/  (harness-coordinate)
    run_sim_in_the_loop.py          # stays (harness-owned)
    teleop_capture.py               # stays (harness-owned)
    collect_demos.py                # stays (harness-owned)
    roller_bounce_probe.py          # stays
    prep_room_usds.py               # stays (active Infinigen pipeline)
    postprocess_scene_usd.py        # stays
    generate_scenes_metadata.py     # stays
    extract_scene_metadata.py       # stays
    asset_authoring/                # one-shot, run-by-hand USD authoring (NEW)
      setup_physics.py              # ← moved from Scripts/
      collapse_redundant_xforms.py  # ← moved from Scripts/
      inspect_robot_prim_layout.py  # ← moved from Scripts/
      run_empty_lab.py              # ← moved from Scripts/
    type_stubs/                     # ← moved from Scripts/type_stubs/ (+ root README merged in)
      generate_stubs.py
      test_stubs.py
      README.md
    retired/                        # deprecated, pending harness deletion (NEW; emptied by harness)
      generate_descriptions.py      # ← from scripts/ (harness mines Stage-2 prompt)
      prepare_vlm_finetune_data.py  # ← from scripts/
      finetune_clip.py              # ← from scripts/
  strafer_lab/
    tools/                          # ALL importable library modules (UNCHANGED location)
      ... 12 live modules stay put (incl. bbox_extractor — live producer) ...
      retired/                      # NEW; needs __init__.py
        __init__.py
        dataset_export.py           # ← from tools/ (importer + autonomy test follow it)

source/strafer_vlm/
  scripts/                          # NEW (if §F "move" chosen)
    live_qwen25vl_grounding.py
    test_qwen25vl_grounding.py

# top-level Scripts/ retained ONLY for the Windows .ps1 launchers — its
# fate (and the scenegen/ decision) handed to windows-workstation-bringup.
```

### The rule (the durable contribution — encode in `conventions.md`)

Add a new **`## Script and tool placement`** section to
[`context/conventions.md`](../../context/conventions.md) stating:

> Workstation/DGX-side, non-test Python belongs to a **package**, never to
> a top-level dir. There is no `Scripts/` (capital-S) for Python. Decide a
> file's home with two questions:
>
> 1. **Is it imported, or run directly?**
>    - Something `import`s it (a library; no `if __name__ == "__main__"`) →
>      it's a **tool** → `source/<pkg>/<pkg>/tools/`.
>    - It's run directly (has a CLI/`__main__`; invoked via `$ISAACLAB -p`,
>      `python -m`, or a `make` target) and nothing imports it → it's a
>      **script** → `source/<pkg>/scripts/`.
> 2. **(scripts only) operator/pipeline entry point, or one-shot asset
>    authoring?** Training/capture/scene-gen/eval/export → `scripts/` (flat).
>    Run-by-hand utilities that (re)build or inspect a robot/asset USD →
>    `scripts/asset_authoring/`.
>
> Tie-breaker: *"does anything import this?"* Yes → `tools/`. No, and it has
> a `__main__` → `scripts/`. Directories are always lowercase
> (`scripts/`, `tools/`) — never `Scripts/`.
>
> **Deprecated-but-not-yet-deletable code** (a file another brief owns the
> deletion of, or that's kept only so its successor can mine it) lives in a
> sibling **`retired/`** under `scripts/` or `tools/` — *deprecated, retained
> for reference, not imported or invoked by any live entry point*. `retired/`
> is a holding area the owning brief empties on ship; nothing in a live
> `scripts/`/`tools/` may import from it. Code with no future consumer is
> deleted outright (git history is the record), not parked in `retired/`.

## Per-file move / retire table

`git mv` everything (preserve history). Update every import / subprocess
target / Makefile path / doc path that points at the old location in the
**same commit** as the move.

| From | To | Coupled fixups |
|---|---|---|
| `Scripts/train_strafer_navigation.py` | `source/strafer_lab/scripts/` | cheatsheet, README, `repo-topology.md`, `Readme.md`, `DGX_SPARK_SETUP.md` paths |
| `Scripts/play_strafer_navigation.py` | `source/strafer_lab/scripts/` | same docs |
| `Scripts/export_policy.py` | `source/strafer_lab/scripts/` | **`tests/test_export_policy.py`** import/path (coordinate with test-tree brief); docs |
| `Scripts/benchmark_policy.py` | `source/strafer_lab/scripts/` | docs |
| `Scripts/test_strafer_env.py` | `source/strafer_lab/scripts/` | docs |
| `Scripts/capture.py` | `source/strafer_lab/scripts/` | **`harness-architecture.md` refs + its tier branches** (see sequencing); `tests/harness/test_capture_cli.py`; the `teleop_capture.py` subprocess path resolution; docs (`HARNESS_DATA_CAPTURE.md`, `INTEGRATION_SIM_IN_THE_LOOP.md`, `SYSTEM_FLOW_DIAGRAMS.md`, `SCENE_PROVIDER_CONTRACT.md`) |
| `Scripts/setup_physics.py` | `source/strafer_lab/scripts/asset_authoring/` | `mujoco-warp-physics-backend-spike.md` ref |
| `Scripts/collapse_redundant_xforms.py` | `source/strafer_lab/scripts/asset_authoring/` | `Readme.md` structure block |
| `Scripts/inspect_robot_prim_layout.py` | `source/strafer_lab/scripts/asset_authoring/` | — |
| `Scripts/run_empty_lab.py` | `source/strafer_lab/scripts/asset_authoring/` | **`launch_isaac_lab.ps1:12`** path (edit the one line; `.ps1` stays in place) |
| `Scripts/type_stubs/` | `source/strafer_lab/scripts/type_stubs/` | merge root `TYPE_STUBS_README.md` → delete root copy; update `TYPE_STUBS_README.md` referrers |
| `scripts/generate_descriptions.py` | `source/strafer_lab/scripts/retired/` | repoint `harness-architecture.md`'s "Retired downstream scripts" path |
| `scripts/prepare_vlm_finetune_data.py` | `source/strafer_lab/scripts/retired/` | update its `import dataset_export` → `strafer_lab.tools.retired.dataset_export`; `strafer_vlm/README.md` ref; `harness-architecture.md` path |
| `scripts/finetune_clip.py` | `source/strafer_lab/scripts/retired/` | `harness-architecture.md` path; README inventory |
| `tools/dataset_export.py` | `source/strafer_lab/strafer_lab/tools/retired/` (+ `__init__.py`) | **`strafer_autonomy/tests/test_dataset_export.py`** import path (keep `make test-dgx` green — or retire test; coordinate autonomy lane); `prepare_vlm_finetune_data.py` importer (above); `harness-architecture.md` path; README inventory |
| `Scripts/scenegen/prep_room_usds.py` | **delete** | confirm no unique behavior is lost vs the active script (git history holds it) |
| `Scripts/scenegen/` (rest) | **flag for human** → delete, or `scripts/scenegen/` (fix `Phase 6` label), or `scripts/retired/scenegen/` | `scenegen/__init__.py` transient-ref cleanup if kept |
| `strafer_vlm/.../live_qwen25vl_grounding.py` | `source/strafer_vlm/scripts/` (recommended) | `strafer_vlm/README.md`; or leave per §F |
| `strafer_vlm/.../test_qwen25vl_grounding.py` | `source/strafer_vlm/scripts/` (recommended) | same |
| `Scripts/*.ps1` | **leave in place** | update internal `.py` path refs only |
| `source/strafer_lab/scripts/*` (live harness + scene-gen set) | **no move** | already correct per rule |
| `source/strafer_lab/strafer_lab/tools/*` (live, incl. `bbox_extractor`) | **no move** | already correct per rule |

## Entry-point fixes

Audited the `Makefile` and launchers:

- **`make lint` / `lint-fix` / `format` / `format-check`** (`Makefile:45–56`)
  reference `source/strafer_ros/ source/strafer_shared/ Scripts/`. After the
  move `Scripts/` holds only `.ps1`. **Fix:** repoint these at
  `source/strafer_lab/scripts/` (+ optionally `source/strafer_lab/strafer_lab/`).
  The deeper observation — these targets **never linted `strafer_lab` at all**
  — is a real gap, but a full lint/format rework is **out of scope**
  (`unify-test-targets-and-ci` explicitly punted it). Do the minimal path
  correction that keeps lint runnable; note the gap for a future lint sweep.
- **`make sim-bridge` / `sim-bridge-gui` / `sim-harness`** (`Makefile:163,171,182`)
  invoke `source/strafer_lab/scripts/run_sim_in_the_loop.py` — **unchanged**
  (that file stays). No edit; but they are smoke tests (below).
- **No `make` target exists** for the most-run scripts (`train_*`, `capture.py`),
  so their canonical invocation is a bare `$ISAACLAB -p <path>` in the
  cheatsheet/runbooks — those paths change. Adding `make train` / `make capture`
  wrappers is tempting but is *invocation* unification — **out of scope**
  (sibling brief's spirit); just fix the documented paths.
- **`launch_isaac_lab.ps1:12`** points at `run_empty_lab.py`; update that one
  path. The `.ps1` files themselves stay where they are.

No targets are *broken* today; the lint targets are the only ones that go
stale on the move.

## Documentation updates

The blast radius is wide but **mostly mechanical path substitution.** Update
every surface that names a moved/retired path, in the same PR. Grouped by
how much thought each needs:

**Structural (rewrite the layout description):**
- `source/strafer_lab/README.md` "Scripts and tools inventory" (~113–160) —
  regroup **by location** (the rule), fix the `Scripts/capture.py` anomaly,
  drop retired entries. **Coordinate with `install-docs-consolidation`:** that
  brief rewrites this README's *Install/Run* sections — touch only the
  *inventory* + path strings here; don't rewrite Install/Run prose (avoids a
  double rewrite of the same file).
- `context/repo-topology.md` "Workspace layout" (~58) + "Key entry-point
  scripts" table (~79–91) — the canonical topology doc; update paths and the
  `Scripts/` line.
- `Readme.md` "Repository structure" (~148–155) — the `Scripts/` tree block.

**Mechanical path-only (find/replace the old path string):**
- `docs/example_commands_cheatsheet.md` (~20 one-liners)
- `docs/INTEGRATION_SIM_IN_THE_LOOP.md` (heaviest; many `run_sim_in_the_loop`/`capture` refs — most paths *unchanged* since those files stay, but `capture.py` path changes)
- `docs/HARNESS_DATA_CAPTURE.md`, `docs/SYSTEM_FLOW_DIAGRAMS.md` (Mermaid hyperlinks — **sweep edge labels per `conventions.md` after editing**), `docs/MISSION_VALIDATION_ARCHITECTURE.md`, `docs/SCENE_PROVIDER_CONTRACT.md`
- `docs/DGX_SPARK_SETUP.md` — **note:** `install-docs-consolidation` retires this file; do the path fix only if that brief hasn't landed first (coordinate so it isn't fixed-then-deleted).
- `source/strafer_vlm/README.md` (if §F move chosen), `TYPE_STUBS_README.md` (merge/delete)
- `harness-architecture.md` — **repoint its "Retired downstream scripts" list** (`generate_descriptions.py`/`prepare_vlm_finetune_data.py`/`finetune_clip.py` → `scripts/retired/`; `dataset_export.py` → `tools/retired/`) **and** its `capture.py` references, so the harness deletion PRs target the right paths. This is the owner-notification the `retired/` move requires.
- Other task briefs that name moved paths (`infinigen-scene-corpus.md`, `mujoco-warp-physics-backend-spike.md`, `windows-workstation-bringup.md`, `strafer-direct-sim-validation.md`, the sim-performance briefs) — update **active/parked** brief paths; **leave `completed/` briefs untouched** (historical record).
- `.env.example` / `env_setup.sh` reference `prep_room_usds.py` by name — that file does **not** move, so no change (verify).

**New:** add the `## Script and tool placement` section to `context/conventions.md`.

Run `tools/check_brief_links.sh` and the `conventions.md` Mermaid sweep
(`rg '\|[^|]*[()[\]{}][^|]*\|' <edited Mermaid files>`) before pushing.

## Coordination & sequencing

This brief deliberately reshapes directories four other in-flight/parked
briefs touch. The non-overlap contract:

- **`harness-architecture` (in flight, P1) owns the capture cluster and the
  four retirements.** It has `capture.py`, `teleop_capture.py`,
  `run_sim_in_the_loop.py`, and `collect_demos.py` baked into its CLI spec,
  three implementation tiers, and acceptance criteria, and it owns the
  **deletion** of `generate_descriptions.py` / `prepare_vlm_finetune_data.py`
  / `finetune_clip.py` / `dataset_export.py`. **Therefore:**
  (a) `run_sim_in_the_loop.py`, `teleop_capture.py`, `collect_demos.py` are
  already in `scripts/` and **stay put** — no move;
  (b) `Scripts/capture.py` → `scripts/capture.py` is the one **live**
  harness-owned file that moves — do it **in coordination with harness**
  (bundle with a tier PR, or land after Tier 2 stabilizes the cluster) and
  update `harness-architecture.md`'s `capture.py` references in the same PR;
  (c) the four retirement-candidates **do move — into `retired/`** (not left
  mixed with live entry points, so the end-state layout is clean), and this
  brief **repoints `harness-architecture.md`'s "Retired downstream scripts"
  paths to the `retired/` locations in the same PR**. The deletion obligation
  stays harness's: it empties `retired/` as it supersedes each (and mines
  `generate_descriptions.py`'s Stage-2 prompt from there). This is a
  reference-preserving relocation + a cheap brief-ref update — but it **does**
  touch harness-owned files, so it must be coordinated, not raced. The
  operator may prefer to **park this brief behind harness Tier 2** for that
  reason; default here is active-P3 with the capture move + retired/ moves
  sequenced. The non-harness ~80% (RL scripts, asset-authoring, type_stubs,
  scenegen, the rule, the doc/lint fixes) has **no harness dependency** and is
  pickable immediately.

- **`strafer-lab-test-tree-unification` (active) owns test *files*.** This
  brief changes the *import targets* those tests reference
  (`test_export_policy.py` → `export_policy.py`; `test_capture_cli.py` →
  `capture.py`). Coordinate: whichever lands second updates the references;
  if both are in flight, the test-tree brief's final file homes are the
  contract, and this brief updates the import targets at those homes. Do not
  move test files here.

- **`unify-test-targets-and-ci` (active) punted lint/format.** This brief
  does the minimal lint *path* fix only (above); the broader unification
  stays that brief's (deferred) scope.

- **`windows-workstation-bringup` (active) lane-claims `Scripts/` + the
  `.ps1` launchers.** Leave the `.ps1` in place; hand the residual
  top-level `Scripts/` dir (and the `scenegen/` retire-or-move decision's
  Windows angle) to that brief. Flag the lane overlap to the operator so the
  two don't reshape `Scripts/` concurrently.

- **`install-docs-consolidation` (parked) rewrites README Install/Run.** As
  noted in Documentation updates: touch only paths + the inventory in
  `strafer_lab/README.md`, not Install/Run prose; don't fix-then-let-it-delete
  `DGX_SPARK_SETUP.md`.

## Acceptance criteria

- [ ] Top-level `Scripts/` holds **no Python** — every `.py` is moved
      (history-preserving `git mv`), retired-with-evidence, or (for the
      scenegen pipeline) dispositioned per the human-confirmed call.
      **Updated during execution (operator decision):** the directory is
      removed **entirely** — its three Windows `.ps1` launchers were
      deleted too; `windows-workstation-bringup` authors fresh launchers
      against the new `source/<pkg>/scripts/` layout.
- [ ] `source/strafer_lab/scripts/` is the single home for runnable
      `strafer_lab` entry points; `source/strafer_lab/strafer_lab/tools/` is
      the single home for importable modules; one-shot USD-authoring
      utilities live under `scripts/asset_authoring/`.
- [ ] **Clean end-state — no deprecated code mixed with live.** The four
      harness-owned retirement-candidates live under `scripts/retired/` +
      `tools/retired/` (with `tools/retired/__init__.py`); `bbox_extractor.py`
      stays **live** in `tools/`. Nothing in a live `scripts/`/`tools/`
      imports from `retired/` (sweep: `rg -n "tools\.retired|scripts\.retired|from .*retired import" source/strafer_lab/scripts source/strafer_lab/strafer_lab/tools --glob '!retired/**'` returns nothing). `harness-architecture.md`'s retirement paths point at the `retired/` locations. `make test-dgx` stays green (the `dataset_export` test import followed the move).
- [ ] The placement rule is written as a new `## Script and tool placement`
      section in `context/conventions.md`, and every future-script path
      named in the epics ([Future script needs](#future-script-needs)) has
      an unambiguous home under it.
- [ ] **Smoke — bridge:** `make sim-bridge` launches Isaac Sim + the ROS 2
      bridge headless (reaches the mainloop / `/clock` publishing) after the
      refactor. `sim-bridge-gui` opens the viewport.
- [ ] **Smoke — training:** `$ISAACLAB -p source/strafer_lab/scripts/train_strafer_navigation.py`
      (with the documented args) starts a training run from its new path
      without import errors.
- [ ] **Smoke — harness capture:** `make sim-harness` (`run_sim_in_the_loop.py
      --mode harness`) launches, **and** `capture.py` teleop
      (`$ISAACLAB -p source/strafer_lab/scripts/capture.py --driver teleop
      --mission-source scene-metadata ...`) reaches the teleop driver from
      its new path (the `capture.py → teleop_capture.py` subprocess hand-off
      resolves correctly).
- [ ] **Smoke — export/play/benchmark:** `export_policy.py` round-trips an
      artifact and `play_strafer_navigation.py` / `benchmark_policy.py` run
      from their new paths.
- [ ] Test suites stay green from their runners: `isaaclab -p
      source/strafer_lab/run_tests.py all`, `make test-harness`, and
      `make test-dgx` (the autonomy tests that import `strafer_lab.tools.*`)
      — no import breaks from moved targets. `make lint` runs against the
      corrected paths.
- [ ] No source file references a moved path by a stale string (sweep:
      `rg -n "Scripts/" --glob '!docs/tasks/completed/**' --glob '!*.ps1'`
      returns only intentional Windows-launcher / historical hits).
- [ ] If your work invalidates a fact in any referenced context module,
      package README, top-level `Readme.md`, or guide under `docs/`, update
      those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics. `tools/check_brief_links.sh`
      passes; Mermaid edge-label sweep is clean.

## Investigation pointers

- The split itself: `Scripts/` vs `source/strafer_lab/scripts/` vs
  `source/strafer_lab/strafer_lab/tools/` (all three listed in
  `source/strafer_lab/README.md:113-160`).
- Duplicate: `diff Scripts/scenegen/prep_room_usds.py source/strafer_lab/scripts/prep_room_usds.py` — different pipelines, same name.
- Orphan pipeline + transient label: `Scripts/scenegen/__init__.py` (the `Phase 6` reference).
- Harness coupling: `harness-architecture.md` (CLI at L52–64, `capture.py` refs at L23/52/484/530, retirements at L456–465).
- Lint mis-scope: `Makefile:45-56`. Bridge/harness targets: `Makefile:158-189`.
- Cross-package test coupling: `source/strafer_autonomy/tests/test_{bbox_extractor,dataset_export,scene_labels,spatial_description,infinigen_label_parser,perception_writer}.py`.
- Subprocess hand-off: `Scripts/capture.py:60` → `teleop_capture.py`.
- `.ps1` path ref: `Scripts/launch_isaac_lab.ps1:12` → `run_empty_lab.py`.

## Out of scope

- **`strafer_ros/` (the Jetson/ROS lane).** Its top-level helper scripts
  (`diagnose_roboclaw.py`, `tune_pid.py`, `ros_test_*.py`, `test_d555_camera.py`)
  and `*/scripts/` dirs are a separate system — do not touch.
- **The test trees** (`test/`, `tests/`) and `make test-*` / CI — owned by
  the two sibling test briefs. This brief only updates *import targets* the
  tests point at, never test files or runners.
- **A full `make lint`/`format` rework** (incl. the strafer_lab-never-linted
  gap). Minimal path correction only.
- **`make` wrappers for `train`/`capture`** — invocation unification, not
  layout. File separately if wanted.
- **Windows / `.ps1` redesign.** (Updated during execution: the operator
  chose to delete the three legacy launchers with the rest of `Scripts/`
  rather than leave them; `windows-workstation-bringup` authors fresh
  Windows launchers and owns the PowerShell story.)
- **Retiring or rewriting the four harness-owned downstream scripts** —
  `harness-architecture` deletes them as it supersedes each.
- **Rewriting package README Install/Run prose** — `install-docs-consolidation`.
- **Editing `completed/` briefs** for path consistency — historical record.
