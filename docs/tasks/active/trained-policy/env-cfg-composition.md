# Composable env_cfg axes — split sensor / scene / realism

**Type:** refactor (env_cfg tree)
**Owner:** DGX agent
**Priority:** P2 — cleans up an axis-conflation that's currently forcing every consumer (teleop, bridge, RL training, future scripted captioner / coverage) to take ALL camera outputs and the full Infinigen scene-loading stack even when they don't need them; also the boundary that makes "USD becomes a parameter" possible.
**Estimate:** M (incremental refactor — keep the existing variants working through a deprecation window; the actual axis split is mechanical).
**Branch:** `task/env-cfg-composition`
**Recommended ordering, not a hard block:** Land [`scene-provider-contract`](../harness/scene-provider-contract.md) first so this brief can commit to the contract's storage-agnostic artifact shape without retroactive design drift. The two can ship in parallel if needed — neither blocks the other at the code level — but minimal rework is contract-first.
**Subsumes:** the **per-env-variant camera toggle** deliverable in [`teleop-perf-architecture.md` §C](../sim-performance/teleop-perf-architecture.md#c-per-env-variant-camera-configuration-design). That item is the narrowest version of the composition this brief proposes; doing it twice would be rework. The perf brief is amended in the same PR that ships this brief's first commit to defer that item here.

## Decision (2026-05-30): clean break, no legacy aliases

The operator's explicit call, overriding this brief's original
backward-compat-first framing: **do not preserve the old gym IDs or
the 39-class hierarchy.** A maximally clean composed interface
(sensors × scene source × realism × any future axis) replaces them
outright, and **every caller is migrated to the new IDs in the same
PR.** We pay the migration cost once, now, rather than carry an
alias layer forward as tech debt.

This inverts the original "all existing gym IDs continue to resolve
under the same names" gate. Two consequences:

1. **Names change freely; the observation/action *contract* does
   not.** What the env emits per step — obs tensor shape, key
   ordering, action layout, DR/noise semantics — must stay
   **byte-identical** for the composition that replaces each old RL
   config, so the in-flight DEPTH inference package
   ([`completed/inference-package.md`](../../completed/inference-package.md))
   keeps working and any trained checkpoint stays valid. Changing
   the obs/action contract is a *different* brief's lane
   ([`observation-contract-cleanup`](../../completed/observation-contract-cleanup.md)
   / [`recurrent-state-contract`](recurrent-state-contract.md)) and
   is explicitly out of scope here. The snapshot gate is re-keyed
   from "old gym ID resolves identically" to "the new composition
   equivalent to old config X emits an identical obs/action tensor."

2. **Complete caller migration is part of acceptance, not a
   follow-up.** The clean break means no caller can keep using an
   old ID. The full inventory (verified by grep 2026-05-30) is in
   [the migration acceptance section](#full-caller--reference-inventory-must-all-migrate-in-this-pr).

## Story

As a **harness operator and a future RL / VLA contributor** I want **env configurations to compose along the orthogonal axes they actually vary on — sensor stack × scene source × realism level — instead of inheriting a monolithic `_BaseInfinigenPerceptionNavEnvCfg` that bakes all three together** so that **a teleop run can ask for "RGB only, Infinigen, realistic" without paying for depth cameras it discards, an RL training run can ask for "RGB+depth, ProcRoom or no-cam, three realism levels" without forking the env class hierarchy, and a hypothetical future scene source (ProcTHOR / Habitat / hand-authored USD per `scene-provider-contract`) can plug in by parameter rather than by writing a new env subclass.**

## Motivation — three orthogonal axes are currently conflated

[`StraferNavEnvCfg_Real_InfinigenPerception`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py) (and its many siblings) bakes three independent concerns into the class name + class body:

| Axis | Today's variant labels | Concern owned |
|---|---|---|
| **Sensor stack** | `Perception` (full-res RGB + depth + policy 80×60 RGB + depth), `Depth` (policy 80×60 depth only), `NoCam` (no camera) | What data products the env exposes per step + what's in the `TiledCamera` graph |
| **Scene source** | Infinigen (hardcoded `_get_infinigen_spawn_points_xy` + scene-USD picker), ProcRoom (procedurally generated room geometry), missing (`NoCam` ditches scene loading entirely) | Where the USD comes from + how the metadata is loaded |
| **Realism level** | `Ideal` (no DR), `Real` (single tier of DR), `Robust` (full DR + extra noise) | Domain randomization events + observation noise terms |

Today, ALL three vary by class name and inheritance. So `StraferNavEnvCfg_Real_InfinigenPerception` is one specific cell in a 3D matrix that gets a class. We have ~15 of these classes because the matrix is partially populated; each new combination needs a new subclass. The harness teleop driver consumes the `_Perception` cell because no narrower one exists; the bridge sim-in-the-loop consumes the same cell for the same reason; and a future scripted captioner running on the same scenes would also have to consume that cell or get its own subclass.

The cost is two-fold:

1. **Per-consumer waste**: teleop doesn't use the policy 80×60 camera's depth — but the env produces it every step because the class includes it. The `teleop-perf-architecture` brief's "drop perception render" lever is downstream of this same conflation.
2. **No place for "USD becomes a parameter"**: when `scene-provider-contract` lands, swapping in a non-Infinigen USD should be a config-level change, not a new env subclass. Today the scene source is baked into `_BaseInfinigenPerceptionNavEnvCfg.__post_init__` via the `_get_infinigen_spawn_points_xy` call.

## Acceptance

Ship a refactor that meets all of:

- [ ] Collapse the env-cfg hierarchy into the composed interface. Note: `_BaseStraferNavEnvCfg` **already exists** as a parent of `_BaseInfinigenPerceptionNavEnvCfg` (`strafer_env_cfg.py:1296`), and there are **39** `NavEnvCfg` classes today (PLAY variants double the count), not the ~15 this brief originally assumed. The task is not a rename — it is to re-express the whole tree as compositions over the three axis-cfgs and **delete the per-combination subclasses** (no thin shims left behind; the clean-break decision above forbids an alias layer). Reconcile the actual current hierarchy before implementing and report the mapping.
- [ ] Introduce three orthogonal mixin-style configurations the variants compose from:
  - `SensorStackCfg` enumerating `cameras_required` (the proposal from the perf brief's §C — `rgb_full`, `depth_full`, `rgb_policy`, `depth_policy`)
  - `SceneSourceCfg` parameterizing the scene-USD discovery / metadata loading (Infinigen today; ProcRoom + foreign-USD via `scene-provider-contract` once that brief ships)
  - `RealismCfg` selecting Ideal / Real / Robust DR + noise (already mostly factored — confirm the split is clean)
- [ ] Concrete composed variants (replacing today's classes):
  | Variant | sensors | scene source | realism | Consumer |
  |---|---|---|---|---|
  | `_TeleopCapture` | `("rgb_full",)` | Infinigen (or contract-conformant) | Real | `capture.py --driver teleop` |
  | `_BridgeAutonomy` | `("rgb_full", "depth_full", "depth_policy")` | Infinigen | Real | bridge sim-in-the-loop |
  | `_RLDepth_Real` | `("depth_policy",)` | ProcRoom | Real | RL depth training |
  | `_RLDepth_Robust` | `("depth_policy",)` | ProcRoom | Robust | RL depth training (DR pass) |
  | `_RLNoCam` | `()` | None / minimal | Real | RL no-cam baseline |
  | `_Coverage` (future) | `("rgb_full", "depth_full")` | Infinigen | Real | scripted driver + coverage mission |
- [ ] Matching writer-side update: `lerobot_writer.build_features` takes the same `cameras_required` tuple and conditionally declares feature columns. `add_frame` validates args match the declared schema (no zero-padding for absent cameras — frames not captured aren't authored at all). This is the perf brief's §C deliverable, owned here.
- [ ] **Clean gym-ID scheme + complete caller migration (no aliases).** Register a small set of gym IDs for the named composed variants under a new composition-legible scheme; **do not** re-register the old names. Every caller below migrates to the new IDs in this same PR. Propose the new naming scheme in the reconciliation note (composition-legible, e.g. encodes sensor/scene/realism); orchestrator reviews it.
- [ ] **Obs/action contract preserved per composition (the snapshot gate).** For each old RL/teleop/bridge config, snapshot the observation-space shape + action-space + a deterministic-seed first observation **before** the refactor; assert the **new composition that replaces it** produces a byte-identical snapshot **after**. Names change; the contract does not. If any composition diverges, STOP and report — a divergent obs contract breaks the DEPTH inference package and any checkpoint.
- [ ] **Training-works smoke.** Launch a short run of `train_strafer_navigation.py` against the renamed RL-depth config and confirm it initializes + steps without error (a few iterations is enough). This is the operator-facing proof that "training works as expected." Autonomous (no gamepad); needs Isaac Sim + GPU.
- [ ] Scene source as parameter: passing `--scene-usd <path>` to `capture.py` (already the operator-facing override per `scene_paths.py`) bypasses the Infinigen-specific spawn-points pool when the USD conforms to `scene-provider-contract`. Lays the wiring so a foreign-source adapter only needs to ship the artifacts, not a new env subclass.
- [ ] Tests:
  - One per axis: a `SensorStackCfg` selecting one camera produces an env with only that camera registered.
  - Composition: a `_TeleopCapture` instance has the rgb_full camera, lacks the depth cameras, has Real DR active, loads from Infinigen scene-source.
  - Registration: `test_env_registration.py` updated to the new gym IDs and passing — no old ID resolves (the clean-break proof at the test layer).

### Full caller / reference inventory (must all migrate in this PR)

Grep-verified 2026-05-30. Every code reference to a gym ID moves to
the new scheme; every doc reference updates.

**Code (migrate — break if missed):**
- `source/strafer_lab/strafer_lab/tasks/navigation/__init__.py` (the registration table)
- `source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py`
- `source/strafer_lab/strafer_lab/sim_in_the_loop/runtime_env.py`
- `Scripts/train_strafer_navigation.py`, `Scripts/test_strafer_env.py`, `Scripts/play_strafer_navigation.py`, `Scripts/export_policy.py`
- `source/strafer_lab/scripts/collect_demos.py`, `source/strafer_lab/scripts/run_sim_in_the_loop.py`, `source/strafer_lab/scripts/teleop_capture.py`
- Tests: `source/strafer_lab/test/env/test_env_registration.py`, `source/strafer_lab/tests/test_export_policy.py`, `source/strafer_lab/test/sensors/test_d555_perception_cfg.py`

**User-facing docs (update):** `Readme.md`, `source/strafer_lab/README.md`, `docs/INTEGRATION_SIM_IN_THE_LOOP.md`, `docs/example_commands_cheatsheet.md`, `docs/SYSTEM_FLOW_DIAGRAMS.md`, `docs/STRAFER_AUTONOMY_NEXT.md`, `docs/DGX_SPARK_SETUP.md`.

**Active/parked task briefs (update the IDs they reference):** `subgoal-env.md`, `depth-subgoal-env.md`, `strafer-direct-sim-validation.md`, `strafer-hybrid-sim-validation.md`, `goal-noise-training.md`, `harness-throughput-measurement.md`, `rl-global-nav2-local.md`, `bridge-throughput-toward-25hz.md`.

**Completed briefs — do NOT edit** (historical record per `branching-and-prs.md`): `inference-package.md`, `trajectory-first-captioning.md`, `mid-mission-validation-investigation.md`. They describe what was true at ship time; leave them.

- [ ] **Doc sweep verified.** After migration, `grep -rl "Isaac-Strafer-Nav" --include="*.py" --include="*.md"` returns only the new-scheme IDs (in code + active docs) and the untouched completed briefs. No active code/doc references a retired ID.
- [ ] Migration note in `docs/INTEGRATION_SIM_IN_THE_LOOP.md` describing the new composition shape + the old→new gym-ID mapping table (so a returning operator can translate muscle-memory commands).

## Approach — clean cutover, snapshot-gated

The clean-break decision means there is **no deprecation window and no
shim layer**. The two-phase shape still helps as a *safety sequence*,
but the end state has zero legacy classes:

**Phase 1 — build the composition + prove the contract.** Introduce the
three axis-cfgs (`SensorStackCfg`, `SceneSourceCfg`, `RealismCfg`) and a
`_BaseStraferNavEnvCfg` that composes them (composition via attribute-typed
sub-cfgs, **not** diamond inheritance — see Risks). Re-express each old
config as a composition. Before touching anything, capture the obs/action
snapshot for every old config; after, assert the equivalent composition is
byte-identical. **This snapshot-identity proof is the gate.** If it can't be
made green, STOP — do not proceed to the cutover on a divergent contract.

**Phase 2 — cut over + delete.** Register the new gym IDs for the named
variants (`_TeleopCapture`, `_RLDepth_Real`, `_RLNoCam`, etc.), migrate
every caller in the inventory to the new IDs, **delete the 39 old
subclasses and their old gym-ID registrations** (no shims), update all
active docs + the migration note, run the training-works smoke. The clean
break lands whole — there is no intermediate state where both old and new
IDs resolve.

Because the cutover is atomic (old IDs gone, callers migrated, in one PR),
the gate is the obs/action snapshot identity + the full test suite +
`test_env_registration.py` green on the new IDs. If session budget can't
fit the whole cutover, it is acceptable to ship Phase 1 (composition built,
snapshot-proven, old hierarchy still present) and do the delete-and-migrate
cutover in an immediate follow-up — but the *brief does not ship* until the
old hierarchy is gone, per the clean-break decision.

## Out of scope

- The full `scene-provider-contract` deliverable (the artifact-shape doc + the last Infinigen-specific knob extraction). That brief is the sibling work; this brief is the consumer.
- The `scene-metadata-in-usd` storage-mode change. Whether scene metadata lives in a sidecar JSON or in USD `customData`, the env consumes it via the contract; storage choice doesn't affect this refactor.
- Adding new realism levels (e.g. ROBUST_OUTDOOR or similar). The `RealismCfg` axis is split out so future tiers can be added trivially, but enumerating them is out of scope.
- Re-architecting `MecanumWheelAction` — that's the [`mecanum-action-throughput`](../sim-performance/mecanum-action-throughput.md) brief's job.

## Risks

- **Obs/action contract divergence (the load-bearing risk under the clean break).** With names changing freely, the danger is no longer "an ID stops resolving" — it's a composed variant emitting a *subtly different* obs tensor (key ordering, DR seed sequence, noise term) than the config it replaces, which would silently break the DEPTH inference package and any checkpoint while every test still "passes." Mitigation: the snapshot-identity gate, keyed by old-config → new-composition equivalence, run before the cutover. This is the gate, not a nice-to-have.
- **A missed caller.** The clean break means a caller still pointing at a retired gym ID is a hard failure (training script won't launch, bridge won't start). Mitigation: the grep-verified inventory in acceptance + `test_env_registration.py` rewritten to the new IDs (a stale reference fails the test) + the doc-sweep grep assertion.
- **Mixin ordering in `@configclass`**: Python's MRO is well-defined but Isaac Lab's `@configclass` has historically had subtle interactions with multiple inheritance. Mitigation: composition (a `_BaseStraferNavEnvCfg` with attribute-typed sub-cfgs) rather than diamond inheritance.

## Triggered by

PR #63 review thread on [`teleop-perf-architecture.md`](../sim-performance/teleop-perf-architecture.md). Operator quote: *"Should we reframe _BaseInfinigenPerceptionNavEnvCfg as _BaseStraferNavEnvCfg? And then we can compose different versions for bridge mode (RGB + depth; Infinigen; 1 realness level), harness capture (Just RGB; Infinigen; 1 realness level), or even continuous control policy training (RGB + depth; ProcRoom USD or None; 3 realness levels). And for the case of teleop and bridge mode, this allows us to abstract away the USD which defines the environment the robot moves in."*

The perf brief's §C captured the camera-toggle slice of this idea but couldn't carry the scene-source-as-parameter or realism-axis split without scope creep. This brief carries all three.
