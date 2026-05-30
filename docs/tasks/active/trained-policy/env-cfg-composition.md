# Composable env_cfg axes — split sensor / scene / realism

**Type:** refactor (env_cfg tree)
**Owner:** DGX agent
**Priority:** P2 — cleans up an axis-conflation that's currently forcing every consumer (teleop, bridge, RL training, future scripted captioner / coverage) to take ALL camera outputs and the full Infinigen scene-loading stack even when they don't need them; also the boundary that makes "USD becomes a parameter" possible.
**Estimate:** M (incremental refactor — keep the existing variants working through a deprecation window; the actual axis split is mechanical).
**Branch:** `task/env-cfg-composition`
**Recommended ordering, not a hard block:** Land [`scene-provider-contract`](../harness/scene-provider-contract.md) first so this brief can commit to the contract's storage-agnostic artifact shape without retroactive design drift. The two can ship in parallel if needed — neither blocks the other at the code level — but minimal rework is contract-first.
**Subsumes:** the **per-env-variant camera toggle** deliverable in [`teleop-perf-architecture.md` §C](../sim-performance/teleop-perf-architecture.md#c-per-env-variant-camera-configuration-design). That item is the narrowest version of the composition this brief proposes; doing it twice would be rework. The perf brief is amended in the same PR that ships this brief's first commit to defer that item here.

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

- [ ] Rename `_BaseInfinigenPerceptionNavEnvCfg` → `_BaseStraferNavEnvCfg` (or similar — the Infinigen + Perception labels move into the composition dimensions, not the class name).
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
- [ ] All existing Gym IDs continue to resolve. Today's named entry points (e.g. `Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0`) get registered as the composed variants under the same names so no downstream consumer needs to update import paths in the same PR.
- [ ] Scene source as parameter: passing `--scene-usd <path>` to `capture.py` (already the operator-facing override per `scene_paths.py`) bypasses the Infinigen-specific spawn-points pool when the USD conforms to `scene-provider-contract`. Lays the wiring so a foreign-source adapter only needs to ship the artifacts, not a new env subclass.
- [ ] Tests:
  - One per axis: a `SensorStackCfg` selecting one camera produces an env with only that camera registered.
  - Composition: a `_TeleopCapture` instance has the rgb_full camera, lacks the depth cameras, has Real DR active, loads from Infinigen scene-source.
  - Backward compat: the legacy Gym IDs round-trip the env construction.
- [ ] Migration note in `docs/INTEGRATION_SIM_IN_THE_LOOP.md` describing the new composition shape + the legacy-ID guarantee.

## Approach — incremental, two-phase

**Phase 1**: Introduce the three mixin cfgs alongside the existing class hierarchy. Re-implement `_BaseInfinigenPerceptionNavEnvCfg` as the composition that produces the same shape; verify no consumer breakage. The legacy class becomes a thin shim that calls the composition.

**Phase 2**: Add the trimmed variants (`_TeleopCapture`, `_RLDepth_Real`, etc.) so consumers can opt into the narrower configs. Update `capture.py` to use `_TeleopCapture` by default. Update the bridge driver to use `_BridgeAutonomy`. Delete the now-unused legacy class after a deprecation window.

The legacy Gym ID re-registration is the gate — until that's solid, no consumer needs to change anything.

## Out of scope

- The full `scene-provider-contract` deliverable (the artifact-shape doc + the last Infinigen-specific knob extraction). That brief is the sibling work; this brief is the consumer.
- The `scene-metadata-in-usd` storage-mode change. Whether scene metadata lives in a sidecar JSON or in USD `customData`, the env consumes it via the contract; storage choice doesn't affect this refactor.
- Adding new realism levels (e.g. ROBUST_OUTDOOR or similar). The `RealismCfg` axis is split out so future tiers can be added trivially, but enumerating them is out of scope.
- Re-architecting `MecanumWheelAction` — that's the [`mecanum-action-throughput`](../sim-performance/mecanum-action-throughput.md) brief's job.

## Risks

- **Gym ID re-registration regression**: if a composed variant produces a subtly different env (e.g. different observation key ordering, different DR seed sequence), a trained checkpoint could behave differently when re-loaded. Mitigation: snapshot one episode's observation tensor from a stock checkpoint under each existing Gym ID before refactor; assert byte-identical after refactor.
- **Mixin ordering matters in `@configclass`**: Python's MRO is well-defined but Isaac Lab's `@configclass` decorator has historically had subtle interactions with multiple inheritance. Mitigation: use composition (a `_BaseStraferNavEnvCfg` with attribute-typed sub-cfgs) rather than diamond inheritance.

## Triggered by

PR #63 review thread on [`teleop-perf-architecture.md`](../sim-performance/teleop-perf-architecture.md). Operator quote: *"Should we reframe _BaseInfinigenPerceptionNavEnvCfg as _BaseStraferNavEnvCfg? And then we can compose different versions for bridge mode (RGB + depth; Infinigen; 1 realness level), harness capture (Just RGB; Infinigen; 1 realness level), or even continuous control policy training (RGB + depth; ProcRoom USD or None; 3 realness levels). And for the case of teleop and bridge mode, this allows us to abstract away the USD which defines the environment the robot moves in."*

The perf brief's §C captured the camera-toggle slice of this idea but couldn't carry the scene-source-as-parameter or realism-axis split without scope creep. This brief carries all three.
