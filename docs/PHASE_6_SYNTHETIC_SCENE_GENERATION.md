# Phase 6: Synthetic Scene Generation for Generalist Obstacle Avoidance

Build a practical synthetic scene pipeline for Isaac Sim/Isaac Lab that increases obstacle diversity (shape, size, clutter, dynamics) and improves sim-to-real transfer for the Strafer depth navigation policy.

This phase is designed to plug into the current repository and training flow with minimal disruption:
- Isaac Lab training remains in `Scripts/train_strafer_navigation.py`.
- Navigation task code remains in `source/strafer_lab/strafer_lab/tasks/navigation/`.
- New scene generation tooling is added under `Scripts/scenegen/`.

## Overview

Current navigation training uses fixed obstacle primitives with randomized positions. Phase 6 adds a scene generator that combines:
1. **Infinigen-based procedural scene synthesis** (for broad structural diversity)
2. **Omniverse Replicator randomization** (for controlled variation and labeling)
3. **Downloaded OpenUSD asset packs** in this repo:
   - `Assets/Residential_NVD@10012`
   - `Assets/SimReady_Furniture_Misc_01_NVD@10010`

The output is a curated library of train/eval USD scenes and manifests that Isaac Lab can consume deterministically or stochastically during training.

## 1. Design Goals and Constraints

### 1.1 Goals

1. Increase geometric diversity without hand-authoring every obstacle type.
2. Produce repeatable scene sets for train/val/test split and regression testing.
3. Support both static clutter and dynamic obstacle behaviors.
4. Preserve current sim-to-real observation/action contract.
5. Keep runtime cost bounded so depth training remains practical on current hardware.

### 1.2 Non-goals (for this phase)

1. No full semantic navigation stack rewrite.
2. No dependence on VLM inference during low-level RL training.
3. No immediate requirement to modify ROS deployment packages.

### 1.3 Practical Constraints from Current Stack

| Constraint | Current Value | Planning Impact |
|-----------|---------------|-----------------|
| Depth input | 80x60 (4800 dims) | Scene complexity must prioritize geometry over texture detail. |
| Control rate | 30 Hz (decimation from 120 Hz sim) | Dynamic obstacles must move smoothly at this control rate. |
| Depth training env count | ~32 practical | Synthetic scenes must avoid excessive per-env memory usage. |
| NoCam training env count | 512-4096 | Keep NoCam pretraining path for rapid policy iteration. |
| Policy contract | `strafer_shared.policy_interface` | Observation ordering/scales remain unchanged. |

## 2. Inputs and Asset Sources

### 2.1 Infinigen + Replicator

Use the Infinigen + Replicator workflow as the procedural backbone for indoor scene variation, with Replicator randomizers controlling object placement, lighting, and camera perturbations.

### 2.2 Local OpenUSD Asset Packs (already downloaded)

| Pack | Local Path | Primary Use |
|------|------------|-------------|
| Residential | `Assets/Residential_NVD@10012` | Household clutter, appliances, decor, plants |
| SimReady Furniture Misc | `Assets/SimReady_Furniture_Misc_01_NVD@10010` | Furniture and medium/large indoor obstacles |

### 2.3 Asset Quality Gate (must pass before use)

Each candidate asset in the scene pool must satisfy:
1. USD loads without missing references.
2. Reasonable bounding box (non-zero extents, no extreme scale).
3. Collider can be generated or approximated.
4. Asset can be placed stably on floor plane.
5. License metadata is retained in manifest.

## 3. Target Architecture

### 3.1 Offline Scene Synthesis Pipeline

```
Asset Sources
  |- Infinigen procedural layouts
  |- Local OpenUSD packs (Residential + SimReady)
         |
         v
Scene Composer (Replicator)
  |- room/layout sampling
  |- obstacle placement
  |- domain randomization
  |- traversability checks
         |
         v
Scene Library Export
  |- USD scene files
  |- scene_manifest.jsonl
  |- train/val/test split manifests
```

### 3.2 Online Isaac Lab Training Integration

```
Isaac Lab env reset
  -> sample scene_id from split manifest
  -> load or reference scene prototype
  -> apply per-episode randomization (positions, dynamics, noise)
  -> run RL loop unchanged
```

## 4. Scene Specification (Concrete)

### 4.1 Scene Classes and Split

Generate an initial **200-scene library**:
- 120 train scenes
- 40 val scenes
- 40 test scenes (never used in training)

Class distribution:
1. Open rooms: 20%
2. Hallway-like constrained layouts: 20%
3. Dense furniture layouts: 40%
4. Mixed clutter + narrow passages: 20%

### 4.2 Obstacle Taxonomy and Sampling Ratios

Per scene, sample obstacle instances from:
1. Primitive parametric obstacles (box/cylinder/capsule): 30%
2. SimReady furniture assets: 40%
3. Residential decor/appliances/plants: 30%

Per-scene counts:
1. Static obstacles: 8-25
2. Dynamic obstacles: 0-6 (curriculum controlled)

### 4.3 Geometric and Physical Randomization Ranges

| Parameter | Range |
|-----------|-------|
| Uniform scale multiplier | 0.6-1.4 |
| Yaw rotation | -180 to +180 deg |
| Mass multiplier | 0.8-1.2 |
| Static friction | 0.4-1.2 |
| Dynamic friction | 0.3-1.0 |
| Restitution | 0.0-0.2 |

### 4.4 Dynamic Obstacle Behavior Set

Use a fixed behavior menu for reproducibility:
1. `crossing_linear`: constant velocity crossings, 0.2-0.8 m/s.
2. `stop_and_go`: alternating move/pause windows (1-3 s).
3. `patrol_loop`: waypoint loops around room perimeter.
4. `drift_noise`: low-speed random-walk perturbations.

Dynamic obstacle ratio over training curriculum:
1. Stage A: 0%
2. Stage B: 10%
3. Stage C: 20%

### 4.5 Traversability Constraints

Reject scene samples that violate:
1. Minimum free corridor width < 0.55 m.
2. Robot spawn or goal collides with geometry.
3. No feasible path from spawn to goal in coarse grid check.

### 4.6 Sensor/Rendering Randomization

For depth-focused policy robustness:
1. Light intensity: 300-3000.
2. Color temperature perturbation: warm-neutral-cool bins.
3. Camera pose perturbation: +/-2 deg roll/pitch/yaw, +/-2 cm translation.
4. Depth corruptions remain governed by existing noise models in `mdp/noise_models.py`.

## 5. Planned Repository Changes

### 5.1 New Scripts (scene generation)

Add under `Scripts/scenegen/`:
1. `build_asset_manifest.py`
2. `validate_asset_pool.py`
3. `compose_scenes_replicator.py`
4. `split_scene_dataset.py`
5. `export_scene_stats.py`

### 5.2 New Data/Manifest Paths

Add under `Assets/generated/`:
1. `Assets/generated/scenes/train/*.usd`
2. `Assets/generated/scenes/val/*.usd`
3. `Assets/generated/scenes/test/*.usd`
4. `Assets/generated/manifests/asset_manifest.jsonl`
5. `Assets/generated/manifests/scene_manifest.jsonl`
6. `Assets/generated/manifests/splits/{train,val,test}.txt`

### 5.3 Isaac Lab Touchpoints

Planned integration points:
1. `source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py`
   - add scene-backed env variants (proc scene train/play).
2. `source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py`
   - add dynamic obstacle behavior update event terms.
3. `source/strafer_lab/strafer_lab/tasks/navigation/__init__.py`
   - register new env IDs.
4. `source/strafer_lab/test/`
   - add scene manifest and registration tests.

## 6. Environment Variants to Add

Add three depth-focused variants:
1. `Isaac-Strafer-Nav-Real-ProcDepth-v0`
2. `Isaac-Strafer-Nav-Robust-ProcDepth-v0`
3. `Isaac-Strafer-Nav-Real-ProcDepth-Play-v0`

Guidelines:
1. Keep observation/action contract identical to existing DEPTH variant.
2. Start with 16-32 envs for proc-scene runs.
3. Preserve NoCam variants for fast baseline iteration.

## 7. Training Curriculum (Achievable)

### 7.1 Four-Stage Curriculum

| Stage | Env | Iterations | Purpose |
|------|-----|------------|---------|
| S0 | Real-NoCam | 600-1000 | Fast locomotion and goal-tracking baseline |
| S1 | Real-Depth | 300-600 | Depth encoder stabilization in simple scenes |
| S2 | Real-ProcDepth (static) | 400-800 | Geometry diversity adaptation |
| S3 | Robust-ProcDepth (dynamic) | 400-800 | Dynamic obstacle robustness |

### 7.2 Evaluation Protocol

Evaluate checkpoints on:
1. In-distribution train scenes.
2. Held-out synthetic test scenes.
3. Real-world clutter courses (same metric definitions).

Track:
1. Goal success rate.
2. Collision rate per meter.
3. Time-to-goal.
4. Recovery behavior after near-collision.

## 8. Success Criteria

Phase 6 is complete when all are met:
1. 200-scene library generated and versioned with manifests.
2. New proc-scene env variants train end-to-end without manual intervention.
3. Held-out synthetic test success improves by >= 15% over box-only baseline.
4. Real-world collision rate decreases by >= 25% at matched route set.
5. Regression tests pass for env registration and observation contract.

## 9. Implementation Roadmap

### 9.1 Phase 6a: Pipeline MVP (Week 1)

| Step | Task | Duration |
|------|------|----------|
| 6a.1 | Build asset manifest from local packs + quality gate | 0.5-1 day |
| 6a.2 | Implement Replicator scene composer (static clutter only) | 2-3 days |
| 6a.3 | Generate first 50 scenes + split manifests | 1 day |
| 6a.4 | Add proc-scene env variant and smoke test in Isaac Lab | 1-2 days |

**Exit criterion**: train one depth run on generated scenes for >= 100 iterations without crashes.

### 9.2 Phase 6b: Dynamic Scene Expansion (Week 2)

| Step | Task | Duration |
|------|------|----------|
| 6b.1 | Add dynamic obstacle behavior library | 1-2 days |
| 6b.2 | Add traversability/path feasibility filter | 1 day |
| 6b.3 | Expand scene library from 50 -> 200 scenes | 1-2 days |
| 6b.4 | Add unit/integration tests for manifests and env registration | 1 day |

**Exit criterion**: dynamic-proc training runs for >= 300 iterations with stable reward trend.

### 9.3 Phase 6c: Sim-to-Real Validation Sprint (Week 3)

| Step | Task | Duration |
|------|------|----------|
| 6c.1 | Baseline vs proc-scene ablation training | 2-3 days |
| 6c.2 | Held-out synthetic evaluation and failure analysis | 1 day |
| 6c.3 | Real robot route set validation | 1-2 days |
| 6c.4 | Finalize Phase 6 report in `docs/` | 0.5 day |

**Exit criterion**: meet Phase 6 success metrics above.

## 10. First-Week Execution Checklist

- [ ] Create `Scripts/scenegen/` scaffolding and CLI stubs.
- [ ] Generate `asset_manifest.jsonl` from both local asset packs.
- [ ] Validate 100 random assets for loadability + collider creation.
- [ ] Generate first 50 static scenes.
- [ ] Add `Isaac-Strafer-Nav-Real-ProcDepth-v0` registration.
- [ ] Run short PPO smoke training (`max_iterations=100`) and verify no env crashes.

## 11. Planned Commands (for this phase)

```powershell
# 1) Build manifest from local packs
python Scripts/scenegen/build_asset_manifest.py `
  --assets_root Assets `
  --out Assets/generated/manifests/asset_manifest.jsonl

# 2) Generate scenes with Replicator + Infinigen templates
python Scripts/scenegen/compose_scenes_replicator.py `
  --asset_manifest Assets/generated/manifests/asset_manifest.jsonl `
  --out_dir Assets/generated/scenes `
  --num_scenes 200

# 3) Split scene dataset
python Scripts/scenegen/split_scene_dataset.py `
  --scene_manifest Assets/generated/manifests/scene_manifest.jsonl `
  --train 120 --val 40 --test 40

# 4) Train on proc-scene env variant
cd IsaacLab
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py `
  --env Isaac-Strafer-Nav-Real-ProcDepth-v0 `
  --num_envs 24 --headless
```

## 12. References

- Infinigen + Replicator tutorial:
  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_infinigen_sdg.html
- Downloadable OpenUSD asset packs:
  https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html
- Existing project context:
  `docs/SIM_TO_REAL_PLAN.md`
- Existing VLM planning style reference:
  `docs/PHASE_5_VLM_INTEGRATION.md`
