# Isaac Lab 3.0 Migration — Integration Test Plan

Manual validation checklist for verifying `strafer_lab` works end-to-end
after the rsl_rl 5.0 / Isaac Lab 3.0 migration. Unit tests (352/352) already
pass — this plan covers GPU-interactive scenarios the test harness cannot.

> **Environment:** DGX SPARK, `env_isaaclab3`, Isaac Sim 6.0.0, Isaac Lab v4.5.24  
> **Branch:** `feature/isaaclab-3.0-migration`

## Results Summary (2026-04-09)

| Test | Status | Notes |
|------|--------|-------|
| 1 — Asset physics | ✅ PASS | 36 MB USD output, exit 0 |
| 2a — NoCam env | ✅ PASS | All 8 patterns completed, reasonable physics |
| 2b — Depth env | ✅ PASS | Forward+Backward patterns, depth camera rendered |
| 2c — ProcRoom env | ✅ PASS | Required fix: `write_body_link_pose_to_sim_index()` → keyword args |
| 3 — NoCam training | ✅ PASS | 20/20 iterations, `model_19.pt` saved |
| 4a — DeFM training | ✅ PASS | 10/10 iterations, `model_9.pt` saved |
| 4b — CNN training | ✅ PASS | 10/10 iterations, `model_9.pt` saved |
| 5 — Checkpoint resume | ✅ PASS | Loaded model_9.pt, resumed from iter 9, trained successfully |
| 6 — Demo collection | ⏳ PENDING | Requires gamepad (manual) |
| 7 — DAPG | ⏳ PENDING | Requires demo HDF5 from Test 6 |
| 8 — GAIL | ⏳ PENDING | Requires demo HDF5 from Test 6 |
| 9 — LSTM runner | ✅ PASS | 10/10 iterations, `model_9.pt` saved |

### Bugs Found & Fixed

1. **`test_strafer_env.py` line 121:** `env_cfg.sim.physx.update_transformations_in_usd = True` — Isaac Lab 3.0 renamed `SimulationCfg.physx` → `SimulationCfg.physics` and removed `update_transformations_in_usd`. Fixed by removing the line.
2. **`proc_room.py` line 809:** `collection.write_body_link_pose_to_sim_index(poses, env_ids, all_object_ids)` — Isaac Lab 3.0 changed to keyword-only arguments. Fixed: `collection.write_body_link_pose_to_sim_index(body_poses=poses, env_ids=env_ids, body_ids=all_object_ids)`

---

## Prerequisites

```bash
cd ~/Documents/repos/IsaacLab
conda activate env_isaaclab3
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
```

All commands below use `./isaaclab.sh -p` (Linux) or `isaaclab.bat -p` (Windows).

---

## Test 1 — Asset Physics Setup (LOW risk)

**What it validates:** USD articulation, collision shapes, joints, and SDF
configuration still apply correctly. No migration code involved, but the
output USD is consumed by every environment.

```bash
./isaaclab.sh -p ~/Documents/repos/Sim2RealLab/Scripts/setup_physics.py \
  --stage ~/Documents/repos/Sim2RealLab/Assets/strafer/3209-0001-0006-no-physics.usd \
  --output-usd /tmp/strafer_physics_test.usd
```

**Pass criteria:**
- [ ] Exits 0, no Python errors
- [ ] Output USD exists and is non-empty
- [ ] (Optional) Inspect with `inspect_robot_prim_layout.py` — all joints and rigid bodies present

---

## Test 2 — Environment Smoke Test (MEDIUM risk)

**What it validates:** All environment variants load, reset, and step without
crashing. Observation shapes match expected dimensions.

### 2a — NoCam variant (proprioceptive only, 19-dim obs)

```bash
./isaaclab.sh -p ~/Documents/repos/Sim2RealLab/Scripts/test_strafer_env.py \
  --env Isaac-Strafer-Nav-Real-NoCam-v0 \
  --pattern all \
  --num_envs 4 \
  --headless
```

**Pass criteria:**
- [ ] All motion patterns complete without error
- [ ] Observation shape is `(4, 19)`

### 2b — Depth variant (depth camera, 4819-dim obs)

```bash
./isaaclab.sh -p ~/Documents/repos/Sim2RealLab/Scripts/test_strafer_env.py \
  --env Isaac-Strafer-Nav-Real-Depth-v0 \
  --pattern forward \
  --num_envs 2 \
  --headless
```

**Pass criteria:**
- [ ] Environment loads cameras and renders depth frames
- [ ] Observation shape is `(2, 4819)`
- [ ] No `wp.array` / `torch.device` errors in depth pipeline

### 2c — ProcRoom variant (procedural rooms)

```bash
./isaaclab.sh -p ~/Documents/repos/Sim2RealLab/Scripts/test_strafer_env.py \
  --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0 \
  --pattern forward \
  --num_envs 2 \
  --headless
```

**Pass criteria:**
- [ ] Procedural room generation works
- [ ] Depth observations populated correctly

---

## Test 3 — NoCam Training (MEDIUM risk)

**What it validates:** MLP policy training with Gaussian distribution —
the simplest pipeline, no depth encoder or Beta distribution involved.

```bash
./isaaclab.sh -p ~/Documents/repos/Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-NoCam-v0 \
  --num_envs 64 \
  --max_iterations 20 \
  --log_dir /tmp/test_nocam \
  --headless
```

**Pass criteria:**
- [ ] Training loop runs 20 iterations without error
- [ ] Mean reward is logged and increasing (or at least not NaN)
- [ ] Checkpoint saved to `/tmp/test_nocam/`
- [ ] `model_*.pt` file exists in log directory

---

## Test 4 — Depth Training with DeFM Encoder (HIGH risk)

**What it validates:** The full migrated pipeline —
`StraferDepthRNNModel` + `AffineBetaDistribution` + DeFM depth encoder +
GRU recurrence. This is the highest-risk test.

```bash
./isaaclab.sh -p ~/Documents/repos/Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-Depth-v0 \
  --num_envs 64 \
  --max_iterations 10 \
  --depth_encoder defm \
  --log_dir /tmp/test_depth_defm \
  --headless
```

**Pass criteria:**
- [ ] DeFM backbone loads from `torch.hub` (requires network or cached model)
- [ ] `StraferDepthRNNModel` initializes for both actor and critic
- [ ] `AffineBetaDistribution` produces actions in `[-1, 1]`
- [ ] Mean reward is logged (not NaN)
- [ ] Entropy is finite and positive
- [ ] GRU hidden states reset correctly on episode boundaries
- [ ] Checkpoint saved to `/tmp/test_depth_defm/`

### 4b — Depth Training with CNN Encoder (fallback)

```bash
./isaaclab.sh -p ~/Documents/repos/Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-Depth-v0 \
  --num_envs 64 \
  --max_iterations 10 \
  --depth_encoder cnn \
  --log_dir /tmp/test_depth_cnn \
  --headless
```

**Pass criteria:**
- [ ] CNN encoder initializes (no DeFM dependency)
- [ ] Training runs, actions bounded, reward logged
- [ ] Checkpoint saved

---

## Test 5 — Checkpoint Resume (HIGH risk)

**What it validates:** Loading a checkpoint produced by the migrated code.
Ensures model state dict keys, optimizer state, and RNN hidden state shapes
are all compatible.

```bash
# Use the checkpoint from Test 4
CKPT=$(ls /tmp/test_depth_defm/*/model_*.pt | tail -1)

./isaaclab.sh -p ~/Documents/repos/Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-Depth-v0 \
  --num_envs 64 \
  --max_iterations 5 \
  --depth_encoder defm \
  --resume "$CKPT" \
  --log_dir /tmp/test_resume \
  --headless
```

**Pass criteria:**
- [ ] `runner.load()` succeeds without state dict key mismatch
- [ ] Training continues from loaded iteration count
- [ ] No shape mismatch errors in actor/critic/distribution parameters
- [ ] Reward is in the same ballpark as where Test 4 left off

---

## Test 6 — Demo Collection (MEDIUM risk)

**What it validates:** Gamepad-driven demo recording produces valid HDF5
files with correct observation dimensions. Requires a gamepad connected.

```bash
./isaaclab.sh -p ~/Documents/repos/Sim2RealLab/source/strafer_lab/scripts/collect_demos.py \
  --task Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0 \
  --output /tmp/test_demos/ \
  --max_episodes 3 \
  --show_depth
```

**Pass criteria:**
- [ ] Environment loads in Play mode with depth rendering
- [ ] Gamepad input maps to robot actions
- [ ] HDF5 file written with `obs`, `actions`, `rewards` datasets
- [ ] `obs.shape[1]` matches depth env obs dim (4819)
- [ ] `actions.shape[1]` is 3

> **No gamepad?** You can verify the environment loads and the HDF5 writer
> initializes even without driving. Ctrl-C after a few seconds — partial
> episodes may still be saved.

---

## Test 7 — DAPG Training with Demos (HIGH risk)

**What it validates:** Demo Augmented Policy Gradient auxiliary loss
integrates with the migrated PPO loop. `ExpertDemoBuffer` validates obs_dim
against the policy, and the BC loss term is added correctly.

> **Prerequisite:** A demo file from Test 6 (or any HDF5 with matching obs dim).

```bash
./isaaclab.sh -p ~/Documents/repos/Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-Depth-v0 \
  --num_envs 64 \
  --max_iterations 10 \
  --depth_encoder defm \
  --aux dapg \
  --dapg_demos /tmp/test_demos/*.h5 \
  --log_dir /tmp/test_dapg \
  --headless
```

**Pass criteria:**
- [ ] `install_strafer_ppo()` patches PPO.update() without error
- [ ] `ExpertDemoBuffer` loads HDF5 and passes obs_dim validation
- [ ] DAPG loss term appears in training logs (non-zero)
- [ ] BC weight decays over iterations
- [ ] No crashes in the auxiliary loss computation

---

## Test 8 — GAIL Training with Demos (HIGHEST risk)

**What it validates:** GAIL discriminator auto-construction from
`ppo.policy.encoded_obs_dim`. This is the most fragile integration point —
the discriminator input dimension is lazily detected from the depth model.

> **Prerequisite:** A demo file from Test 6.

```bash
./isaaclab.sh -p ~/Documents/repos/Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-Depth-v0 \
  --num_envs 64 \
  --max_iterations 10 \
  --depth_encoder defm \
  --aux gail \
  --gail_demos /tmp/test_demos/*.h5 \
  --log_dir /tmp/test_gail \
  --headless
```

**Pass criteria:**
- [ ] `GAILAuxiliary` reads `encoded_obs_dim` from `StraferDepthRNNModel`
- [ ] Discriminator builds with correct input dim (`encoded_obs_dim + 3`)
- [ ] GAIL reward signal appears in logs
- [ ] Discriminator loss is finite and decreasing
- [ ] No dimension mismatches between policy encoded obs and discriminator input

---

## Test 9 — LSTM Runner (LOW risk)

**What it validates:** The LSTM (system-ID) runner config still works. Uses
stock `RslRlRNNModelCfg` (not migrated), but verifies the training script's
config selection logic is correct.

```bash
./isaaclab.sh -p ~/Documents/repos/Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-NoCam-v0 \
  --num_envs 64 \
  --max_iterations 10 \
  --log_dir /tmp/test_lstm \
  --headless
```

> Note: LSTM variant is selected automatically when env uses NoCam + the
> runner cfg matches. Check that the training script picks
> `STRAFER_PPO_RUNNER_CFG` (or `LSTM` if the env is wired that way).

**Pass criteria:**
- [ ] Training runs without error
- [ ] Correct runner config selected (check log output)

---

## Risk Summary

| Test | Component | Risk | Migration Dependency |
|------|-----------|------|----------------------|
| 1 | Asset physics | LOW | None |
| 2 | Env smoke test | MEDIUM | Obs pipeline, camera, quaternions |
| 3 | NoCam training | MEDIUM | Runner config, Gaussian dist |
| 4 | Depth training | HIGH | `StraferDepthRNNModel`, `AffineBetaDistribution`, depth encoder |
| 5 | Checkpoint resume | HIGH | State dict compatibility |
| 6 | Demo collection | MEDIUM | Obs dim contract |
| 7 | DAPG | HIGH | PPO patch, obs_dim validation, BC loss |
| 8 | GAIL | HIGHEST | `encoded_obs_dim` auto-detection, discriminator build |
| 9 | LSTM runner | LOW | Config selection logic |

---

## Known Risks & Follow-up Items

### Risks

1. **`encoded_obs_dim` exposure (GAIL):** `GAILAuxiliary` lazily reads
   `ppo.policy.encoded_obs_dim` to build the discriminator. If
   `StraferDepthRNNModel` doesn't set this attribute correctly, GAIL will
   crash at `on_update_start()`. Verify this attribute exists and has the
   expected value (scalar_obs_dim + depth_embedding_dim).

2. **DeFM `torch.hub` network dependency:** First run downloads the
   pretrained backbone from GitHub. If the DGX SPARK has no internet or
   the LeggedRobotics repo changes its URL, DeFM loading will fail. Consider
   pre-caching the model or verifying the CNN fallback works.

3. **Checkpoint format stability:** Checkpoints saved by the migrated code
   have different state dict keys than pre-migration `StraferActorCritic`
   checkpoints. Old checkpoints are incompatible (this is expected and
   accepted — no existing checkpoints to preserve).

4. **Beta distribution entropy scale:** `AffineBetaDistribution` entropy
   differs from Gaussian. The tuned `entropy_coef=0.005` was calibrated
   for the old `AffineBeta` — verify the new implementation produces
   similar entropy magnitudes (should be identical, but worth checking).

5. **RNN hidden state reset timing:** `StraferDepthRNNModel` must reset
   hidden states when environments reset. The reset mask
   (`dones` from the rollout buffer) propagation through rsl_rl's
   `RNNModel.forward()` should handle this, but verify no stale hidden
   states leak across episode boundaries.

6. **Obs group routing (`resolve_obs_groups`):** rsl_rl 5.0 uses
   `resolve_obs_groups()` to map `"actor" → ["policy"]`. Ensure the
   env's obs_groups (`{"policy": ..., "critic": ...}`) are correctly
   consumed by the separate actor/critic models.

### Follow-up Items

1. **TODO #3 from DGX_SPARK_SETUP.md:** Replace `AppLauncher(headless=True,
   enable_cameras=True)` with `--viz none` when Isaac Lab 3.0 reaches
   stable release.

2. **Long training run validation:** Run a full 3000+ iteration NoCam
   training and 10000 iteration depth training to verify convergence
   matches pre-migration baselines.

3. **Sim-to-real deployment check:** If deploying to physical Strafer,
   verify the exported ONNX/TorchScript model from the new architecture
   matches the expected input/output contract.
