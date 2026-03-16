# Training Improvement Plan

**Date**: 2026-03-15
**Context**: After running `Isaac-Strafer-Nav-Robust-ProcRoom-Depth-v0` with 128 envs for ~4000 iterations, the policy failed to learn meaningful navigation. TensorBoard showed no `goal_reached` signal, `action_smoothness` dominating at -20 episode reward, and collision penalties overwhelming goal-seeking signals.

**Root cause**: Catastrophic reward scale imbalance + stacking three difficulty multipliers (Robust noise + ProcRoom obstacles + CNN from scratch) simultaneously.

---

## Table of Contents

1. [Diagnosis](#1-diagnosis)
2. [Reward Modeling](#2-reward-modeling)
3. [Network Architecture — Pretrained Depth Encoder](#3-network-architecture--pretrained-depth-encoder)
4. [PPO Hyperparameters](#4-ppo-hyperparameters)
5. [Curriculum Tuning](#5-curriculum-tuning)
6. [Noise & Domain Randomization](#6-noise--domain-randomization)
7. [Code Correctness](#7-code-correctness)
8. [Training Plan](#8-training-plan)

---

## 1. Diagnosis

### Reward Scale Analysis (Episode Totals from TensorBoard)

| Signal | Episode Total | Relative Magnitude |
|--------|-------------:|-------------------:|
| `action_smoothness` | **-20** | **1000x dominant** |
| `collision` | -0.1 to -0.4 | 10–40x |
| `energy_penalty` | -0.05 | 5x |
| `heading_alignment` | ~0.02 | 2x |
| `goal_proximity` | ~0.01 | 1x (baseline) |
| `goal_progress` | ~0.006 | 0.6x |
| `goal_reached` | ~0.003 | 0.3x |
| `speed_near_goal` | ~-0.003 | 0.3x |

**The policy is learning to minimize action jerk, not to reach goals.** `action_smoothness` is 1000x larger than `goal_reached`. The agent's optimal strategy is "don't move" — which eliminates action jerk, collision, and energy penalties simultaneously.

### Compounding Difficulty

Three difficulty multipliers were stacked for the first training run:

1. **Robust noise** — 2.5x encoder noise, 2x depth disparity noise, sensor failures, 60ms motor lag, 1–5 step command delay
2. **ProcRoom obstacles** — procedural walls/furniture/clutter the robot must navigate around
3. **Depth CNN from scratch** — 80K-param CNN training from random initialization alongside PPO

Any one of these is fine for a policy that already works. All three simultaneously make learning nearly impossible.

---

## 2. Reward Modeling

### 2.1 Rebalance Reward Weights

The core fix: **make goal-seeking signals 5–10x larger and regularization signals 10x smaller**.

#### Current vs Proposed Weights

| Reward Term | Current Weight | Proposed Weight | Rationale |
|-------------|------:|------:|-----------|
| `goal_progress` | 2.0 | **10.0** | Primary dense signal — must dominate |
| `goal_proximity` | 1.5 | **5.0** | Continuous gradient to goal |
| `goal_reached` | 10.0 | **50.0** | Sparse but must be unmistakable |
| `heading_alignment` | 0.05 | **0.1** | Slight increase, still secondary |
| `collision` | -5.0 | **-2.0** | Strong enough to discourage, not paralyze |
| `collision_sustained` | -2.0 | **-1.0** | Reduced — termination handles sustained cases |
| `speed_near_goal` | -0.3 | **-0.1** | Gentler braking encouragement |
| `energy_penalty` | -0.01 | **-0.001** | 10x reduction — tiny guardrail |
| `action_smoothness` | -0.05 | **-0.005** | **10x reduction — THE critical fix** |

#### Additional: `goal_proximity` sigma

Change sigma from 0.3 to **0.5**. At sigma=0.3, `exp(-3.0/0.3) = 4.5e-5` — essentially zero signal at 3m. At sigma=0.5, `exp(-3.0/0.5) = 0.0025` — 50x more signal at typical goal distances.

```python
goal_proximity = RewTerm(
    func=mdp.goal_proximity_reward, weight=5.0,
    params={"command_name": "goal_command", "sigma": 0.5},  # was 0.3
)
```

### 2.2 Add Sustained Collision Termination

Currently, the only terminations are time-out and robot-flipped. A robot stuck against a wall accumulates 20 seconds of collision penalties without learning anything useful.

Add to `terminations.py`:

```python
def sustained_collision(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    max_steps: int = 30,  # ~1 second at 30Hz
) -> torch.Tensor:
    """Terminate if robot in continuous collision for max_steps."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w
    force_mag = torch.norm(net_forces, dim=-1)
    has_collision = (force_mag > threshold).any(dim=-1)

    if not hasattr(env, "_collision_step_count"):
        env._collision_step_count = torch.zeros(env.num_envs, device=env.device)

    env._collision_step_count = torch.where(
        has_collision,
        env._collision_step_count + 1,
        torch.zeros_like(env._collision_step_count),
    )

    reset_mask = env.episode_length_buf == 0
    env._collision_step_count[reset_mask] = 0

    return env._collision_step_count >= max_steps
```

Add to `TerminationsCfg`:

```python
sustained_collision = DoneTerm(
    func=mdp.sustained_collision,
    params={
        "sensor_cfg": SceneEntityCfg("contact_sensor"),
        "threshold": 1.0,
        "max_steps": 30,
    },
)
```

This forces faster episode turnover when the robot is stuck, generating more diverse experience.

### 2.3 Remove `alive_bonus`

Already noted as useless in its own docstring — provides no gradient in continuous multi-goal episodes.

---

## 3. Network Architecture — Pretrained Depth Encoder

### 3.1 Problem: CNN Training from Scratch + PPO

Training an 80K-param CNN from random initialization alongside PPO is a well-known failure mode. The value function depends on CNN features, but CNN features are garbage for the first 1000+ iterations. Neither the policy nor the visual representation can stabilize.

### 3.2 Solution: DeFM (Depth Foundation Model)

[DeFM](https://github.com/leggedrobotics/defm) is a depth-specific vision foundation model from ETH Zurich's Robotic Systems Lab (Jan 2026). It is purpose-built for this exact use case.

**Key properties:**

- Trained on **60M depth images** via DINOv2-style self-distillation
- Provides **metric-aware representations** that work across sim-to-real without fine-tuning
- **11 model variants** from 3M to 307M parameters (ViT-S/L, ResNet, EfficientNet, RegNet)
- Frozen features are **directly usable for RL** — no task-specific fine-tuning needed
- Proven on Habitat Point-Goal Navigation (directly relevant to our task)

**Recommended variant: DeFM-EfficientNet-B0**

| Property | Value |
|----------|-------|
| Parameters | ~3M |
| Jetson Orin latency | **3.01 ms** |
| RTX 4090 latency | <1 ms |
| Input resolution | 224×224 (we resize 80×60 → 224×224) |
| Output embedding | 1280-dim (project to 128 with linear layer) |

This replaces the custom `DepthEncoder` CNN entirely with a frozen pretrained backbone that already understands depth geometry.

**Alternative: DeFM-EfficientNet-B2** (5ms on Orin, better features, still real-time).

### 3.3 Integration Plan

Replace `DepthEncoder` in `strafer_network.py`:

```python
class DeFMDepthEncoder(nn.Module):
    """Frozen DeFM encoder + trainable projection head."""

    def __init__(self, output_dim: int = 128, model_name: str = "efficientnet_b0"):
        super().__init__()
        # Load pretrained DeFM encoder (frozen)
        self.backbone = defm.create_model(model_name, pretrained=True)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Trainable projection: backbone_dim → output_dim
        backbone_dim = self.backbone.num_features  # e.g. 1280 for EfficientNet-B0
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.ELU(),
        )
        # Resize transform for input normalization
        self._resize = torchvision.transforms.Resize(
            (224, 224), antialias=True,
        )

    def forward(self, depth_flat: torch.Tensor) -> torch.Tensor:
        # Reshape: (B, 4800) → (B, 1, 60, 80)
        x = depth_flat.view(-1, 1, 60, 80)
        # DeFM expects 3-channel input (log-compressed)
        x = self._resize(x)
        x = defm.normalize_depth(x)  # DeFM's 3-channel log normalization
        # Frozen forward pass
        with torch.no_grad():
            features = self.backbone(x)  # (B, backbone_dim)
        # Trainable projection
        return self.projection(features)  # (B, output_dim)
```

**Benefits:**
- CNN features are meaningful from step 0 — PPO immediately gets useful spatial information
- Only the 128-dim projection layer trains (~164K weights) — dramatically fewer gradients to manage
- Sim-to-real transfer is built into DeFM's pretraining (trained on sim + real depth data)
- Jetson Orin deployment: 3ms (EfficientNet-B0) fits comfortably in the 33ms control loop

### 3.4 Weight Transfer Strategy (NoCam → Depth)

If starting with a NoCam pretrained policy (obstacle-free), the MLP actor/critic weights can be partially loaded:

1. Train NoCam policy → saves `actor` and `critic` MLPs with `scalar_obs_dim` input
2. For Depth policy, the MLP input is `scalar_obs_dim + depth_embedding_dim`
3. Load strategy:
   - **Cannot directly load MLP weights** — input dimension changed
   - **Can load all non-MLP weights** (std parameter, normalization stats)
   - The DeFM frozen encoder needs no loading (pretrained)
   - MLP trains from scratch but with a **meaningful depth embedding from step 0**

This is why DeFM is strictly better than the "train NoCam first, transfer" approach — the depth features are already good, so the MLP learns quickly even from random init.

### 3.5 Empirical Normalization — Saving for Deployment

When `empirical_normalization=True`, RSL-RL's `EmpiricalNormalization` module maintains running mean/std statistics. These are saved as part of the model checkpoint (`model_*.pt`) automatically by RSL-RL. When loading for inference (export to JIT/ONNX), include the normalization layers in the traced graph. The `strafer_shared.policy_interface.load_policy()` function should handle this during export.

**Important**: When resuming or fine-tuning, normalization stats continue updating. If the obs distribution changes (e.g., NoCam → Depth), the stats adapt within a few hundred iterations.

---

## 4. PPO Hyperparameters

### 4.1 Depth Config Updates

Changes to `STRAFER_PPO_DEPTH_RUNNER_CFG` in `rsl_rl_ppo_cfg.py`:

```python
STRAFER_PPO_DEPTH_RUNNER_CFG = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=48,           # was 96 — shorter rollouts, faster updates
    max_iterations=10000,
    save_interval=100,
    empirical_normalization=True,   # *** was False — CRITICAL for mixed obs spaces ***
    obs_groups={"policy": ["policy"], "critic": ["critic"]},
    policy=RslRlPpoActorCriticCfg(
        class_name="StraferActorCritic",
        init_noise_std=0.3,         # was 0.5 — less exploration for complex obs
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,        # was 1.0 — stabilize early training
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,         # was 0.01 — less entropy for high-dim obs
        num_learning_epochs=5,      # was 8 — prevent overfitting per rollout
        num_mini_batches=4,
        learning_rate=3.0e-4,       # was 1.0e-4 — faster learning
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,            # was 0.008 — slightly more permissive
        max_grad_norm=0.5,          # was 1.0 — tighter clip for CNN stability
    ),
)
```

### 4.2 Rationale

| Parameter | Old → New | Why |
|-----------|-----------|-----|
| `empirical_normalization` | False → **True** | Obs space mixes tiny scalars with 4800 depth pixels. Without normalization, first MLP layers are dominated by depth contribution. **This is likely a silent killer.** |
| `num_steps_per_env` | 96 → **48** | 96 steps = 3.2s of experience per rollout. In ProcRoom, the robot barely reaches one goal. Shorter rollouts = more frequent gradient updates. |
| `init_noise_std` | 0.5 → **0.3** | 0.5 is too exploratory with obstacles — robot collides immediately and learns nothing. |
| `learning_rate` | 1e-4 → **3e-4** | With a frozen DeFM encoder, fewer trainable params → can use higher LR. |
| `max_grad_norm` | 1.0 → **0.5** | Even with frozen DeFM, the projection head gradients can spike. Tighter clipping prevents catastrophic updates. |
| `value_loss_coef` | 1.0 → **0.5** | Reduce critic influence while value estimates are still noisy from reward rebalancing. |
| `entropy_coef` | 0.01 → **0.005** | With DeFM features, the policy needs less random exploration — the features are informative. |

---

## 5. Curriculum Tuning

### 5.1 ProcRoom Goal Distance — Start Closer

```python
goal_distance = CurrTerm(
    func=mdp.GoalDistanceCurriculum,
    params={
        "initial_range": 1.5,       # was 2.0 — closer initial goals
        "max_range": 4.0,           # was 5.0 — rooms are ~7m, 5m is excessive
        "step_size": 0.5,
        "success_threshold": 3,     # was 5 — promote faster
        "goal_threshold": 0.3,
    },
)
```

### 5.2 Room Complexity — Promote Faster

```python
room_complexity = CurrTerm(
    func=mdp.RoomComplexityCurriculum,
    params={
        "initial_level": 0,
        "max_level": 5,
        "success_threshold": 5,     # was 10 — promote faster
        "goal_threshold": 0.3,
    },
)
```

### 5.3 Rationale

- Starting with `initial_range=1.5` and `initial_level=0` (empty room) lets the robot learn basic "go to point" before encountering obstacles.
- Lowering `success_threshold` from 10→5 (room) and 5→3 (goal distance) means the robot sees increasing difficulty sooner, preventing overfitting to easy scenarios.
- `max_range=4.0` instead of 5.0 — ProcRoom generates ~7m box rooms. A 5m goal in a 7m room often places the goal behind multiple obstacles, requiring long detours. 4m is challenging but achievable.

---

## 6. Noise & Domain Randomization

### 6.1 Start with Realistic, Not Robust

The Robust contract is designed for **stress-testing a policy that already works**. For initial training, use Realistic:

| Parameter | Realistic | Robust | Impact |
|-----------|-----------|--------|--------|
| Motor time constant | 50ms | 60ms | Sluggish response |
| Command delay | 1–3 steps | 1–5 steps | 166ms max latency |
| Encoder noise (σ) | 0.02 | 0.05 | 2.5x noise |
| Depth disparity noise | 0.08 px | 0.16 px | 2x noise |
| Depth hole probability | 1% | 3% | 3x more holes |
| Sensor failures | Disabled | Enabled | Random zero-outs |
| Goal position noise (σ) | 0.15m | 0.35m | Misplaced goals |

**The Robust goal noise (0.35m std) is especially problematic**: with `goal_reach_threshold=0.3m`, the goal is shifted by more than the reach threshold on average. The robot literally cannot reliably "reach" a goal that's been randomly displaced by more than the success radius.

### 6.2 Reduce Goal Noise for Training

Even with Realistic, 0.15m goal noise is aggressive for initial training. Recommend:
- Phase 1 (basic navigation): **0.0** (no goal noise)
- Phase 2 (fine-tuning): **0.05m**
- Phase 3 (robustness): **0.15m**
- Deployment prep: **0.25m** (per Phase 5 VLM integration plan)

### 6.3 Training Environment Selection

| Phase | Environment | Envs | Purpose |
|-------|-------------|------|---------|
| 1 | `Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0` | 64 | Learn with DeFM features + Realistic noise |
| 2 | `Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0` | 64 | Same env, curriculum advances naturally |
| 3 | `Isaac-Strafer-Nav-Robust-ProcRoom-Depth-v0` | 64 | Harden with aggressive noise |

Note: NoCam-first training is skipped because ProcRoom obstacles exist that the robot cannot "see" without depth. With DeFM's pretrained encoder, the depth-based policy can learn from step 0 — the CNN feature quality problem is solved.

---

## 7. Code Correctness

### 7.1 `scalar_obs_dim` Hardcoded as 19

In `StraferActorCritic.__init__`, `scalar_obs_dim=19` is a constructor default. If observations change (e.g., adding episode progress, removing a sensor), this must be manually updated. Compute dynamically:

```python
# In __init__, after computing num_actor_obs:
if num_actor_obs > _DEPTH_PIXELS:
    self.scalar_obs_dim = num_actor_obs - _DEPTH_PIXELS
else:
    self.scalar_obs_dim = num_actor_obs
```

### 7.2 `energy_penalty` — Verify Magnitude

TensorBoard confirms `energy_penalty` is non-zero (~-0.05 episode total), which is correct for velocity-controlled joints. The `applied_torque` is the PD controller effort. At the proposed weight of -0.001, this contributes ~-0.005 per episode — negligible regularization as intended.

### 7.3 `action_smoothness_penalty` — Consider L1 Instead of L2

The current L2 penalty (`sum(diff^2)`) amplifies large jumps quadratically. With actions in [-1, 1], a full reversal (diff=2.0) costs 4.0 per dimension vs 0.04 for a small adjustment (diff=0.2) — a 100x ratio. This creates a very strong "don't move at all" gradient.

Consider switching to L1 (`sum(|diff|)`), which penalizes large and small changes more proportionally:

```python
def action_smoothness_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    # ... (same setup code) ...
    action_diff = current_action - env._prev_action
    smoothness_cost = torch.sum(torch.abs(action_diff), dim=-1)  # L1 instead of L2
    env._prev_action = current_action.clone()
    return smoothness_cost
```

With L1 + weight=-0.005, the max per-step penalty is 0.005 × 6.0 = 0.03 (full reversal on all 3 dims). This is a gentle nudge toward smoothness without suppressing movement.

---

## 8. Training Plan

### Phase 1: Validated Reward Shaping (Realistic + ProcRoom + DeFM Depth)

```powershell
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py `
  --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0 `
  --num_envs 64 --headless --max_iterations 5000
```

**Changes applied:**
- Rebalanced reward weights (Section 2.1)
- Sustained collision termination (Section 2.2)
- DeFM-EfficientNet-B0 frozen encoder (Section 3)
- Updated PPO hyperparameters (Section 4)
- Adjusted curriculum (Section 5)
- Goal noise = 0.0 (Section 6.2)
- L1 action smoothness (Section 7.3)

**Success criteria:**
- `goal_reached` episode reward climbs above 1.0 by iteration 1000
- `goal_progress` trends positive
- `action_smoothness` stays below 1.0 in magnitude (not -20)
- Mean episode length increases (robot surviving longer, reaching goals)

**If Phase 1 fails**, debug in isolation:
1. Test with Ideal (no noise) to verify reward shaping works at all
2. Test with NoCam + flat ground to verify basic go-to-goal
3. Add one difficulty factor at a time

### Phase 2: Increase Difficulty via Curriculum

Continue from Phase 1 checkpoint. The curriculum will naturally advance:
- Goal distance expands 1.5m → 4.0m
- Room complexity advances level 0 → 5

```powershell
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py `
  --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0 `
  --num_envs 64 --headless --max_iterations 5000 `
  --resume logs\rsl_rl\strafer_navigation\<phase1>\model_5000.pt
```

**Changes applied:**
- Goal noise = 0.05m

**Success criteria:**
- Robot navigates around furniture at room complexity level 3+
- `goal_reached` maintains positive trend even as difficulty increases
- Collision penalty stays bounded (not increasing with difficulty)

### Phase 3: Robust Hardening

Fine-tune with Robust noise for 1000–2000 iterations. This is short — the policy is already competent, we're just expanding its noise tolerance.

```powershell
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py `
  --env Isaac-Strafer-Nav-Robust-ProcRoom-Depth-v0 `
  --num_envs 64 --headless --max_iterations 2000 `
  --resume logs\rsl_rl\strafer_navigation\<phase2>\model_5000.pt
```

**Changes applied:**
- Robust noise contract (2.5x encoder, 2x depth, sensor failures)
- Goal noise = 0.15m
- Lower learning rate (1e-4) to prevent catastrophic forgetting

**Success criteria:**
- `goal_reached` doesn't collapse (stays above 50% of Phase 2 level)
- Policy handles occasional sensor failures gracefully

### Phase 4: Export & Deploy

```powershell
python Scripts/export_policy.py --checkpoint logs/.../model_best.pt --output model.pt
```

- Export includes empirical normalization layers in the traced graph
- DeFM encoder exports as part of the model (frozen weights included)
- Benchmark on Jetson Orin: target <10ms total (DeFM 3ms + MLP <1ms + overhead)

---

## Summary of All Changes

| Category | File | Change |
|----------|------|--------|
| **Rewards** | `strafer_env_cfg.py` | Rebalance all weights (Section 2.1) |
| **Rewards** | `strafer_env_cfg.py` | `goal_proximity` sigma 0.3 → 0.5 |
| **Terminations** | `terminations.py` | Add `sustained_collision` |
| **Terminations** | `strafer_env_cfg.py` | Register `sustained_collision` in `TerminationsCfg` |
| **Network** | `strafer_network.py` | Replace `DepthEncoder` with frozen DeFM + projection |
| **Network** | `strafer_network.py` | Dynamic `scalar_obs_dim` computation |
| **PPO** | `rsl_rl_ppo_cfg.py` | Update depth config (Section 4) |
| **Curriculum** | `strafer_env_cfg.py` | Adjust goal/room curriculum params (Section 5) |
| **Events** | `strafer_env_cfg.py` | Remove/reduce goal noise for initial training |
| **Rewards** | `rewards.py` | L1 action smoothness (optional, Section 7.3) |

---

## References

- [DeFM: Learning Foundation Representations from Depth for Robotics](https://arxiv.org/abs/2601.18923) — ETH Zurich, Jan 2026
- [DeFM GitHub (source + pretrained models)](https://github.com/leggedrobotics/defm)
- [DeFM Hugging Face Models](https://huggingface.co/leggedrobotics/defm)
- [DeFM Project Page](https://de-fm.github.io/)
- [Depth Transfer: Learning to See Like a Simulator for Real-World Drone Navigation](https://arxiv.org/html/2505.12428)
