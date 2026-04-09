# Migration Plan: `STRAFER_PPO_DEPTH_RUNNER_CFG`

Migrate the depth-training runner from the deprecated `policy` field
(`StraferActorCritic` monolith) to separate `actor`/`critic` model configs
using rsl_rl 5.0.1's new architecture.

## Problem Statement

`STRAFER_PPO_DEPTH_RUNNER_CFG` uses a deprecated `policy` field pointing to a
single `StraferActorCritic` class that bundles actor and critic.  Isaac Lab 3.0
expects separate `actor=` / `critic=` model configs.  Two features of
`StraferActorCritic` don't have stock equivalents:

1. **Beta distribution** — `AffineBeta` maps actions to a true `[-1, 1]` range
   with well-defined entropy.  rsl_rl only ships `GaussianDistribution` and
   `HeteroscedasticGaussianDistribution`.
2. **Pretrained depth encoder** — DeFM/EfficientNet frozen backbone with
   trainable projection.  rsl_rl's built-in `CNN` class only builds random
   conv stacks.

## Key Findings

### rsl_rl 5.0.1 is fully extensible

| Extension point | Mechanism |
|---|---|
| Custom distribution | Subclass `rsl_rl.modules.distribution.Distribution` (13-method interface). Set `distribution_cfg.class_name = "pkg.mod:ClassName"` — resolved via `resolve_callable()`. The deprecation shim passes through any `distribution_cfg` untouched on rsl_rl ≥ 5.0.0. |
| Custom model class | Set `class_name = "pkg.mod:ClassName"` on any model config (MLP/RNN/CNN). PPO instantiates via `resolve_callable()`. Must follow `MLPModel` interface. |
| Shared CNN encoders | `share_cnn_encoders=True` in `RslRlPpoAlgorithmCfg` passes actor's CNN ModuleDict to critic. Not mandatory. |
| obs_groups routing | `resolve_obs_groups()` auto-fills `"actor"` → `["policy"]` from the fallback, so existing `obs_groups={"policy": ..., "critic": ...}` just works. |

### The new actor/critic interface contract

PPO calls `forward(obs, masks, hidden_state, stochastic_output)` instead of
`act()`/`evaluate()`.  Distribution properties are `output_mean`, `output_std`,
`output_entropy`, `output_distribution_params`, `get_output_log_prob()`,
`get_kl_divergence()` — all delegated to `self.distribution` automatically by
`MLPModel`.  No custom plumbing needed.

## Architecture

### Current (deprecated)

```
StraferActorCritic (single class)
├── actor path:  obs → normalize → encode_depth → GRU → MLP → AffineBeta → actions
├── critic path: obs → normalize → encode_depth → GRU → MLP → value
├── act() / evaluate() / act_inference()
└── action_mean / action_std / entropy / get_actions_log_prob()
```

### Target (new)

```
Actor: StraferDepthRNNModel (extends RNNModel)
├── get_latent: obs_groups → concat → split scalar/depth → encode_depth → concat → normalize → RNN
├── forward: get_latent → MLP → AffineBetaDistribution → actions
└── output_* properties inherited from MLPModel → delegated to distribution

Critic: StraferDepthRNNModel (extends RNNModel, no distribution)
├── get_latent: obs_groups → concat → split scalar/depth → encode_depth → concat → normalize → RNN
└── forward: get_latent → MLP → value scalar
```

Actor and critic are **separate model instances** with separate depth encoders,
RNNs, and MLPs — matching rsl_rl's expected architecture.

## New Files

### 1. `strafer_lab/tasks/navigation/agents/distributions.py`

**`AffineBetaDistribution(rsl_rl.modules.distribution.Distribution)`**

Implements the full 13-method Distribution interface:

| Method/Property | Implementation |
|---|---|
| `input_dim` | `[2, output_dim]` — MLP outputs `2 × num_actions` flat logits |
| `update(mlp_output)` | Reshape to `(B, 2, A)`, softplus + min_concentration → α, β concentrations, create internal `AffineBeta` |
| `sample()` | `AffineBeta.sample()` — reparameterized rsample mapped to `[-1, 1]` |
| `deterministic_output(mlp_output)` | Parse → compute `α/(α+β) * 2 - 1` (Beta mean on `[-1, 1]`) |
| `mean` | `AffineBeta.mean` |
| `std` | `AffineBeta.stddev` |
| `entropy` | `AffineBeta.entropy().sum(-1)` |
| `params` | `(concentration1, concentration0)` — stored for KL |
| `log_prob(outputs)` | `AffineBeta.log_prob(outputs).sum(-1)` |
| `kl_divergence(old, new)` | `torch.distributions.kl_divergence(Beta_old, Beta_new).sum(-1)` |
| `init_mlp_weights(mlp)` | Zero last-layer weights, set bias to `inverse_softplus(base_concentration - min_concentration)` |
| `as_deterministic_output_module()` | Returns `nn.Module` that computes Beta mean from raw logits |

**`BetaDistributionCfg`** (configclass):

```python
@configclass
class BetaDistributionCfg(RslRlMLPModelCfg.DistributionCfg):
    class_name: str = "strafer_lab.tasks.navigation.agents.distributions:AffineBetaDistribution"
    init_std: float = MISSING
    min_concentration: float = 0.2
```

Constructor kwargs after `class_name` is popped: `(output_dim, init_std=..., min_concentration=...)`.

### 2. `strafer_lab/tasks/navigation/agents/depth_rnn_model.py`

**`StraferDepthRNNModel(rsl_rl.models.RNNModel)`**

Extends `RNNModel` to add depth encoding between obs concatenation and the RNN.

**Constructor** — accepts all `RNNModel` kwargs plus:

| Param | Type | Default |
|---|---|---|
| `depth_encoder_type` | `str` | `"defm"` |
| `defm_model_name` | `str` | `"efficientnet_b0"` |
| `depth_embedding_dim` | `int` | `128` |
| `depth_obs_dim` | `int` | `4800` |

Stores depth config attributes *before* calling `super().__init__()` (which
triggers `_get_obs_dim()`).  Creates depth encoder *after* `super().__init__()`.

**Overridden methods:**

| Method | Behavior |
|---|---|
| `_get_obs_dim()` | Calls `super()._get_obs_dim()` → gets raw dim → if `raw_dim > depth_obs_dim`: returns `scalar_dim + depth_embedding_dim`. This makes the normalizer, RNN, and MLP all see the compressed dimension. |
| `get_latent(obs, masks, hidden_state)` | 1. Concat obs groups. 2. Split scalar/depth. 3. Encode depth (handles 2D and 3D from recurrent mini-batches). 4. Concat scalar + depth embedding. 5. Normalize. 6. RNN. |
| `update_normalization(obs)` | Concat obs → split → encode depth (no grad) → concat → update normalizer stats on compressed representation. |

When depth is absent (`raw_dim ≤ depth_obs_dim`), all methods fall through to
`RNNModel` behavior unchanged.

**`StraferDepthRNNModelCfg(RslRlRNNModelCfg)`** (configclass):

```python
@configclass
class StraferDepthRNNModelCfg(RslRlRNNModelCfg):
    class_name: str = "strafer_lab.tasks.navigation.agents.depth_rnn_model:StraferDepthRNNModel"
    depth_encoder_type: str = "defm"
    defm_model_name: str = "efficientnet_b0"
    depth_embedding_dim: int = 128
    depth_obs_dim: int = 4800
```

## Modified Files

### 3. `strafer_lab/tasks/navigation/agents/rsl_rl_ppo_cfg.py`

Replace the deprecated `policy=` with separate `actor=`/`critic=`:

```python
from .distributions import BetaDistributionCfg
from .depth_rnn_model import StraferDepthRNNModelCfg

STRAFER_PPO_DEPTH_RUNNER_CFG = RslRlOnPolicyRunnerCfg(
    ...
    obs_groups={"actor": ["policy"], "critic": ["critic"]},
    actor=StraferDepthRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=BetaDistributionCfg(
            init_std=0.3,
            min_concentration=0.2,
        ),
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
        depth_encoder_type="defm",
        defm_model_name="efficientnet_b0",
        depth_embedding_dim=128,
    ),
    critic=StraferDepthRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
        depth_encoder_type="defm",
        defm_model_name="efficientnet_b0",
        depth_embedding_dim=128,
    ),
    algorithm=RslRlPpoAlgorithmCfg(...),  # unchanged
)
```

Remove imports: `RslRlPpoActorCriticRecurrentCfg`.

Move `empirical_normalization=True` from runner-level to per-model
`obs_normalization=True` (rsl_rl 5.0 pattern).

### 4. `strafer_lab/tasks/navigation/agents/strafer_network.py`

- Keep `StraferActorCritic` temporarily for checkpoint loading / rollback.
- Keep `AffineBeta`, `DepthEncoder`, `DeFMDepthEncoder`, `SpatialSoftArgmax`,
  `_create_depth_encoder()` — reused by `StraferDepthRNNModel`.
- Add deprecation notice on `StraferActorCritic`.

### 5. `docs/DGX_SPARK_SETUP.md`

Update TODO #1 status from "cannot migrate" to completed.

## What We Keep vs. Discard

| From `StraferActorCritic` | Disposition |
|---|---|
| `AffineBeta` distribution | → Wrapped in `AffineBetaDistribution(Distribution)` |
| `DepthEncoder` (CNN) | → Reused by `StraferDepthRNNModel` |
| `DeFMDepthEncoder` | → Reused by `StraferDepthRNNModel` |
| `SpatialSoftArgmax` | → Reused by `DepthEncoder` |
| `_create_depth_encoder()` | → Reused |
| Concentration parameterization | → Moved into `AffineBetaDistribution.update()` |
| Beta weight init | → Moved into `AffineBetaDistribution.init_mlp_weights()` |
| `act()` / `evaluate()` / `act_inference()` | → **Discarded** — replaced by `forward(stochastic_output=)` |
| `action_mean` / `action_std` / `entropy` | → **Discarded** — replaced by `output_*` delegation in `MLPModel` |
| `_get_actor_obs()` / `_get_critic_obs()` | → **Discarded** — handled by `MLPModel.get_latent()` obs_groups mechanism |
| Manual obs normalization setup | → **Discarded** — handled by `MLPModel.__init__` |
| Dual encoder (actor + critic in one class) | → **Discarded** — separate model instances each get their own encoder |

## Implementation Phases

### Phase 1: `AffineBetaDistribution` (standalone, testable immediately)

Create `distributions.py` with the distribution + config.  **Test with NoCam
configs** by temporarily swapping `GaussianDistributionCfg` → `BetaDistributionCfg`
in `STRAFER_PPO_RUNNER_CFG`.  This validates the distribution interface without
any depth encoder complexity.

**Validation:** Existing 353 tests pass.  NoCam smoke-test trains for ~50
iterations without NaN/crash.

### Phase 2: `StraferDepthRNNModel` with CNN encoder (MVP)

Create `depth_rnn_model.py`.  Start with `depth_encoder_type="cnn"` only (no
DeFM dependency).  Wire up the config.

**Validation:** Instantiate model with a sample obs TensorDict.  Verify
`forward()` produces correct output shapes.  Run depth env smoke-test.

### Phase 3: DEFM encoder integration

Add `depth_encoder_type="defm"` support — reuses existing `DeFMDepthEncoder`
from `strafer_network.py`.

**Validation:** Depth env smoke-test with DEFM.  Compare initial loss curves
with the old `StraferActorCritic` under the deprecated `policy` config.

### Phase 4: Config migration

Update `STRAFER_PPO_DEPTH_RUNNER_CFG` to use `actor=`/`critic=` with the
new configs.  Remove `policy=` field.  Update docs.

**Validation:** Full 353+ test suite.  Side-by-side training comparison against
a checkpoint from the old config to verify equivalent behavior.

### Phase 5: Cleanup

- Add `@deprecated` notice on `StraferActorCritic`
- Update `DGX_SPARK_SETUP.md` TODO #1
- Remove `RslRlPpoActorCriticRecurrentCfg` import

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Normalization mismatch — old normalizer runs on raw `4819`-dim obs; new normalizer runs on compressed `(scalar + depth_embed)`-dim obs | Verify convergence matches by training both configs for 500 iterations and comparing reward curves. The compressed normalization is architecturally cleaner (normalizes what the RNN actually sees). |
| `resolve_callable()` import path sensitivity | Use fully qualified `"strafer_lab.tasks.navigation.agents.distributions:AffineBetaDistribution"` with colon separator. Verify the package is importable from the training entry point. |
| Broken checkpoint loading | Old `StraferActorCritic` checkpoints won't load into the new separated model. This is expected — retraining is required. Keep the old class for loading legacy checkpoints if needed. |
| `rsl_rl` `MLP` class handling of list `output_dim` | `HeteroscedasticGaussianDistribution` already uses `input_dim=[2, output_dim]` and works fine, so this pattern is proven. |
| DeFM hub load during model construction | Already handled by `_create_depth_encoder()`'s try/except fallback to CNN. |

## NoCam Configs: Optional Beta Upgrade

The NoCam MLP and LSTM configs currently use `GaussianDistributionCfg`.
After Phase 1, these can optionally be switched to `BetaDistributionCfg`
for bounded `[-1, 1]` actions with proper entropy.  This is independent of
the depth encoder work and can be done at any time.
