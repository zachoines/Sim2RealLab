# Share a single encoder noise sample between `wheel_encoder_velocities` and `body_velocity_xy`

**Type:** refactor
**Owner:** DGX (`strafer_lab` env config + `mdp/observations.py`)
**Priority:** P2 — picks up after
[`observation-contract-cleanup`](observation-contract-cleanup.md) ships.
That brief lands the primary fix (encoder-derived FK replaces
`root_lin_vel_b`) but cannot honor its own acceptance criterion #3
("`body_velocity_xy` carries the *same* encoder noise sample as
`wheel_encoder_velocities`") without touching the obs-manager wiring
and the policy/critic obs split — both flagged out of scope by that
brief. This brief is that follow-up.
**Estimate:** M (~2–3 days: per-tick env cache for noised encoder ticks,
policy-vs-critic obs-function split so the critic stays clean, env_cfg
plumbing across DEPTH/NOCAM/Robust groups, tests).
**Branch:** task/encoder-noise-shared-sample

## Story

As a **DGX operator training a DEPTH PPO checkpoint for real-robot
deployment**, I want **`wheel_encoder_velocities` and `body_velocity_xy`
in the policy observation group to be computed from a single noised
encoder-ticks tensor per tick**, so that **the policy at training time
sees the same correlated-noise structure the real robot exhibits
(encoder noise → odom noise → body velocity noise is one physical
noise channel, not two independent ones)**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [observation-contract-cleanup.md](../../completed/observation-contract-cleanup.md) —
  predecessor that landed the FK rewrite and documented this gap
  honestly in `body_velocity_xy`'s docstring. **Move the path to
  `completed/` after that brief ships.**
- [inference-package.md](inference-package.md) — Phase 2's NOCAM-fields
  obs-parity acceptance becomes tighter once the policy sees the same
  noise structure the inference node sees.

## Context

### The gap as it stands today

After [`observation-contract-cleanup`](../../completed/observation-contract-cleanup.md):

- [`wheel_encoder_velocities(env)`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
  returns raw ticks/s. The PolicyCfg obs term applies
  `noise=_REAL_ENCODER_NOISE` via Isaac Lab's obs-manager noise
  hook, AFTER the function returns. Sample call it `N_A`.
- `body_velocity_xy(env)` calls `wheel_encoder_velocities(env)`
  internally to get raw ticks, then runs the inverse mecanum FK.
  Its obs term has no `noise=` config. The policy sees clean FK
  over raw ticks; no encoder noise propagates.

So today the policy sees:
- `wheel_encoder_velocities`: raw_ticks + N_A
- `body_velocity_xy`: FK(raw_ticks)

The real robot sees:
- `/strafer/joint_states.velocity`: real_ticks (electrical encoder
  noise built in)
- `/strafer/odom`: FK(real_ticks) — driven by the SAME noisy ticks

This is the correlation gap: at training, encoder noise and body-velocity
noise are independent; at deployment they are the same physical noise.
A policy trained on independent channels learns a noise-cancelling
behavior that breaks when the two channels suddenly correlate at
deployment.

### Why the simple "apply noise inside `wheel_encoder_velocities`" fix breaks the critic

`ObsCfg_NoCam_Realistic` and `ObsCfg_Real_Depth` both define a
**critic group** with `enable_corruption = False` and no `noise=` on
any term — the asymmetric actor-critic convention requires the critic
to see clean privileged state.

If encoder noise is applied INSIDE `wheel_encoder_velocities` (so it
naturally flows through to `body_velocity_xy`), the critic also gets
noised encoders. That breaks the convention.

The fix is a policy-vs-critic obs-function split: one noised function
for the policy group, one clean function for the critic group.

### Why a per-tick env cache, not duplicated work

Once a "noised encoder ticks" function exists for the policy group,
both `noisy_wheel_encoder_velocities` and `noisy_body_velocity_xy`
need to read from a single noised tensor — otherwise each term draws
an independent sample and we are back to two noise channels.

The clean pattern is a per-tick cache on `env`:

```python
def _get_or_compute_noisy_encoder_ticks(env) -> torch.Tensor:
    """Per-tick cached noisy encoder ticks. Both policy-side obs functions
    read from this so they share a single noise sample per tick."""
    step = int(env.common_step_counter)
    cache = getattr(env, "_noisy_encoder_ticks_cache", None)
    if cache is None or cache["step"] != step:
        raw = wheel_encoder_velocities(env)
        noisy = _encoder_noise_model(env)(raw)
        env._noisy_encoder_ticks_cache = {"step": step, "value": noisy}
    return env._noisy_encoder_ticks_cache["value"]
```

The cache is invalidated by step-counter change; one noise draw per
tick is shared across all policy-side terms that descend from
encoders.

## Approach

### Phase 1 — Policy/critic obs-function split

In
[`source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py),
add policy-side noisy variants while leaving the existing clean
functions intact for the critic groups:

- `wheel_encoder_velocities_noisy(env)` — calls the per-tick
  noised-ticks helper, returns noised ticks.
- `body_velocity_xy_noisy(env)` — calls the per-tick noised-ticks
  helper, runs inverse mecanum FK, returns `(vx, vy)` derived from
  the same sample.

The existing `wheel_encoder_velocities` and `body_velocity_xy` stay
as-is and continue to feed the critic groups.

### Phase 2 — Per-tick noise cache + noise-model accessor

The cache needs the encoder noise model instance. Two options:

A. **Construct on first use** from the env's `cfg` chain (walk to
   the obs group's `wheel_encoder_velocities.noise` and clone it).
   Brittle to env-cfg changes; rejected.

B. **Attach the noise model at env-setup time** via a one-shot
   event term (e.g. `mode="startup"`) that constructs an
   `EncoderNoiseModel` from the contract and stores it on
   `env._encoder_noise_model`. The cache reads from there.

Pick B. Add a startup event in the Realistic / Robust event configs
that constructs the noise model and stashes it on `env`. IDEAL
contracts skip the event; the cache helper falls back to identity
when `_encoder_noise_model` is absent.

### Phase 3 — Env_cfg rewiring

Across the three Realistic / Robust obs config classes in
[`strafer_env_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py):

| Group | obs term | func today | func after this brief |
|---|---|---|---|
| `ObsCfg_NoCam_Realistic.PolicyCfg` | `wheel_encoder_velocities` | `mdp.wheel_encoder_velocities` (+ `noise=`) | `mdp.wheel_encoder_velocities_noisy` (no `noise=`) |
| `ObsCfg_NoCam_Realistic.PolicyCfg` | `body_velocity_xy` | `mdp.body_velocity_xy` | `mdp.body_velocity_xy_noisy` |
| `ObsCfg_NoCam_Realistic.CriticCfg` | both | unchanged (clean) | unchanged (clean) |
| `ObsCfg_Real_Depth.PolicyCfg` | both | same upgrade | same upgrade |
| `ObsCfg_Full_Robust.PolicyCfg` | both | same upgrade (with ROBUST noise) | same upgrade |
| `ObsCfg_Full_Robust.CriticCfg` | both | unchanged (clean) | unchanged (clean) |

Remove the `noise=_REAL_ENCODER_NOISE` / `noise=_ROBUST_ENCODER_NOISE`
from the policy-side `wheel_encoder_velocities` terms — the noise is
now inside the function, applied via the cache.

### Phase 4 — Tests

Extend
[`source/strafer_lab/tests/test_obs_contract_parity.py`](../../../../source/strafer_lab/tests/test_obs_contract_parity.py)
or add a sibling, exercising the new functions against a stubbed env:

1. **Same-sample invariant.** Two same-tick calls to
   `wheel_encoder_velocities_noisy(env)` return byte-identical
   tensors (cache hit).
2. **Correlation invariant.** `body_velocity_xy_noisy(env)` exactly
   equals `FK(wheel_encoder_velocities_noisy(env))` for the same
   `env.common_step_counter` value (read same cache).
3. **Cache invalidation.** Advancing `env.common_step_counter` produces
   a fresh draw.
4. **Critic-clean.** The original `wheel_encoder_velocities(env)` and
   `body_velocity_xy(env)` remain unnoised regardless of cache state.
5. **No double-noise.** With the obs-term `noise=` removed in env_cfg,
   running the obs assembly under REAL contract does not double-apply
   encoder noise (assert variance against expected, not 2× expected).

### Phase 5 — Resume-train DEPTH baseline against the corrected obs

After Phase 4 passes, resume training from the most recent converged
DEPTH ProcRoom checkpoint for 10–15% of the original training-iter
budget against the updated obs contract. The expected delta is
small (noise is now correlated where it wasn't before — slightly less
information for the policy, slightly more sim-to-real fidelity), but
the brief is gated on a converged checkpoint that reflects the new
obs structure.

Comparative evaluation on the standard ProcRoom eval set:
- Baseline-against-baseline: success rate ≈ same
- Updated-against-baseline at sim-in-the-loop with the deployment
  obs chain: should not regress vs. baseline; ideally tightens
  oscillation around goal.

## Acceptance criteria

### Code

- [ ] `wheel_encoder_velocities_noisy(env)` and
      `body_velocity_xy_noisy(env)` exist in
      [`observations.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
      and read from a shared per-tick cache.
- [ ] `_get_or_compute_noisy_encoder_ticks(env)` cache helper
      invalidates on `env.common_step_counter` change.
- [ ] Existing `wheel_encoder_velocities` / `body_velocity_xy`
      remain untouched (critic groups stay clean).
- [ ] All Realistic / Robust PolicyCfg groups in
      [`strafer_env_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py)
      swapped to the `_noisy` variants; obs-term `noise=` on
      `wheel_encoder_velocities` removed for those groups.
- [ ] Startup event term constructs `_encoder_noise_model` on env
      from the active contract.

### Tests

- [ ] `python -m pytest source/strafer_lab/tests/test_obs_contract_parity.py`
      passes — covers same-sample, correlation, cache-invalidation,
      critic-clean, and no-double-noise.

### Training validation

- [ ] DEPTH resume-train completes against the updated obs contract;
      checkpoint saved under
      `logs/rsl_rl/strafer_navigation/depth_shared_noise_v1/`.
- [ ] Eval-set success rate within 5% of baseline at original-DR eval
      (no regression from the correlation tightening).

### Cross-brief consistency

- [ ] `body_velocity_xy`'s docstring is updated to remove the
      "noise samples are independent" caveat, with a one-line note
      that the `_noisy` variant is the policy-facing function.
- [ ] [`observation-contract-cleanup.md`](../../completed/observation-contract-cleanup.md)
      gets a one-line "follow-up shipped" note referencing this brief.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
  — `wheel_encoder_velocities` and `body_velocity_xy` current
  implementations.
- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py)
  — `EncoderNoiseModel` + `EncoderNoiseModelCfg`. The same class is
  reused inside the noisy-ticks helper.
- [`source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py)
  — `get_encoder_noise(contract)` returns the `EncoderNoiseModelCfg`
  to instantiate at startup.
- [`source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py)
  — `ObsCfg_NoCam_Realistic`, `ObsCfg_Real_Depth`, `ObsCfg_Full_Robust`
  are the groups that need the policy-side swap.
- Isaac Lab `ObservationManager` source — how `noise=` is applied
  AFTER the obs function returns; this brief works around that
  by moving noise INTO the function for the policy group only.

## Out of scope

- **Sharing noise across non-encoder obs terms** (IMU, depth,
  goal). Each has its own noise model; sharing samples there would
  require the same cache pattern per modality, but those terms are
  not derivable from each other so the correlation gap is much
  smaller. File separately if a future audit shows it matters.
- **Critic-side noise.** The asymmetric actor-critic convention
  explicitly wants the critic clean; this brief preserves that.
- **Redesigning the obs-term `noise=` mechanism upstream.** Isaac
  Lab's per-term-independent noise sampling is the framework
  default; we work around it for the encoder chain only, not at
  framework level.
- **NOCAM_SUBGOAL group rewiring.** That group is owned by
  [`subgoal-env`](subgoal-env.md); apply the same pattern there
  when that brief lands.
