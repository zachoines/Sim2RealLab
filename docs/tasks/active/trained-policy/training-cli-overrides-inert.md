# Make the training script's CLI overrides reach their consumers

**Type:** bug fix (four overrides in one file + tests)
**Owner:** DGX (`strafer_lab` lane — training entry point)
**Priority:** P2 — none of the four is load-bearing at the shipped defaults, so
nothing recorded is wrong. `--depth_encoder` is the urgent one of the four: it
prints a confirming banner while doing nothing, so it will mislabel the first
experiment that reaches for it.
**Estimate:** S (one session, pure; no GPU, no Kit)
**Branch:** `task/training-cli-overrides-inert`

## Story

As a **DGX operator running an ablation**, I want **a CLI override that prints
`[Override] …` to have changed something**, so that **an arm labelled with a
flag is an arm that ran with it.**

## Context

Surfaced by the audit that shipped with
[`lr-schedule-optimizer`](../../completed/lr-schedule-optimizer.md), which
repaired one instance of this defect class — a value written where the
framework reads it only under a condition that does not hold — and swept the
rest of `train_strafer_navigation.py` for the same shape. Four more instances
came back. All were verified by running the real config pipeline
(`load_cfg_from_registry` → `handle_deprecated_rsl_rl_cfg` → `to_dict` → the
script's exact override line) against `STRAFER_PPO_DEPTH_RUNNER_CFG`, Kit-free.

**1. `--depth_encoder` has never done anything.** The script writes
`agent_dict["policy"]["depth_encoder_type"]`. The encoder is actually selected
from `agent_dict["actor"]["depth_encoder_type"]` and the matching `critic` key,
which stay at `"defm"`. The write does not raise, because
`RslRlOnPolicyRunnerCfg.policy` is `MISSING` and Isaac Lab's `to_dict()`
serializes a `MISSING` field as `{}` rather than omitting it — so
`agent_dict["policy"]` is a real, empty, writable dict that nothing reads. rsl_rl
never reads a `policy` config key at all. Observed directly:

```
BEFORE  actor.depth_encoder_type = defm
d["policy"]["depth_encoder_type"] = "cnn"        # the script's exact line
AFTER   actor.depth_encoder_type = defm
AFTER   d["policy"] = {'depth_encoder_type': 'cnn'}
```

The script then prints `[Override] depth_encoder_type = cnn`. Nothing recorded
is affected — no brief, measurement or run in the history ever passed the flag —
but a banner that confirms an override that did not happen is the same failure
shape the LR schedule had.

Writing `actor` and `critic` unconditionally is not the fix. Only the depth
runner config uses `StraferDepthRNNModelCfg`; the other two use
`RslRlMLPModelCfg` / `RslRlRNNModelCfg`, and those sub-dicts are splatted into
`MLPModel.__init__`, which takes no `**kwargs`. Injecting the key on a
non-depth env would trade a silent no-op for a `TypeError` at model
construction. The override has to establish that the key already exists on both
and fail loudly with a clear message when it does not.

**2. `agent_cfg.device` is never derived from `--device`.** `--device` is an
AppLauncher argument and reaches only the environment, via
`parse_env_cfg` → `cfg.sim.device`. `agent_cfg.device` keeps its class default
`"cuda:0"`, and that value is what constructs the runner — and therefore the
actor, the critic and the rollout storage. At the default `--device cuda:0`
everything coincides. Under `--device cuda:1` or `--device cpu` the environment
produces tensors on one device and the networks live on another.

**3. `agent_cfg.seed` is dead.** rsl_rl 5.0.1 contains no reference to `seed`;
the key survives into `agent_dict` and is read only by the wandb/neptune
`store_config` path, which the default `tensorboard` logger does not take. Run
determinism comes entirely from `env_cfg.seed`, which seeds
`random`/`numpy`/`torch`/`warp` before the scene is built and therefore before
the networks are initialized. This one may be correct to delete rather than
plumb — decide which and say so. Note that deleting the line does not remove the
key: the field defaults to 42, so under `--logger wandb`/`neptune` the uploaded
config would then report 42 regardless of `--seed`.

**4. `clip_actions` is never plumbed into the wrapper.** `RslRlVecEnvWrapper(env)`
is constructed without the argument, so the wrapper takes its own `None` default
and `_modify_action_space` early-returns — action clipping is off regardless of
the agent config. All three runner configs currently leave the field at `None`,
so there is no behavioural difference today; the trap is that the field is
present in the config and reads like a knob. Upstream Isaac Lab passes it
explicitly.

## Acceptance criteria

- [ ] `--depth_encoder` reaches the keys that build the encoder, or the flag is
      removed. Whichever is chosen, an override that cannot take effect does not
      print a confirming banner.
- [ ] `agent_cfg.device` is derived from the launcher's device, so the networks
      and the environment agree under a non-default `--device`.
- [ ] `agent_cfg.seed` is either plumbed to something that reads it or removed,
      with the decision recorded. If removed, the brief states where run
      determinism comes from instead.
- [ ] `clip_actions` is passed to `RslRlVecEnvWrapper`, or the field's inertness
      is documented where someone setting it would read.
- [ ] Pure tests cover each override that survives, asserting on the value the
      framework consumes rather than the value the script writes — the same rule
      the LR schedule's regression test follows.
- [ ] Pure suite and the contract two-file gate hold at their counts.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`source/strafer_lab/scripts/train_strafer_navigation.py`](../../../../source/strafer_lab/scripts/train_strafer_navigation.py)
  — the override block, immediately after `agent_cfg.to_dict()`.
- [`source/strafer_lab/strafer_lab/tasks/navigation/agents/rsl_rl_ppo_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/rsl_rl_ppo_cfg.py)
  — where `actor` / `critic` carry `depth_encoder_type`, and where
  `clip_actions` sits at its default.
- [`source/strafer_lab/tests/policy_tooling/test_train_lr_schedule.py`](../../../../source/strafer_lab/tests/policy_tooling/test_train_lr_schedule.py)
  — the pure-test pattern to follow, including the conftest `sys.path` hook
  that makes `scripts/` importable.

## Out of scope

- The `--lr_schedule` block, repaired in
  [`lr-schedule-optimizer`](../../completed/lr-schedule-optimizer.md).
- `desired_kl=0.01` beside `schedule="fixed"` at three sites in
  `rsl_rl_ppo_cfg.py` — inert config that reads as an active KL target. Owned
  by the in-flight edit to that file's comment block.
- The two non-depth runner configs declaring `obs_groups={"policy": [...]}`.
  rsl_rl resolves the missing `"actor"` set through a deprecation fallback that
  warns `This behavior will be removed in a future version`. Checked against
  rsl-rl-lib 5.4.2, the version the Isaac Lab upgrade moves to: the fallback is
  still present, so this is not an upgrade blocker — it is a future one, and it
  belongs to whoever next edits that file.
- Auxiliary resume state (`DAPGAuxiliary.update_count`, GAIL's discriminator
  and its optimizer are not checkpointed). Superseded by the removal of the
  auxiliary modules.
